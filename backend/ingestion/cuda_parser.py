"""Tree-sitter-backed CUDA/C++ parser. Falls back to regex if grammar unavailable."""
from __future__ import annotations
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

try:
    import tree_sitter_cpp as tscpp
    from tree_sitter import Language, Parser
    _CPP = Language(tscpp.language())
    _PARSER = Parser(_CPP)
    _TS_OK = True
except Exception:  # pragma: no cover
    _TS_OK = False
    _PARSER = None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class KernelLaunchInfo:
    kernel_name: str
    grid_dim: str
    block_dim: str
    shared_mem_bytes: str = "0"
    stream: str = "0"
    source_line: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class KernelFunction:
    name: str
    parameters: List[str]
    return_type: str
    launch_bounds: Optional[str]
    uses_shared_memory: bool
    shared_memory_declarations: List[str]
    syncthreads_count: int
    has_atomic_ops: bool
    estimated_register_pressure: str  # low|medium|high
    body: str
    start_line: int
    end_line: int
    filepath: str
    detected_patterns: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DeviceFunction:
    name: str
    parameters: List[str]
    body: str
    start_line: int
    end_line: int
    filepath: str
    called_by: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HostFunction:
    name: str
    body: str
    contains_kernel_launches: bool
    kernel_launches: List[KernelLaunchInfo]
    cuda_api_calls: List[str]
    has_loop_with_memcpy: bool
    has_loop_with_kernel_launch: bool
    start_line: int
    end_line: int
    filepath: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["kernel_launches"] = [k.to_dict() for k in self.kernel_launches]
        return d


@dataclass
class ParsedFile:
    filepath: str
    is_cuda: bool
    includes: List[str]
    defines: List[str]
    kernels: List[KernelFunction]
    device_functions: List[DeviceFunction]
    host_functions: List[HostFunction]
    cuda_api_calls: List[str]
    parse_error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "filepath": self.filepath,
            "is_cuda": self.is_cuda,
            "includes": self.includes,
            "defines": self.defines,
            "kernels": [k.to_dict() for k in self.kernels],
            "device_functions": [d.to_dict() for d in self.device_functions],
            "host_functions": [h.to_dict() for h in self.host_functions],
            "cuda_api_calls": self.cuda_api_calls,
            "parse_error": self.parse_error,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CUDA_API_RE = re.compile(r"\bcuda[A-Z][A-Za-z0-9_]+\b")
KERNEL_LAUNCH_RE = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_:<>]*)\s*<<<\s*"
    r"(?P<grid>[^,>]+?)\s*,\s*(?P<block>[^,>]+?)"
    r"(?:\s*,\s*(?P<smem>[^,>]+?))?"
    r"(?:\s*,\s*(?P<stream>[^>]+?))?\s*>>>\s*\("
)
SHARED_DECL_RE = re.compile(r"__shared__\s+[^;]+;")
LAUNCH_BOUNDS_RE = re.compile(r"__launch_bounds__\s*\(([^)]*)\)")
INCLUDE_RE = re.compile(r"^\s*#\s*include\s+[<\"]([^>\"]+)[>\"]", re.MULTILINE)
DEFINE_RE = re.compile(r"^\s*#\s*define\s+(\w+)(?:\s+(.*))?$", re.MULTILINE)
SYNC_RE = re.compile(r"\b__syncthreads\s*\(")
ATOMIC_RE = re.compile(r"\batomic[A-Z][A-Za-z]+\s*\(")
RESTRICT_RE = re.compile(r"__restrict__")


def _estimate_register_pressure(body: str) -> str:
    """Rough heuristic — counts local variable declarations."""
    decl_pattern = re.compile(
        r"\b(?:float|double|int|unsigned|long|short|char|half|bfloat16|__half|__nv_bfloat16|"
        r"float2|float4|int2|int4)\s+\*?\s*[A-Za-z_][A-Za-z0-9_]*\s*[=,;\[]"
    )
    n = len(decl_pattern.findall(body))
    if n < 10:
        return "low"
    if n <= 30:
        return "medium"
    return "high"


def _byte_offsets_to_lines(text: bytes) -> list[int]:
    """Return list mapping byte index -> line number (1-indexed) lazily."""
    # Returns sorted list of newline positions
    positions = [-1]
    for i, b in enumerate(text):
        if b == 0x0A:
            positions.append(i)
    return positions


def _line_of(offset: int, newline_positions: list[int]) -> int:
    # binary search
    import bisect
    return bisect.bisect_right(newline_positions, offset)


# ---------------------------------------------------------------------------
# Tree-sitter walker
# ---------------------------------------------------------------------------

def _node_text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _find_function_definitions(root):
    """Yield function_definition nodes anywhere in the tree."""
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type == "function_definition":
            yield n
        for c in reversed(n.children):
            stack.append(c)


def _classify_function(fn_node, src: bytes) -> tuple[str, str, list[str], str]:
    """
    Returns (kind, name, parameters, return_type) where kind is
    'kernel' (__global__), 'device' (__device__), or 'host'.
    """
    text = _node_text(fn_node, src)
    head = text.split("{", 1)[0]
    is_global = "__global__" in head
    is_device = "__device__" in head and not is_global
    kind = "kernel" if is_global else ("device" if is_device else "host")

    name = ""
    params: list[str] = []
    return_type = "void" if is_global else ""

    # Walk children to find declarator + params
    decl = None
    for c in fn_node.children:
        if c.type == "function_declarator":
            decl = c
            break
        # Sometimes declarator is nested
        for gc in getattr(c, "children", []) or []:
            if gc.type == "function_declarator":
                decl = gc
                break

    if decl is not None:
        for c in decl.children:
            if c.type in ("identifier", "field_identifier", "qualified_identifier"):
                name = _node_text(c, src)
            elif c.type == "parameter_list":
                for pc in c.children:
                    if pc.type == "parameter_declaration":
                        params.append(_node_text(pc, src).strip())

    if not name:
        # Best-effort regex fallback
        m = re.search(r"([A-Za-z_][A-Za-z0-9_:]*)\s*\(", head)
        name = m.group(1) if m else "<anonymous>"

    if not return_type:
        # Strip qualifiers from head text before name
        cleaned = re.sub(r"__(?:global|device|host|forceinline|noinline|launch_bounds__\s*\([^)]*\))__?", "", head)
        cleaned = cleaned.split(name)[0].strip() if name in cleaned else cleaned
        return_type = cleaned.strip() or "void"

    return kind, name, params, return_type


def _extract_launches(body: str, body_start_line: int) -> list[KernelLaunchInfo]:
    out = []
    for m in KERNEL_LAUNCH_RE.finditer(body):
        name = m.group("name").strip()
        # Skip C++ template comparison false-positives (very rough)
        if name in {"if", "while", "for", "return", "switch"}:
            continue
        grid = m.group("grid").strip()
        block = m.group("block").strip()
        smem = (m.group("smem") or "0").strip()
        stream = (m.group("stream") or "0").strip()
        line = body_start_line + body[: m.start()].count("\n")
        out.append(KernelLaunchInfo(name, grid, block, smem, stream, line))
    return out


def _has_loop_with(body: str, needle_re: re.Pattern) -> bool:
    """Cheap heuristic: a loop keyword followed (within ~600 chars) by needle."""
    for m in re.finditer(r"\b(for|while)\s*\(", body):
        window = body[m.start(): m.start() + 800]
        if needle_re.search(window):
            return True
    return False


def parse_file(path: Path) -> ParsedFile:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return ParsedFile(str(path), False, [], [], [], [], [], [], parse_error=str(e))

    src = text.encode("utf-8", errors="replace")
    is_cuda = (
        path.suffix.lower() in {".cu", ".cuh"}
        or "cuda_runtime.h" in text
        or "<cuda.h>" in text
        or "__global__" in text
        or "__device__" in text
        or "<<<" in text
    )

    includes = INCLUDE_RE.findall(text)
    defines = [m.group(1) for m in DEFINE_RE.finditer(text)]
    cuda_api_calls = sorted(set(CUDA_API_RE.findall(text)))

    kernels: list[KernelFunction] = []
    device_funcs: list[DeviceFunction] = []
    host_funcs: list[HostFunction] = []

    if _TS_OK:
        try:
            tree = _PARSER.parse(src)
            newlines = _byte_offsets_to_lines(src)
            for fn in _find_function_definitions(tree.root_node):
                kind, name, params, ret = _classify_function(fn, src)
                body = _node_text(fn, src)
                start_line = _line_of(fn.start_byte, newlines)
                end_line = _line_of(fn.end_byte, newlines)
                if kind == "kernel":
                    lb = LAUNCH_BOUNDS_RE.search(body)
                    smem_decls = SHARED_DECL_RE.findall(body)
                    kernels.append(KernelFunction(
                        name=name,
                        parameters=params,
                        return_type="void",
                        launch_bounds=lb.group(1).strip() if lb else None,
                        uses_shared_memory=bool(smem_decls),
                        shared_memory_declarations=smem_decls,
                        syncthreads_count=len(SYNC_RE.findall(body)),
                        has_atomic_ops=bool(ATOMIC_RE.search(body)),
                        estimated_register_pressure=_estimate_register_pressure(body),
                        body=body,
                        start_line=start_line,
                        end_line=end_line,
                        filepath=str(path),
                    ))
                elif kind == "device":
                    device_funcs.append(DeviceFunction(
                        name=name, parameters=params, body=body,
                        start_line=start_line, end_line=end_line, filepath=str(path),
                    ))
                else:  # host
                    launches = _extract_launches(body, start_line)
                    apis = sorted(set(CUDA_API_RE.findall(body)))
                    host_funcs.append(HostFunction(
                        name=name,
                        body=body,
                        contains_kernel_launches=bool(launches),
                        kernel_launches=launches,
                        cuda_api_calls=apis,
                        has_loop_with_memcpy=_has_loop_with(body, re.compile(r"cudaMemcpy")),
                        has_loop_with_kernel_launch=_has_loop_with(body, re.compile(r"<<<")),
                        start_line=start_line,
                        end_line=end_line,
                        filepath=str(path),
                    ))
            # Resolve device-function call sites
            for d in device_funcs:
                callers = [k.name for k in kernels if re.search(rf"\b{re.escape(d.name)}\s*\(", k.body)]
                d.called_by = callers
        except Exception as e:
            return _regex_fallback_parse(path, text, includes, defines, cuda_api_calls, is_cuda, str(e))
    else:
        return _regex_fallback_parse(path, text, includes, defines, cuda_api_calls, is_cuda, "tree-sitter unavailable")

    return ParsedFile(
        filepath=str(path),
        is_cuda=is_cuda,
        includes=includes,
        defines=defines,
        kernels=kernels,
        device_functions=device_funcs,
        host_functions=host_funcs,
        cuda_api_calls=cuda_api_calls,
    )


# ---------------------------------------------------------------------------
# Regex fallback (when tree-sitter unavailable / errors)
# ---------------------------------------------------------------------------

_FN_HEAD_RE = re.compile(
    r"(?P<head>(?:[\w:\*\&\s]|<[^>]*>|__\w+__(?:\s*\([^)]*\))?)+?)"
    r"\b(?P<name>[A-Za-z_]\w*)\s*\((?P<params>[^)]*)\)\s*\{",
    re.DOTALL,
)


def _match_brace_body(text: str, open_idx: int) -> int:
    depth = 0
    i = open_idx
    in_string = False
    in_char = False
    in_line_comment = False
    in_block_comment = False
    while i < len(text):
        c = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if in_line_comment:
            if c == "\n":
                in_line_comment = False
        elif in_block_comment:
            if c == "*" and nxt == "/":
                in_block_comment = False
                i += 1
        elif in_string:
            if c == "\\":
                i += 1
            elif c == '"':
                in_string = False
        elif in_char:
            if c == "\\":
                i += 1
            elif c == "'":
                in_char = False
        else:
            if c == "/" and nxt == "/":
                in_line_comment = True
                i += 1
            elif c == "/" and nxt == "*":
                in_block_comment = True
                i += 1
            elif c == '"':
                in_string = True
            elif c == "'":
                in_char = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return -1


def _regex_fallback_parse(
    path: Path, text: str, includes: list[str], defines: list[str],
    cuda_api_calls: list[str], is_cuda: bool, err: str,
) -> ParsedFile:
    kernels: list[KernelFunction] = []
    device_funcs: list[DeviceFunction] = []
    host_funcs: list[HostFunction] = []

    for m in _FN_HEAD_RE.finditer(text):
        head = m.group("head")
        name = m.group("name")
        params_raw = m.group("params").strip()
        params = [p.strip() for p in params_raw.split(",") if p.strip()] if params_raw else []
        open_brace = m.end() - 1
        close = _match_brace_body(text, open_brace)
        if close < 0:
            continue
        body = text[m.start(): close + 1]
        start_line = text[: m.start()].count("\n") + 1
        end_line = text[: close].count("\n") + 1
        if "__global__" in head:
            lb = LAUNCH_BOUNDS_RE.search(head)
            smem_decls = SHARED_DECL_RE.findall(body)
            kernels.append(KernelFunction(
                name=name, parameters=params, return_type="void",
                launch_bounds=lb.group(1).strip() if lb else None,
                uses_shared_memory=bool(smem_decls),
                shared_memory_declarations=smem_decls,
                syncthreads_count=len(SYNC_RE.findall(body)),
                has_atomic_ops=bool(ATOMIC_RE.search(body)),
                estimated_register_pressure=_estimate_register_pressure(body),
                body=body, start_line=start_line, end_line=end_line, filepath=str(path),
            ))
        elif "__device__" in head:
            device_funcs.append(DeviceFunction(
                name=name, parameters=params, body=body,
                start_line=start_line, end_line=end_line, filepath=str(path),
            ))
        else:
            launches = _extract_launches(body, start_line)
            apis = sorted(set(CUDA_API_RE.findall(body)))
            host_funcs.append(HostFunction(
                name=name, body=body,
                contains_kernel_launches=bool(launches),
                kernel_launches=launches, cuda_api_calls=apis,
                has_loop_with_memcpy=_has_loop_with(body, re.compile(r"cudaMemcpy")),
                has_loop_with_kernel_launch=_has_loop_with(body, re.compile(r"<<<")),
                start_line=start_line, end_line=end_line, filepath=str(path),
            ))

    for d in device_funcs:
        d.called_by = [k.name for k in kernels if re.search(rf"\b{re.escape(d.name)}\s*\(", k.body)]

    return ParsedFile(
        filepath=str(path), is_cuda=is_cuda, includes=includes, defines=defines,
        kernels=kernels, device_functions=device_funcs, host_functions=host_funcs,
        cuda_api_calls=cuda_api_calls, parse_error=f"fallback: {err}",
    )
