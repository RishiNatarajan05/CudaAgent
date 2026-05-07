"""CUDA-aware chunking — kernels and device functions stay whole."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Literal, Optional

from .cuda_parser import ParsedFile

ChunkType = Literal["kernel", "device_func", "host_func", "header", "misc"]


@dataclass
class CudaChunk:
    chunk_id: str
    chunk_type: ChunkType
    name: str
    filepath: str
    rel_path: str
    start_line: int
    end_line: int
    text: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _context_prefix(file_text: str, start_line: int, n: int = 10) -> str:
    lines = file_text.splitlines()
    lo = max(0, start_line - 1 - n)
    hi = max(0, start_line - 1)
    if lo >= hi:
        return ""
    return "\n".join(lines[lo:hi])


def chunk_parsed_file(parsed: ParsedFile, repo_root: Path) -> List[CudaChunk]:
    out: List[CudaChunk] = []
    path = Path(parsed.filepath)
    rel = str(path.relative_to(repo_root)) if repo_root in path.parents or path == repo_root else str(path)
    try:
        file_text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        file_text = ""

    base = rel.replace("/", "__")

    for k in parsed.kernels:
        ctx = _context_prefix(file_text, k.start_line, 10)
        body = (ctx + "\n" + k.body) if ctx else k.body
        out.append(CudaChunk(
            chunk_id=f"{base}::kernel::{k.name}::{k.start_line}",
            chunk_type="kernel",
            name=k.name,
            filepath=parsed.filepath,
            rel_path=rel,
            start_line=k.start_line,
            end_line=k.end_line,
            text=body,
            metadata={
                "parameters": k.parameters,
                "launch_bounds": k.launch_bounds,
                "uses_shared_memory": k.uses_shared_memory,
                "shared_memory_declarations": k.shared_memory_declarations,
                "syncthreads_count": k.syncthreads_count,
                "has_atomic_ops": k.has_atomic_ops,
                "estimated_register_pressure": k.estimated_register_pressure,
                "detected_patterns": list(k.detected_patterns),
            },
        ))

    for d in parsed.device_functions:
        ctx = _context_prefix(file_text, d.start_line, 5)
        body = (ctx + "\n" + d.body) if ctx else d.body
        out.append(CudaChunk(
            chunk_id=f"{base}::device::{d.name}::{d.start_line}",
            chunk_type="device_func",
            name=d.name,
            filepath=parsed.filepath,
            rel_path=rel,
            start_line=d.start_line,
            end_line=d.end_line,
            text=body,
            metadata={"parameters": d.parameters, "called_by": d.called_by},
        ))

    for h in parsed.host_functions:
        if not h.contains_kernel_launches and not h.cuda_api_calls:
            continue  # skip plain host funcs to keep index focused
        out.append(CudaChunk(
            chunk_id=f"{base}::host::{h.name}::{h.start_line}",
            chunk_type="host_func",
            name=h.name,
            filepath=parsed.filepath,
            rel_path=rel,
            start_line=h.start_line,
            end_line=h.end_line,
            text=h.body,
            metadata={
                "contains_kernel_launches": h.contains_kernel_launches,
                "kernel_launches": [kl.to_dict() for kl in h.kernel_launches],
                "cuda_api_calls": h.cuda_api_calls,
                "has_loop_with_memcpy": h.has_loop_with_memcpy,
                "has_loop_with_kernel_launch": h.has_loop_with_kernel_launch,
            },
        ))

    if path.suffix.lower() in {".h", ".hpp", ".cuh"} and not out:
        # keep small headers as misc chunks for retrieval
        if file_text:
            out.append(CudaChunk(
                chunk_id=f"{base}::header::0",
                chunk_type="header",
                name=path.name,
                filepath=parsed.filepath,
                rel_path=rel,
                start_line=1,
                end_line=file_text.count("\n") + 1,
                text=file_text[:8000],
                metadata={"includes": parsed.includes, "defines": parsed.defines},
            ))

    return out
