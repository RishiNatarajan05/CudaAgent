"""Two-pass anti-pattern detection.

Pass 1 — fast static structural analysis (during ingestion).
Pass 2 — LLM-assisted deep analysis (on demand).
"""
from __future__ import annotations
import os
import re
from dataclasses import dataclass, asdict, field
from typing import List, Literal, Optional, Union

from ..ingestion.cuda_parser import KernelFunction, HostFunction, ParsedFile
from ..knowledge.antipatterns import ANTIPATTERNS, format_for_prompt

Severity = Literal["critical", "warning", "info"]
Confidence = Literal["high", "medium", "low"]


@dataclass
class DetectedIssue:
    pattern_id: str
    pattern_name: str
    severity: Severity
    confidence: Confidence
    evidence: str
    line_numbers: List[int]
    explanation: str
    kernel_name: Optional[str] = None
    filepath: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Pass 1 — static structural detection
# ---------------------------------------------------------------------------

_THREAD_BRANCH_RE = re.compile(
    r"\bif\s*\([^)]*\b(?:threadIdx|blockIdx)\.[xyz]\b[^)]*\)"
    r"|\bswitch\s*\([^)]*\b(?:threadIdx|blockIdx)\.[xyz]\b[^)]*\)"
)

_STRIDED_INDEX_RE = re.compile(
    r"\[\s*(?:[A-Za-z_]\w*\s*=\s*)?[A-Za-z_]\w*\s*\*\s*[A-Za-z_0-9]+\s*[+\-]?\s*[A-Za-z_0-9]*\s*\]"
)

_AOS_PATTERN_RE = re.compile(r"\b\w+\s*\[\s*(?:i|idx|tid|threadIdx\.x)\s*\]\s*\.\s*\w+")

_BANK_CONFLICT_RE = re.compile(
    r"\b\w+\s*\[\s*threadIdx\.[xyz]\s*\*\s*(?:32|64|128|256)\b"
)

_SHARED_2D_NO_PAD_RE = re.compile(
    r"__shared__\s+\w+\s+\w+\s*\[\s*32\s*\]\s*\[\s*32\s*\]"
)


def _line_offsets(body: str) -> list[int]:
    out, off = [0], 0
    for line in body.split("\n"):
        out.append(off + len(line) + 1)
        off += len(line) + 1
    return out


def _line_at(body: str, char_index: int, base: int) -> int:
    return base + body[:char_index].count("\n")


def _extract_excerpt(body: str, char_index: int, span: int = 80) -> str:
    start = body.rfind("\n", 0, char_index) + 1
    end = body.find("\n", char_index + span)
    if end == -1:
        end = len(body)
    return body[start:end].strip()


def _kernel_static_passes(k: KernelFunction) -> list[DetectedIssue]:
    issues: list[DetectedIssue] = []
    body = k.body
    base = k.start_line

    # WARP_DIVERGENCE — thread-id keyed branches
    for m in _THREAD_BRANCH_RE.finditer(body):
        line = _line_at(body, m.start(), base)
        issues.append(DetectedIssue(
            pattern_id="WARP_DIVERGENCE",
            pattern_name=ANTIPATTERNS["WARP_DIVERGENCE"].name,
            severity="critical",
            confidence="medium",
            evidence=_extract_excerpt(body, m.start()),
            line_numbers=[line],
            explanation="Branch condition depends on thread/block index, may serialize warp execution.",
            kernel_name=k.name,
            filepath=k.filepath,
        ))

    # UNCOALESCED_MEMORY — strided indexing
    for m in _STRIDED_INDEX_RE.finditer(body):
        snippet = m.group(0)
        # filter very common safe patterns like [i*1] or constant 0
        if re.search(r"\*\s*1\s*[\]\+\-]", snippet):
            continue
        line = _line_at(body, m.start(), base)
        issues.append(DetectedIssue(
            pattern_id="UNCOALESCED_MEMORY",
            pattern_name=ANTIPATTERNS["UNCOALESCED_MEMORY"].name,
            severity="critical",
            confidence="low",
            evidence=_extract_excerpt(body, m.start()),
            line_numbers=[line],
            explanation="Strided index expression — may yield non-coalesced global memory access.",
            kernel_name=k.name,
            filepath=k.filepath,
        ))
        break  # only flag once per kernel from regex

    for m in _AOS_PATTERN_RE.finditer(body):
        line = _line_at(body, m.start(), base)
        issues.append(DetectedIssue(
            pattern_id="UNCOALESCED_MEMORY",
            pattern_name=ANTIPATTERNS["UNCOALESCED_MEMORY"].name,
            severity="critical",
            confidence="low",
            evidence=_extract_excerpt(body, m.start()),
            line_numbers=[line],
            explanation="Array-of-structs field access — single-field reads waste bandwidth (consider SoA).",
            kernel_name=k.name,
            filepath=k.filepath,
        ))
        break

    # SHARED_MEMORY_BANK_CONFLICT
    for m in _BANK_CONFLICT_RE.finditer(body):
        line = _line_at(body, m.start(), base)
        issues.append(DetectedIssue(
            pattern_id="SHARED_MEMORY_BANK_CONFLICT",
            pattern_name=ANTIPATTERNS["SHARED_MEMORY_BANK_CONFLICT"].name,
            severity="warning",
            confidence="medium",
            evidence=_extract_excerpt(body, m.start()),
            line_numbers=[line],
            explanation="Index stride is a multiple of 32 — likely bank-conflicting on shared memory.",
            kernel_name=k.name,
            filepath=k.filepath,
        ))
    if k.uses_shared_memory and _SHARED_2D_NO_PAD_RE.search(body):
        m = _SHARED_2D_NO_PAD_RE.search(body)
        line = _line_at(body, m.start(), base)
        issues.append(DetectedIssue(
            pattern_id="SHARED_MEMORY_BANK_CONFLICT",
            pattern_name=ANTIPATTERNS["SHARED_MEMORY_BANK_CONFLICT"].name,
            severity="warning",
            confidence="medium",
            evidence=_extract_excerpt(body, m.start()),
            line_numbers=[line],
            explanation="32x32 shared-memory tile without +1 padding — column accesses will bank-conflict.",
            kernel_name=k.name,
            filepath=k.filepath,
        ))

    # MISSING_RESTRICT
    pointer_params = [p for p in k.parameters if "*" in p]
    if len(pointer_params) >= 2 and not any("__restrict__" in p for p in pointer_params):
        issues.append(DetectedIssue(
            pattern_id="MISSING_RESTRICT",
            pattern_name=ANTIPATTERNS["MISSING_RESTRICT"].name,
            severity="info",
            confidence="high",
            evidence=", ".join(pointer_params),
            line_numbers=[k.start_line],
            explanation=(
                f"Kernel has {len(pointer_params)} pointer parameters but none are __restrict__-qualified — "
                "compiler must assume aliasing."
            ),
            kernel_name=k.name,
            filepath=k.filepath,
        ))

    # OCCUPANCY_LIMITER
    if k.estimated_register_pressure == "high":
        issues.append(DetectedIssue(
            pattern_id="OCCUPANCY_LIMITER",
            pattern_name=ANTIPATTERNS["OCCUPANCY_LIMITER"].name,
            severity="warning",
            confidence="medium",
            evidence="High local-variable count detected",
            line_numbers=[k.start_line],
            explanation=(
                "Heuristic estimates high register pressure. Consider __launch_bounds__ or refactor to "
                "reduce live variables."
            ),
            kernel_name=k.name,
            filepath=k.filepath,
        ))

    # SYNC_OVERUSE — multiple syncs and no shared writes between them is too hard for static; flag if >3 syncs
    if k.syncthreads_count >= 4:
        issues.append(DetectedIssue(
            pattern_id="SYNC_OVERUSE",
            pattern_name=ANTIPATTERNS["SYNC_OVERUSE"].name,
            severity="info",
            confidence="low",
            evidence=f"{k.syncthreads_count} __syncthreads() calls",
            line_numbers=[k.start_line],
            explanation="Many barriers — verify each synchronizes a real shared-memory dependency.",
            kernel_name=k.name,
            filepath=k.filepath,
        ))

    return issues


def _host_static_passes(h: HostFunction, parsed_file: Optional[ParsedFile]) -> list[DetectedIssue]:
    issues: list[DetectedIssue] = []

    if h.has_loop_with_kernel_launch:
        issues.append(DetectedIssue(
            pattern_id="EXCESSIVE_KERNEL_LAUNCHES",
            pattern_name=ANTIPATTERNS["EXCESSIVE_KERNEL_LAUNCHES"].name,
            severity="warning",
            confidence="medium",
            evidence="kernel launch (<<<>>>) inside a loop",
            line_numbers=[h.start_line],
            explanation="Per-iteration kernel launches accrue ~5-10us CPU overhead each — batch the work.",
            filepath=h.filepath,
        ))

    if h.has_loop_with_memcpy:
        issues.append(DetectedIssue(
            pattern_id="HOST_DEVICE_TRANSFER_IN_LOOP",
            pattern_name=ANTIPATTERNS["HOST_DEVICE_TRANSFER_IN_LOOP"].name,
            severity="critical",
            confidence="medium",
            evidence="cudaMemcpy inside a loop",
            line_numbers=[h.start_line],
            explanation="PCIe transfer per iteration — keep data resident on the GPU instead.",
            filepath=h.filepath,
        ))

    return issues


def detect_patterns_static(
    target: Union[KernelFunction, HostFunction],
    parsed_file: Optional[ParsedFile] = None,
) -> list[DetectedIssue]:
    if isinstance(target, KernelFunction):
        return _kernel_static_passes(target)
    if isinstance(target, HostFunction):
        return _host_static_passes(target, parsed_file)
    return []


# ---------------------------------------------------------------------------
# Pass 2 — LLM-assisted deep analysis
# ---------------------------------------------------------------------------

def _llm_client():
    from anthropic import Anthropic
    return Anthropic()


_DEEP_PROMPT = """\
You are a Nvidia GPU performance engineer. Analyze the CUDA kernel below and \
identify performance anti-patterns from the registry. For each finding, give a \
specific line number, the evidence, and a one-paragraph explanation grounded in \
GPU hardware (warps, memory coalescing, banks, occupancy, etc).

Return STRICT JSON with this schema:
{{
  "findings": [
    {{
      "pattern_id": "<one of: WARP_DIVERGENCE | UNCOALESCED_MEMORY | SHARED_MEMORY_BANK_CONFLICT | EXCESSIVE_KERNEL_LAUNCHES | MISSING_RESTRICT | OCCUPANCY_LIMITER | SYNC_OVERUSE | HOST_DEVICE_TRANSFER_IN_LOOP>",
      "line_numbers": [<int>, ...],
      "evidence": "<exact code snippet>",
      "explanation": "<1-2 sentences specific to this kernel>",
      "estimated_impact": "<short, e.g. 'affects ~50% of warps'>"
    }}
  ]
}}

Anti-pattern registry:
{registry}

Kernel name: {kernel_name}
Filepath: {filepath}
Source (line numbers shown):
{source}

Repo context (excerpts of related code):
{context}
"""


def _number_lines(body: str, base: int) -> str:
    out = []
    for i, line in enumerate(body.split("\n")):
        out.append(f"{base + i:5d} | {line}")
    return "\n".join(out)


def analyze_kernel_deep(kernel: KernelFunction, repo_context: str = "") -> list[DetectedIssue]:
    """LLM-assisted deep analysis. Returns high-confidence issues; falls back on static results on failure."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return _kernel_static_passes(kernel)

    try:
        client = _llm_client()
        prompt = _DEEP_PROMPT.format(
            registry=format_for_prompt(),
            kernel_name=kernel.name,
            filepath=kernel.filepath,
            source=_number_lines(kernel.body, kernel.start_line),
            context=(repo_context or "(none)")[:6000],
        )
        msg = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2000,
            system="You produce only valid JSON when asked. No prose outside the JSON object.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        import json
        # Strip code fences if any
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text[text.find("{"): text.rfind("}") + 1] or "{}")
        issues: list[DetectedIssue] = []
        for f in data.get("findings", []):
            pid = f.get("pattern_id", "")
            if pid not in ANTIPATTERNS:
                continue
            ap = ANTIPATTERNS[pid]
            issues.append(DetectedIssue(
                pattern_id=pid,
                pattern_name=ap.name,
                severity=ap.severity,
                confidence="high",
                evidence=f.get("evidence", "")[:400],
                line_numbers=f.get("line_numbers", []) or [kernel.start_line],
                explanation=f.get("explanation", "") + (
                    f" Estimated impact: {f['estimated_impact']}." if f.get("estimated_impact") else ""
                ),
                kernel_name=kernel.name,
                filepath=kernel.filepath,
            ))
        if issues:
            return issues
    except Exception:
        pass

    # Fallback to static
    return _kernel_static_passes(kernel)
