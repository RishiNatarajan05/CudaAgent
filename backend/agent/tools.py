"""LangGraph tool definitions backed by the ingestion + analysis modules."""
from __future__ import annotations
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from ..ingestion.indexer import (
    REPOS_DIR, load_summary, load_kernels, find_kernel, load_parsed,
)
from ..ingestion.cuda_parser import KernelFunction
from ..retrieval.hybrid import hybrid_search, search_cuda_docs
from ..analysis.pattern_detector import detect_patterns_static, analyze_kernel_deep
from ..analysis.roofline import estimate_roofline as _roofline
from ..analysis.kernel_extractor import extract_context, context_to_text
from ..knowledge.cuda_concepts import CONCEPTS, search as concept_search


def _kernel_obj(d: dict) -> KernelFunction:
    return KernelFunction(
        name=d["name"], parameters=d["parameters"], return_type=d.get("return_type", "void"),
        launch_bounds=d.get("launch_bounds"),
        uses_shared_memory=d.get("uses_shared_memory", False),
        shared_memory_declarations=d.get("shared_memory_declarations", []),
        syncthreads_count=d.get("syncthreads_count", 0),
        has_atomic_ops=d.get("has_atomic_ops", False),
        estimated_register_pressure=d.get("estimated_register_pressure", "low"),
        body=d["body"], start_line=d["start_line"], end_line=d["end_line"],
        filepath=d["filepath"], detected_patterns=d.get("detected_patterns", []),
    )


@tool
def search_codebase(query: str, repo_id: str) -> str:
    """Hybrid vector + BM25 search over an indexed CUDA repo. Kernel chunks are boosted."""
    hits = hybrid_search(repo_id, query, k=8)
    if not hits:
        return "No results."
    out = []
    for h in hits:
        m = h["metadata"] or {}
        chunk_type = m.get("chunk_type", "?")
        name = m.get("name", "?")
        rel = m.get("rel_path", m.get("filepath", "?"))
        sl, el = m.get("start_line", "?"), m.get("end_line", "?")
        detected = m.get("detected_patterns") or "[]"
        preview = h["document"][:600]
        out.append(
            f"--- {chunk_type} `{name}` @ {rel}:{sl}-{el}\n"
            f"detected_patterns: {detected}\n"
            f"{preview}\n"
        )
    return "\n".join(out)


@tool
def search_cuda_documentation(query: str) -> str:
    """Search Nvidia CUDA Programming Guide / Best Practices Guide and the local CUDA knowledge base."""
    hits = search_cuda_docs(query, k=5)
    if not hits:
        return "No documentation results found."
    out = []
    for h in hits:
        m = h["metadata"] or {}
        out.append(
            f"--- source={m.get('source','?')} section={m.get('section','?')} url={m.get('url','')}\n"
            f"{h['document'][:1000]}\n"
        )
    return "\n".join(out)


@tool
def read_kernel(kernel_name: str, repo_id: str) -> str:
    """Read the full source of a kernel by name plus its parsed metadata."""
    k = find_kernel(repo_id, kernel_name)
    if not k:
        return f"Kernel `{kernel_name}` not found in repo {repo_id}."
    lines = []
    lines.append(f"Kernel: {k['name']}")
    lines.append(f"File: {k['filepath']} lines {k['start_line']}-{k['end_line']}")
    lines.append(f"Parameters: {k['parameters']}")
    lines.append(f"launch_bounds: {k.get('launch_bounds')}")
    lines.append(f"uses_shared_memory: {k.get('uses_shared_memory')}")
    lines.append(f"syncthreads_count: {k.get('syncthreads_count')}")
    lines.append(f"has_atomic_ops: {k.get('has_atomic_ops')}")
    lines.append(f"estimated_register_pressure: {k.get('estimated_register_pressure')}")
    lines.append(f"detected_patterns (Pass 1): {k.get('detected_patterns', [])}")
    lines.append("Source:")
    base = k["start_line"]
    for i, line in enumerate(k["body"].split("\n")):
        lines.append(f"{base + i:5d} | {line}")
    return "\n".join(lines)


@tool
def analyze_kernel(kernel_name: str, repo_id: str) -> str:
    """Deep performance analysis of a kernel: anti-patterns, severity, recommended fixes."""
    ctx = extract_context(repo_id, kernel_name)
    if not ctx:
        return f"Kernel `{kernel_name}` not found in repo {repo_id}."
    kobj = _kernel_obj(ctx["kernel"])
    repo_ctx_text = context_to_text(ctx)
    issues = analyze_kernel_deep(kobj, repo_ctx_text)
    sev_counts = Counter(i.severity for i in issues)

    out = [f"# Analysis of `{kernel_name}` ({ctx['filepath']})"]
    out.append(
        f"Kernel summary: {len(kobj.parameters)} params, "
        f"shared_mem={kobj.uses_shared_memory}, syncs={kobj.syncthreads_count}, "
        f"reg_pressure={kobj.estimated_register_pressure}, atomics={kobj.has_atomic_ops}"
    )
    if ctx.get("launcher"):
        l = ctx["launcher"]["launch"]
        out.append(f"Launch config: grid={l['grid_dim']} block={l['block_dim']} smem={l['shared_mem_bytes']}")
    out.append(
        f"Severity: {sev_counts['critical']} critical, "
        f"{sev_counts['warning']} warning, {sev_counts['info']} info"
    )

    if not issues:
        out.append("No anti-patterns detected.")
        return "\n".join(out)

    out.append("\n## Findings")
    for i, issue in enumerate(issues, 1):
        out.append(
            f"\n{i}. [{issue.severity.upper()}/{issue.confidence} confidence] "
            f"{issue.pattern_id} — {issue.pattern_name}"
        )
        out.append(f"   Lines: {issue.line_numbers}")
        if issue.evidence:
            out.append(f"   Evidence: {issue.evidence}")
        out.append(f"   Why: {issue.explanation}")

    # Top fix recommendation = highest-severity issue's antipattern fix
    from ..knowledge.antipatterns import ANTIPATTERNS
    severity_rank = {"critical": 0, "warning": 1, "info": 2}
    top = sorted(issues, key=lambda x: severity_rank.get(x.severity, 9))[0]
    ap = ANTIPATTERNS.get(top.pattern_id)
    if ap:
        out.append(f"\n## Top Recommended Fix: {ap.name}")
        out.append(ap.fix)
        out.append(f"Reference: {ap.nvidia_doc_reference}")

    # Emit issues_data marker for the frontend stream consumer
    issues_payload = [i.to_dict() for i in issues]
    out.append("\n[[ISSUES_DATA]]" + json.dumps(issues_payload))
    return "\n".join(out)


@tool
def estimate_roofline(kernel_name: str, repo_id: str) -> str:
    """Static-analysis roofline estimate (arithmetic intensity, bound, A100 reference)."""
    k = find_kernel(repo_id, kernel_name)
    if not k:
        return f"Kernel `{kernel_name}` not found in repo {repo_id}."
    est = _roofline(k)
    payload = est.to_dict()
    out = [
        f"# Roofline estimate — `{kernel_name}`",
        f"Arithmetic intensity: {est.flops_per_byte} FLOPs/byte",
        f"Bound: {est.bound} (confidence: {est.confidence})",
        f"Compute: {est.compute_ops_estimate}",
        f"Memory:  {est.memory_ops_estimate}",
        f"Roofline (A100 ref): peak_compute={est.roofline_data['peak_compute_tflops']} TFLOP/s, "
        f"peak_bw={est.roofline_data['peak_bandwidth_tbps']} TB/s, "
        f"ridge_point={est.roofline_data['ridge_point']} FLOPs/byte",
        "Assumptions:",
    ]
    out.extend("  - " + a for a in est.assumptions)
    out.append(est.disclaimer)
    out.append("\n[[ROOFLINE_DATA]]" + json.dumps(payload))
    return "\n".join(out)


@tool
def get_repo_cuda_summary(repo_id: str) -> str:
    """High-level GPU performance health report for an indexed repo."""
    s = load_summary(repo_id)
    if not s:
        return f"Repo {repo_id} not indexed."
    sev = s["total_issues_by_severity"]
    out = [
        f"# Repo {s['repo_id']}",
        f"URL: {s['url']}",
        f"Languages: {', '.join(s.get('languages_detected', []))}",
        f"Kernels: {s['total_kernels']}, device functions: {s['total_device_functions']}, "
        f"kernel launch sites: {s['total_kernel_launches']}",
        f"Issues: {sev['critical']} critical, {sev['warning']} warning, {sev['info']} info",
        f"Files with issues: {len(s['files_with_issues'])}",
    ]
    if s["files_with_issues"]:
        out.append("\nTop files with issues:")
        for f in s["files_with_issues"][:15]:
            out.append(f"  - {f}")
    if s.get("cuda_api_calls"):
        out.append("\nMost-used CUDA APIs:")
        for api, n in list(s["cuda_api_calls"].items())[:15]:
            out.append(f"  - {api}: {n}")
    return "\n".join(out)


@tool
def compare_kernels(kernel_a: str, kernel_b: str, repo_id: str) -> str:
    """Side-by-side comparison of two kernels: roofline + anti-patterns + likely-faster verdict."""
    ka = find_kernel(repo_id, kernel_a)
    kb = find_kernel(repo_id, kernel_b)
    if not ka:
        return f"Kernel `{kernel_a}` not found."
    if not kb:
        return f"Kernel `{kernel_b}` not found."
    ra = _roofline(ka)
    rb = _roofline(kb)
    # Run static-pattern detection
    ia = detect_patterns_static(_kernel_obj(ka))
    ib = detect_patterns_static(_kernel_obj(kb))

    def fmt(name, k, r, issues):
        sev = Counter(i.severity for i in issues)
        return [
            f"## {name}: {k['name']} ({k['filepath']}:{k['start_line']})",
            f"  AI: {r.flops_per_byte} FLOPs/byte | bound: {r.bound} | reg_pressure: {k['estimated_register_pressure']}",
            f"  syncs: {k['syncthreads_count']} | shared_mem: {k['uses_shared_memory']}",
            f"  issues: {sev['critical']}C / {sev['warning']}W / {sev['info']}I",
        ]

    out = ["# Kernel comparison"]
    out.extend(fmt("A", ka, ra, ia))
    out.extend(fmt("B", kb, rb, ib))

    # Verdict
    score_a = ra.flops_per_byte + (-3 * sum(1 for i in ia if i.severity == "critical"))
    score_b = rb.flops_per_byte + (-3 * sum(1 for i in ib if i.severity == "critical"))
    verdict = "A" if score_a > score_b else "B"
    out.append(f"\nLikely faster: Kernel {verdict} (heuristic).")
    return "\n".join(out)


@tool
def explain_cuda_concept(concept: str) -> str:
    """Look up a GPU architecture concept in the knowledge base."""
    key = concept.lower().replace(" ", "_").replace("-", "_")
    if key in CONCEPTS:
        c = CONCEPTS[key]
    else:
        results = concept_search(concept)
        c = results[0] if results else None
    if not c:
        return f"No concept found for '{concept}'."
    return (
        f"# {c.name}\n\n"
        f"TL;DR: {c.one_line_summary}\n\n"
        f"{c.detailed_explanation}\n\n"
        f"Common misconception: {c.common_misconception}\n\n"
        f"Architecture note: {c.nvidia_arch_note}"
    )


@tool
def get_file_tree(repo_id: str) -> str:
    """File tree showing only CUDA-relevant files, with kernel counts per .cu file."""
    s = load_summary(repo_id)
    if not s:
        return f"Repo {repo_id} not indexed."
    parsed = load_parsed(repo_id)
    by_dir: dict[str, list[str]] = defaultdict(list)
    src_root = Path(s["src_path"])
    for pf in parsed:
        p = Path(pf["filepath"])
        try:
            rel = p.relative_to(src_root)
        except Exception:
            rel = p
        kcount = len(pf["kernels"])
        annot = f" [{kcount} kernel{'s' if kcount != 1 else ''}]" if kcount else ""
        by_dir[str(rel.parent)].append(f"{rel.name}{annot}")

    out = []
    for d in sorted(by_dir):
        out.append(f"{d}/")
        for f in sorted(by_dir[d]):
            out.append(f"  {f}")
    return "\n".join(out) or "(no CUDA-relevant files)"


ALL_TOOLS = [
    search_codebase,
    search_cuda_documentation,
    read_kernel,
    analyze_kernel,
    estimate_roofline,
    get_repo_cuda_summary,
    compare_kernels,
    explain_cuda_concept,
    get_file_tree,
]
