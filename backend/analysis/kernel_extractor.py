"""Builds focused context packages for deep kernel analysis."""
from __future__ import annotations
import re
from typing import Optional

from ..ingestion.indexer import load_parsed, find_kernel
from ..ingestion.cuda_parser import KernelFunction


def _kernel_obj_from_dict(d: dict) -> KernelFunction:
    return KernelFunction(
        name=d["name"], parameters=d["parameters"], return_type=d.get("return_type", "void"),
        launch_bounds=d.get("launch_bounds"),
        uses_shared_memory=d.get("uses_shared_memory", False),
        shared_memory_declarations=d.get("shared_memory_declarations", []),
        syncthreads_count=d.get("syncthreads_count", 0),
        has_atomic_ops=d.get("has_atomic_ops", False),
        estimated_register_pressure=d.get("estimated_register_pressure", "low"),
        body=d["body"], start_line=d["start_line"], end_line=d["end_line"],
        filepath=d["filepath"],
        detected_patterns=d.get("detected_patterns", []),
    )


def extract_context(repo_id: str, kernel_name: str) -> Optional[dict]:
    """Returns dict: kernel, called_device_functions, launching_host_function, defines, detected_patterns."""
    parsed = load_parsed(repo_id)
    target_pf = None
    target_kernel = None
    for pf in parsed:
        for k in pf["kernels"]:
            if k["name"] == kernel_name:
                target_pf = pf
                target_kernel = k
                break
        if target_kernel:
            break
    if not target_kernel:
        return None

    # Recursively collect device functions called by the kernel
    all_device = []
    for pf in parsed:
        all_device.extend(pf["device_functions"])

    called = []
    seen = set()
    queue = [target_kernel["body"]]
    while queue:
        body = queue.pop()
        for d in all_device:
            if d["name"] in seen:
                continue
            if re.search(rf"\b{re.escape(d['name'])}\s*\(", body):
                seen.add(d["name"])
                called.append(d)
                queue.append(d["body"])

    # Find host function that launches this kernel
    launcher = None
    for pf in parsed:
        for h in pf["host_functions"]:
            for kl in h.get("kernel_launches", []):
                if kl["kernel_name"] == kernel_name:
                    launcher = {"host_function": h, "launch": kl, "filepath": pf["filepath"]}
                    break
            if launcher:
                break
        if launcher:
            break

    return {
        "kernel": target_kernel,
        "filepath": target_pf["filepath"],
        "called_device_functions": called,
        "launcher": launcher,
        "defines": target_pf.get("defines", []),
        "detected_patterns": target_kernel.get("detected_patterns", []),
    }


def context_to_text(ctx: dict, max_chars: int = 6000) -> str:
    parts = []
    if ctx.get("launcher"):
        l = ctx["launcher"]
        parts.append(
            f"// Launching host function ({l['filepath']}):\n"
            f"// launch config: grid={l['launch']['grid_dim']} block={l['launch']['block_dim']} "
            f"smem={l['launch']['shared_mem_bytes']} stream={l['launch']['stream']}\n"
            f"{l['host_function']['body'][:1500]}"
        )
    for d in ctx.get("called_device_functions", []):
        parts.append(f"// __device__ {d['name']} ({d['filepath']}):\n{d['body'][:1200]}")
    if ctx.get("defines"):
        parts.append("// #defines: " + ", ".join(ctx["defines"][:30]))
    out = "\n\n".join(parts)
    return out[:max_chars]
