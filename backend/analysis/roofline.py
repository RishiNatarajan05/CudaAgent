"""Static roofline-model estimator for CUDA kernels."""
from __future__ import annotations
import re
from dataclasses import dataclass, asdict, field
from typing import List, Literal


# A100 reference numbers
PEAK_COMPUTE_TFLOPS_FP32 = 19.5
PEAK_COMPUTE_TFLOPS_FP16_TC = 312.0  # tensor core peak
PEAK_BANDWIDTH_TBPS = 2.0
RIDGE_POINT = PEAK_COMPUTE_TFLOPS_FP16_TC * 1000 / (PEAK_BANDWIDTH_TBPS * 1000)  # 156 FLOPs/byte


@dataclass
class RooflineEstimate:
    flops_per_byte: float
    compute_ops_estimate: str
    memory_ops_estimate: str
    bound: Literal["compute", "memory", "unknown"]
    confidence: Literal["high", "medium", "low"]
    assumptions: List[str]
    roofline_data: dict = field(default_factory=dict)
    disclaimer: str = "Static analysis estimate. Use Nvidia Nsight Compute for accurate profiling."

    def to_dict(self) -> dict:
        return asdict(self)


# ---- counters ----------------------------------------------------------------

_FLOAT_TYPE = r"(?:float|double|half|__half|__nv_bfloat16|float2|float4|double2)"
_DECL_RE = re.compile(rf"\b{_FLOAT_TYPE}\b\s+([A-Za-z_]\w*(?:\s*=\s*[^,;]+)?(?:\s*,\s*[A-Za-z_]\w*(?:\s*=\s*[^,;]+)?)*)\s*[;,]")

# arithmetic: count operators inside kernel body (rough)
_ARITH_RE = re.compile(r"(?<![/=!<>+\-*])[+\-*/](?![/=*])")
_FMA_RE = re.compile(r"\b(?:fma|fmaf|__fmaf_rn|__fma_rn|__fmul_rn|__fadd_rn)\b")
_TRANS_RE = re.compile(r"\b(?:sin|cos|tan|asin|acos|atan|exp|expf|log|logf|sqrt|sqrtf|rsqrt|rsqrtf|pow|powf|tanh|tanhf|sinh|cosh)\b")

# memory transactions: array reads/writes against pointer params (rough)
_INDEX_RE = re.compile(r"([A-Za-z_]\w*)\s*\[")
_FLOAT_PTR_PARAM_RE = re.compile(rf"\b{_FLOAT_TYPE}\s*\*\s*(?:__restrict__\s*)?([A-Za-z_]\w*)")
_DOUBLE_PTR_PARAM_RE = re.compile(r"\bdouble\s*\*\s*(?:__restrict__\s*)?([A-Za-z_]\w*)")
_SHARED_RE = re.compile(r"__shared__\s+\w+\s+([A-Za-z_]\w*)")

_LOOP_RE = re.compile(r"\bfor\s*\(\s*(?:int|unsigned)?\s*\w+\s*=\s*(\d+)\s*;\s*\w+\s*<\s*(\d+)\s*;")


def _loop_multiplier(body: str) -> int:
    """Multiply ops by loop bound if both bounds are compile-time integers; else 1."""
    mult = 1
    for m in _LOOP_RE.finditer(body):
        try:
            lo, hi = int(m.group(1)), int(m.group(2))
            if 0 <= lo < hi <= 1024:
                mult *= max(1, hi - lo)
        except Exception:
            continue
    return min(mult, 1024)  # cap to keep estimate bounded


def estimate_roofline(kernel) -> RooflineEstimate:
    """Estimate AI for a KernelFunction-like dict-or-object."""
    body = kernel["body"] if isinstance(kernel, dict) else kernel.body
    params = kernel["parameters"] if isinstance(kernel, dict) else kernel.parameters

    assumptions: list[str] = []

    # Determine pointer params (by type) — to filter memory accesses to global mem
    param_text = ", ".join(params)
    float_ptrs = set(_FLOAT_PTR_PARAM_RE.findall(param_text))
    double_ptrs = set(_DOUBLE_PTR_PARAM_RE.findall(param_text))
    global_ptrs = float_ptrs | double_ptrs
    shared_arrays = set(_SHARED_RE.findall(body))

    # FLOPs
    arith_count = len(_ARITH_RE.findall(body))
    fma_count = len(_FMA_RE.findall(body))
    trans_count = len(_TRANS_RE.findall(body))

    base_flops = arith_count + 2 * fma_count + 20 * trans_count
    loop_mult = _loop_multiplier(body)
    total_flops = base_flops * loop_mult

    # Memory bytes
    bytes_total = 0
    accesses_global = 0
    accesses_shared = 0
    for m in _INDEX_RE.finditer(body):
        ident = m.group(1)
        if ident in global_ptrs:
            accesses_global += 1
            bytes_total += 8 if ident in double_ptrs else 4
        elif ident in shared_arrays:
            accesses_shared += 1
        # local arrays / unknowns — skip

    bytes_total *= loop_mult

    if total_flops == 0 and bytes_total == 0:
        return RooflineEstimate(
            flops_per_byte=0.0,
            compute_ops_estimate="0 (no recognizable arithmetic)",
            memory_ops_estimate="0 (no recognizable global accesses)",
            bound="unknown",
            confidence="low",
            assumptions=["Unable to recognize arithmetic or memory ops in kernel body."],
            roofline_data={
                "estimated_ai": 0.0,
                "peak_compute_tflops": PEAK_COMPUTE_TFLOPS_FP16_TC,
                "peak_bandwidth_tbps": PEAK_BANDWIDTH_TBPS,
                "ridge_point": RIDGE_POINT,
                "is_compute_bound": False,
            },
        )

    if bytes_total == 0:
        ai = float("inf")
        bound = "compute"
    else:
        ai = total_flops / bytes_total
        bound = "compute" if ai > RIDGE_POINT else "memory"

    confidence = "medium" if (accesses_global > 0 and base_flops > 0) else "low"

    assumptions.extend([
        "FP32 reference: each (+,-,*,/) counted as 1 FLOP; fma family counted as 2.",
        "Transcendentals (sin/cos/exp/log) approximated as 20 FLOPs each.",
        f"Loop multiplier applied: {loop_mult}x (compile-time bounds only).",
        "Global memory bytes = 4 (float) or 8 (double) per indexed pointer-param access.",
        "Shared memory accesses excluded — they don't count toward roofline.",
        "Assumes A100 reference: 312 TFLOP/s (FP16 TC), 2.0 TB/s HBM.",
    ])

    return RooflineEstimate(
        flops_per_byte=round(ai, 3) if ai != float("inf") else 1e9,
        compute_ops_estimate=f"{base_flops} flops/iter * {loop_mult}x = {total_flops}",
        memory_ops_estimate=f"{accesses_global} global accesses * {loop_mult}x = {bytes_total} bytes",
        bound=bound,
        confidence=confidence,
        assumptions=assumptions,
        roofline_data={
            "estimated_ai": round(ai, 3) if ai != float("inf") else 1e6,
            "peak_compute_tflops": PEAK_COMPUTE_TFLOPS_FP16_TC,
            "peak_bandwidth_tbps": PEAK_BANDWIDTH_TBPS,
            "ridge_point": round(RIDGE_POINT, 2),
            "is_compute_bound": bound == "compute",
        },
    )
