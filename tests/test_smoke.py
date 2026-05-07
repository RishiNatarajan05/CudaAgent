"""Smoke test: parse → detect → roofline on a synthetic CUDA file."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.ingestion.cuda_parser import parse_file
from backend.analysis.pattern_detector import detect_patterns_static
from backend.analysis.roofline import estimate_roofline


def main():
    sample = ROOT / "tests" / "sample.cu"
    pf = parse_file(sample)
    print(f"parse_error: {pf.parse_error}")
    print(f"is_cuda: {pf.is_cuda}")
    print(f"kernels: {[k.name for k in pf.kernels]}")
    print(f"device_funcs: {[d.name for d in pf.device_functions]}")
    print(f"host_funcs: {[h.name for h in pf.host_functions]}")
    print(f"cuda_api_calls: {pf.cuda_api_calls[:8]}")
    assert "bad_kernel" in [k.name for k in pf.kernels]
    assert "good_kernel" in [k.name for k in pf.kernels]
    assert "tiled_matmul" in [k.name for k in pf.kernels]
    assert "relu" in [d.name for d in pf.device_functions]

    print("\n--- per-kernel detection ---")
    for k in pf.kernels:
        issues = detect_patterns_static(k)
        print(f"{k.name}: shared={k.uses_shared_memory} syncs={k.syncthreads_count} reg={k.estimated_register_pressure}")
        for i in issues:
            print(f"  [{i.severity}/{i.confidence}] {i.pattern_id} L{i.line_numbers}: {i.evidence[:60]}")

    print("\n--- per-host detection ---")
    for h in pf.host_functions:
        issues = detect_patterns_static(h, parsed_file=pf)
        print(f"{h.name}: launches={len(h.kernel_launches)} loop_memcpy={h.has_loop_with_memcpy} loop_launch={h.has_loop_with_kernel_launch}")
        for i in issues:
            print(f"  [{i.severity}] {i.pattern_id}: {i.explanation}")

    print("\n--- roofline ---")
    for k in pf.kernels:
        est = estimate_roofline(k)
        print(f"{k.name}: AI={est.flops_per_byte} bound={est.bound} conf={est.confidence}")
        print(f"   compute={est.compute_ops_estimate}")
        print(f"   memory ={est.memory_ops_estimate}")

    # Assertions on detections
    bad_issues = detect_patterns_static([k for k in pf.kernels if k.name == "bad_kernel"][0])
    pids = {i.pattern_id for i in bad_issues}
    assert "WARP_DIVERGENCE" in pids, f"expected WARP_DIVERGENCE in bad_kernel, got {pids}"
    assert "MISSING_RESTRICT" in pids, f"expected MISSING_RESTRICT in bad_kernel, got {pids}"

    tile_issues = detect_patterns_static([k for k in pf.kernels if k.name == "tiled_matmul"][0])
    tile_pids = {i.pattern_id for i in tile_issues}
    assert "SHARED_MEMORY_BANK_CONFLICT" in tile_pids, f"expected bank conflict in tiled_matmul, got {tile_pids}"

    bad_host = [h for h in pf.host_functions if h.name == "bad_host"][0]
    host_issues = detect_patterns_static(bad_host, parsed_file=pf)
    host_pids = {i.pattern_id for i in host_issues}
    assert "HOST_DEVICE_TRANSFER_IN_LOOP" in host_pids, f"got {host_pids}"
    assert "EXCESSIVE_KERNEL_LAUNCHES" in host_pids, f"got {host_pids}"

    print("\nALL ASSERTIONS PASSED ✓")


if __name__ == "__main__":
    main()
