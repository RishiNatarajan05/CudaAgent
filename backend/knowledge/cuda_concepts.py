"""Structured GPU architecture concept explanations."""
from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass
class CudaConcept:
    name: str
    one_line_summary: str
    detailed_explanation: str
    common_misconception: str
    nvidia_arch_note: str

    def to_dict(self) -> dict:
        return asdict(self)


CONCEPTS: Dict[str, CudaConcept] = {
    "thread_hierarchy": CudaConcept(
        name="Thread Hierarchy: thread → warp → block → grid",
        one_line_summary="GPU work is organized as a grid of blocks, each block is a group of threads scheduled in 32-thread warps.",
        detailed_explanation=(
            "A CUDA kernel launch creates a grid of thread blocks. Each block contains "
            "up to 1024 threads which run on a single SM and can communicate via shared "
            "memory and __syncthreads(). The hardware schedules threads in groups of 32 "
            "called warps — every instruction a warp issues runs in lockstep across its "
            "32 threads (SIMT). Blocks within a grid cannot synchronize with each other "
            "(except via cooperative groups on grid-level kernels)."
        ),
        common_misconception=(
            "Threads within a block do NOT all run simultaneously — only the warps that "
            "the SM has scheduled at any given moment. Many warps can be 'in flight' "
            "while waiting on memory."
        ),
        nvidia_arch_note=(
            "Volta+ supports independent thread scheduling — threads in a warp can have "
            "independent program counters, which changes how divergence and __syncwarp() "
            "behave compared to Pascal and earlier."
        ),
    ),
    "memory_hierarchy": CudaConcept(
        name="Memory Hierarchy: registers → shared → L1 → L2 → global → host",
        one_line_summary="Latency and bandwidth differ by orders of magnitude across GPU memory levels — pick the right one.",
        detailed_explanation=(
            "Registers (~1 cycle, per-thread, ~256 KB/SM total) > shared memory "
            "(~30 cycles, per-block, configurable up to ~228 KB/SM on H100) > L1 cache "
            "(~30 cycles) > L2 cache (~200 cycles, ~50 MB on H100) > global memory "
            "(~400-800 cycles, HBM, hundreds of GB) > host memory (over PCIe, ~16-32 GB/s). "
            "Coalesced global memory access combines 32 thread requests into one or two "
            "transactions; uncoalesced access does the opposite."
        ),
        common_misconception=(
            "Shared memory is NOT 'just like a cache' — it is software-managed scratchpad. "
            "Bank conflicts arise when threads access the same bank, not the same address."
        ),
        nvidia_arch_note=(
            "On Hopper (H100), the L2 is partitioned and supports access policy windows. "
            "Distributed shared memory across thread block clusters is new in H100."
        ),
    ),
    "sm_architecture": CudaConcept(
        name="Streaming Multiprocessor (SM) Architecture",
        one_line_summary="An SM is a GPU's basic compute unit: warp schedulers + dispatchers + CUDA cores + tensor cores + shared mem + register file.",
        detailed_explanation=(
            "An SM holds a fixed pool of resources: a register file (~64K 32-bit "
            "registers on A100), shared memory, warp schedulers (4 on A100), and "
            "execution units (FP32 cores, FP64 cores, tensor cores, special function "
            "units). The SM picks an eligible warp every cycle from the pool of resident "
            "warps and issues an instruction. Resident warp count (occupancy) depends on "
            "registers/thread and shared memory/block."
        ),
        common_misconception=(
            "Higher occupancy is not always better — once you have enough warps to hide "
            "memory latency, more occupancy hurts (less registers per thread)."
        ),
        nvidia_arch_note=(
            "A100 SM: 64 FP32 + 32 FP64 + 4 Tensor Cores per SM. H100 SM: doubled FP32, "
            "4th-gen tensor cores with FP8, plus a tensor memory accelerator (TMA)."
        ),
    ),
    "warp_scheduling": CudaConcept(
        name="Warp Scheduling and Latency Hiding",
        one_line_summary="GPUs hide memory latency by switching between many resident warps, not by caching.",
        detailed_explanation=(
            "When a warp issues a memory load that misses L1/L2, it stalls. Instead of "
            "waiting (like a CPU), the SM's warp scheduler picks another eligible warp "
            "and issues from it. With enough resident warps, the math units stay busy "
            "while waiting threads' loads complete. This is why occupancy and arithmetic "
            "intensity both matter — high AI means less stalling per math op."
        ),
        common_misconception="GPU performance comes from caches like CPUs — false. It comes from massive parallel latency hiding.",
        nvidia_arch_note="Hopper adds asynchronous TMA loads that don't tie up warp slots while loading large tiles.",
    ),
    "occupancy": CudaConcept(
        name="Occupancy",
        one_line_summary="Ratio of resident warps per SM to the hardware maximum (typically 64 warps/SM).",
        detailed_explanation=(
            "Occupancy = resident_warps / max_warps_per_sm. It is limited by the "
            "minimum of three factors: (1) registers — total_registers_per_sm / "
            "(threads_per_block * registers_per_thread); (2) shared memory — "
            "total_shared_per_sm / shared_per_block; (3) blocks/SM hardware cap. "
            "Use --ptxas-options=-v to see register usage, or CUDA Occupancy Calculator."
        ),
        common_misconception=(
            "100% occupancy is the goal — actually no. ~50% is often optimal. What matters "
            "is whether you have enough warps to hide your memory access latency."
        ),
        nvidia_arch_note="A100/H100 max is 64 warps/SM = 2048 threads/SM. Earlier arches differ.",
    ),
    "tensor_cores": CudaConcept(
        name="Tensor Cores",
        one_line_summary="Specialized matrix-multiply-accumulate units that operate on small tiles in one instruction.",
        detailed_explanation=(
            "Tensor cores compute D = A*B + C where A, B, C, D are small matrices "
            "(e.g., 16x16). They activate when you use WMMA, mma.sync PTX, or libraries "
            "like cuBLAS/cuDNN with the right dtypes. Supported dtypes vary by arch: "
            "Volta=FP16, Turing+INT8/4, Ampere+TF32/BF16, Hopper+FP8."
        ),
        common_misconception="Any matmul kernel uses tensor cores — false. Naive __global__ matmul code does NOT; you must use WMMA, CUTLASS, or cuBLAS.",
        nvidia_arch_note="H100 4th-gen tensor cores: ~2x perf vs A100, FP8 support, asynchronous via wgmma instruction.",
    ),
    "cuda_streams": CudaConcept(
        name="CUDA Streams and Concurrency",
        one_line_summary="Streams are queues of GPU work that can execute concurrently with other streams.",
        detailed_explanation=(
            "Operations issued to the same stream execute in order; operations in "
            "different streams can overlap (kernel/copy concurrency, multi-kernel "
            "concurrency). The default stream is synchronizing unless you opt into "
            "per-thread default-stream. Use cudaMemcpyAsync + pinned host memory for "
            "copy-compute overlap."
        ),
        common_misconception="cudaMemcpyAsync from pageable host memory is asynchronous — false. It silently blocks until pinned.",
        nvidia_arch_note="Hopper adds programmatic dependent launch (PDL) for tighter kernel-to-kernel handoff.",
    ),
    "unified_memory": CudaConcept(
        name="Unified Memory and Page Faults",
        one_line_summary="cudaMallocManaged gives one pointer usable on host and device; the driver migrates pages on demand.",
        detailed_explanation=(
            "When the GPU touches a page resident on the host, the GPU page faults, "
            "the driver migrates the page over PCIe/NVLink, and execution resumes. "
            "Convenient but slow if migrations happen on the hot path. Use "
            "cudaMemPrefetchAsync and cudaMemAdvise to control placement."
        ),
        common_misconception="Unified memory makes performance the same as explicit transfers — false; first-touch migration is expensive.",
        nvidia_arch_note="Grace Hopper (GH200) makes UM cheap because of NVLink-C2C between CPU and GPU.",
    ),
}


def get(name: str) -> CudaConcept:
    return CONCEPTS[name]


def all_concepts() -> List[CudaConcept]:
    return list(CONCEPTS.values())


def search(query: str) -> List[CudaConcept]:
    """Naive substring fallback when vector search is unavailable."""
    q = query.lower()
    hits = []
    for c in CONCEPTS.values():
        blob = (c.name + " " + c.one_line_summary + " " + c.detailed_explanation).lower()
        if q in blob or any(w in blob for w in q.split()):
            hits.append(c)
    return hits or list(CONCEPTS.values())[:3]
