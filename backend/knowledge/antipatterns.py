"""Structured registry of CUDA performance anti-patterns."""
from dataclasses import dataclass, asdict
from typing import List, Literal, Dict

Severity = Literal["critical", "warning", "info"]


@dataclass
class AntiPattern:
    id: str
    name: str
    severity: Severity
    description: str
    impact: str
    detection_hints: List[str]
    fix: str
    nvidia_doc_reference: str
    example_bad: str
    example_good: str

    def to_dict(self) -> dict:
        return asdict(self)


ANTIPATTERNS: Dict[str, AntiPattern] = {
    "WARP_DIVERGENCE": AntiPattern(
        id="WARP_DIVERGENCE",
        name="Warp Divergence",
        severity="critical",
        description=(
            "Conditional branches inside a kernel where threads in the same warp "
            "take different execution paths, causing serialization."
        ),
        impact=(
            "Up to 32x slowdown depending on divergence depth. Inactive threads "
            "still consume warp slots."
        ),
        detection_hints=[
            "if/else statements inside __global__ or __device__ functions",
            "switch statements keyed on threadIdx or blockIdx values",
            "while loops with thread-dependent exit conditions",
            "ternary operators where condition depends on thread ID",
        ],
        fix=(
            "Restructure data so threads in the same warp process similar data. "
            "Use predication for short branches. Consider sorting input data."
        ),
        nvidia_doc_reference="CUDA C++ Best Practices Guide, Section 6.1: Branching and Divergence",
        example_bad=(
            "__global__ void k(int* a) {\n"
            "  int i = threadIdx.x;\n"
            "  if (i % 2 == 0) { a[i] = expensive_even(a[i]); }\n"
            "  else            { a[i] = expensive_odd(a[i]);  }\n"
            "}"
        ),
        example_good=(
            "// Sort inputs so even-indexed work is contiguous, or use predication:\n"
            "__global__ void k(int* a) {\n"
            "  int i = threadIdx.x;\n"
            "  int x = a[i];\n"
            "  int e = expensive_even(x), o = expensive_odd(x);\n"
            "  a[i] = (i % 2 == 0) ? e : o;  // both paths executed but no branch\n"
            "}"
        ),
    ),
    "UNCOALESCED_MEMORY": AntiPattern(
        id="UNCOALESCED_MEMORY",
        name="Uncoalesced Global Memory Access",
        severity="critical",
        description=(
            "Threads in a warp accessing non-sequential global memory addresses, "
            "preventing the memory controller from combining requests."
        ),
        impact=(
            "Up to 32x reduction in effective memory bandwidth. Each thread's "
            "access becomes a separate transaction."
        ),
        detection_hints=[
            "array indexing with stride > 1 (e.g., array[threadIdx.x * N])",
            "struct-of-arrays vs array-of-structs — accessing single fields of an array of structs",
            "2D array access with column-major indexing in a row-major kernel",
            "indirect memory access through an index array",
        ],
        fix=(
            "Ensure consecutive threads access consecutive memory addresses. "
            "Transpose data if needed. Use shared memory as a staging area for "
            "irregular access patterns."
        ),
        nvidia_doc_reference="CUDA C++ Best Practices Guide, Section 9.2.1: Global Memory",
        example_bad=(
            "__global__ void k(float* a) {\n"
            "  int i = threadIdx.x;\n"
            "  a[i * 32] = 1.0f; // stride-32 access — 32 separate transactions\n"
            "}"
        ),
        example_good=(
            "__global__ void k(float* a) {\n"
            "  int i = threadIdx.x;\n"
            "  a[i] = 1.0f; // contiguous — coalesced into 1 transaction\n"
            "}"
        ),
    ),
    "SHARED_MEMORY_BANK_CONFLICT": AntiPattern(
        id="SHARED_MEMORY_BANK_CONFLICT",
        name="Shared Memory Bank Conflict",
        severity="warning",
        description=(
            "Multiple threads in a warp accessing the same shared memory bank "
            "simultaneously, causing serialized access."
        ),
        impact=(
            "Up to 32x slowdown for shared memory operations. 32 banks on modern "
            "GPUs, each 4 bytes wide."
        ),
        detection_hints=[
            "shared memory array indexed by threadIdx.x * constant where constant is a multiple of 32",
            "2D shared memory tiles accessed with non-padded stride",
            "reduction kernels accessing shared memory with power-of-2 strides",
        ],
        fix=(
            "Pad shared memory arrays by 1 element per row. Rearrange access "
            "patterns so threads access different banks. For reductions, use "
            "sequential addressing instead of interleaved."
        ),
        nvidia_doc_reference="CUDA C++ Best Practices Guide, Section 9.2.3: Shared Memory",
        example_bad=(
            "__shared__ float tile[32][32];\n"
            "tile[threadIdx.y][threadIdx.x] = ...;\n"
            "// later: tile[threadIdx.x][threadIdx.y] — column access has bank conflicts"
        ),
        example_good=(
            "__shared__ float tile[32][33]; // pad by 1 to break bank-conflict stride\n"
            "tile[threadIdx.y][threadIdx.x] = ...;"
        ),
    ),
    "EXCESSIVE_KERNEL_LAUNCHES": AntiPattern(
        id="EXCESSIVE_KERNEL_LAUNCHES",
        name="Excessive Kernel Launch Overhead",
        severity="warning",
        description=(
            "Launching many small kernels in a loop instead of batching work into "
            "fewer larger kernels."
        ),
        impact=(
            "Each kernel launch has ~5-10 microsecond CPU overhead plus GPU "
            "pipeline flush. Thousands of launches per second destroys throughput."
        ),
        detection_hints=[
            "kernel launch (<<<>>>) inside a for loop or while loop on the CPU side",
            "kernel that processes only a single element or row",
            "cudaDeviceSynchronize() called inside a loop",
        ],
        fix=(
            "Batch work into a single kernel launch. Use CUDA streams for "
            "overlapping computation. Consider persistent kernels for iterative "
            "algorithms."
        ),
        nvidia_doc_reference="CUDA C++ Best Practices Guide, Section 7: Execution Configuration Optimizations",
        example_bad=(
            "for (int i = 0; i < N; i++)\n"
            "  process_one<<<1, 256>>>(d_data, i);"
        ),
        example_good="process_all<<<grid, block>>>(d_data, N);",
    ),
    "MISSING_RESTRICT": AntiPattern(
        id="MISSING_RESTRICT",
        name="Missing __restrict__ Qualifier",
        severity="info",
        description=(
            "Pointer arguments to kernels not marked __restrict__, preventing "
            "compiler alias analysis and optimization."
        ),
        impact=(
            "Compiler must assume pointers may alias, generating extra loads and "
            "preventing vectorization. Can reduce performance 10-30%."
        ),
        detection_hints=[
            "__global__ function with multiple pointer parameters not marked __restrict__",
            "pointer parameters that are read-only but not marked const __restrict__",
        ],
        fix=(
            "Add __restrict__ to all pointer parameters that do not alias each "
            "other. Combine with const for read-only pointers."
        ),
        nvidia_doc_reference="CUDA C++ Programming Guide, Section B.2.5",
        example_bad="__global__ void k(float* a, float* b, float* c) { ... }",
        example_good="__global__ void k(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) { ... }",
    ),
    "OCCUPANCY_LIMITER": AntiPattern(
        id="OCCUPANCY_LIMITER",
        name="Low Occupancy Due to Register Pressure",
        severity="warning",
        description=(
            "Kernel uses too many registers per thread, limiting the number of "
            "warps that can be active simultaneously on an SM."
        ),
        impact=(
            "Low occupancy reduces ability to hide memory latency. A kernel "
            "using 128 registers/thread can only run 16 warps/SM vs the 64 maximum."
        ),
        detection_hints=[
            "large number of local variables in kernel scope",
            "deeply nested loops with many accumulator variables",
            "large local arrays (stored in registers if small enough)",
            "complex math with many intermediate values",
        ],
        fix=(
            "Use __launch_bounds__ to give compiler register budget hints. "
            "Refactor to reduce live variables. Consider splitting kernel into "
            "multiple passes."
        ),
        nvidia_doc_reference="CUDA C++ Best Practices Guide, Section 8.1: Occupancy",
        example_bad="__global__ void k(...) { float a0,a1,...,a63; /* huge live range */ }",
        example_good="__global__ void __launch_bounds__(256, 4) k(...) { /* compiler budgets registers */ }",
    ),
    "SYNC_OVERUSE": AntiPattern(
        id="SYNC_OVERUSE",
        name="Unnecessary __syncthreads()",
        severity="info",
        description=(
            "__syncthreads() called more often than necessary, stalling all "
            "threads in a block."
        ),
        impact=(
            "Every __syncthreads() is a barrier — all threads wait for the "
            "slowest thread. Unnecessary barriers waste compute cycles."
        ),
        detection_hints=[
            "__syncthreads() called when no shared memory has been written since the last sync",
            "__syncthreads() at the end of a kernel with no subsequent shared memory reads",
            "multiple __syncthreads() calls within the same if branch",
        ],
        fix=(
            "Only synchronize when threads need to read data written by other "
            "threads. Analyze data dependencies carefully."
        ),
        nvidia_doc_reference="CUDA C++ Programming Guide, Section B.6",
        example_bad="smem[tid] = a; __syncthreads(); /* nothing reads smem */ __syncthreads();",
        example_good="smem[tid] = a; __syncthreads(); float v = smem[tid ^ 1];",
    ),
    "HOST_DEVICE_TRANSFER_IN_LOOP": AntiPattern(
        id="HOST_DEVICE_TRANSFER_IN_LOOP",
        name="Host-Device Memory Transfer Inside Loop",
        severity="critical",
        description=(
            "cudaMemcpy or similar transfer called inside a CPU loop, "
            "synchronizing the GPU on every iteration."
        ),
        impact=(
            "PCIe bandwidth is 16-32 GB/s vs GPU memory bandwidth of 900+ GB/s. "
            "Frequent small transfers are catastrophically slow."
        ),
        detection_hints=[
            "cudaMemcpy inside a for/while loop",
            "cudaMemcpyAsync without corresponding stream management inside a loop",
            "thrust::copy or similar inside a loop",
        ],
        fix=(
            "Batch all data onto GPU before computation. If iterative updates "
            "are needed, keep data on GPU and only transfer final results."
        ),
        nvidia_doc_reference="CUDA C++ Best Practices Guide, Section 9.1: Data Transfer",
        example_bad=(
            "for (int i = 0; i < N; i++) {\n"
            "  cudaMemcpy(d, h+i, sizeof(float), cudaMemcpyHostToDevice);\n"
            "  kernel<<<...>>>(d);\n"
            "}"
        ),
        example_good=(
            "cudaMemcpy(d, h, N*sizeof(float), cudaMemcpyHostToDevice);\n"
            "kernel<<<grid, block>>>(d, N);"
        ),
    ),
}


def get(pattern_id: str) -> AntiPattern:
    return ANTIPATTERNS[pattern_id]


def all_patterns() -> List[AntiPattern]:
    return list(ANTIPATTERNS.values())


def format_for_prompt() -> str:
    """Compact formatting for inclusion in LLM prompts."""
    lines = []
    for p in ANTIPATTERNS.values():
        lines.append(f"### {p.id} — {p.name} [{p.severity}]")
        lines.append(f"What: {p.description}")
        lines.append(f"Impact: {p.impact}")
        lines.append("Detection hints:")
        for h in p.detection_hints:
            lines.append(f"  - {h}")
        lines.append(f"Fix: {p.fix}")
        lines.append("")
    return "\n".join(lines)
