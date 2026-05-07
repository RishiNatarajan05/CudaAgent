# CUDA Agent

A CUDA-aware code intelligence agent that analyzes GPU codebases, detects performance anti-patterns, explains kernel code in plain English, and estimates roofline characteristics — all powered by Claude `claude-sonnet-4-20250514` via the Anthropic SDK. Exposed as both an MCP server (for Claude Desktop) and a Next.js web UI.

```
   ┌────────────────────────────────────────────────────────┐
   │                 CUDA Knowledge Base                    │
   │  antipatterns + concepts + (optional) Nvidia docs      │
   └────────────────────────┬───────────────────────────────┘
                            │
   ┌────────────────────────┴───────────────────────────────┐
   │  Ingestion: clone → tree-sitter parse → kernel-aware   │
   │  chunking → static anti-pattern detection → embed →    │
   │  ChromaDB (vector) + BM25 (keyword)                    │
   └────────────────────────┬───────────────────────────────┘
                            │
   ┌────────────────────────┴───────────────────────────────┐
   │  Analysis Engine:                                      │
   │   - pattern_detector (Pass 1 static, Pass 2 LLM)       │
   │   - roofline (arithmetic intensity, A100 reference)    │
   │   - kernel_extractor (build kernel context packages)   │
   └────────────────────────┬───────────────────────────────┘
                            │
   ┌─── LangGraph ReAct Agent (Claude sonnet-4-20250514) ───┐
   │  Tools: search_codebase, read_kernel, analyze_kernel,  │
   │  estimate_roofline, get_repo_cuda_summary,             │
   │  compare_kernels, search_cuda_documentation, …         │
   └─────────┬─────────────────────────────────┬────────────┘
             │                                 │
   ┌─────────┴────────┐               ┌────────┴────────────┐
   │  MCP Server      │               │  FastAPI + Web UI   │
   │  (stdio,         │               │  (Next.js + Recharts│
   │   Claude Desktop)│               │   roofline plot)    │
   └──────────────────┘               └─────────────────────┘
```

## What makes this CUDA-specific

Most "code agents" treat CUDA the same as any C++ — they miss everything that matters on a GPU. This one is built around CUDA-as-a-first-class-citizen:

- **Kernel-aware parsing** — `__global__` and `__device__` functions are extracted as first-class entities with `__shared__` decls, `__syncthreads()` count, register-pressure heuristic, atomics, and `__launch_bounds__` recorded.
- **GPU anti-pattern registry** — warp divergence, uncoalesced global memory, shared-memory bank conflicts, register-pressure occupancy limiters, excessive kernel launches, host-device transfer in loops, missing `__restrict__`, sync overuse — each with detection hints, fix advice, and Nvidia doc references.
- **Two-pass detection** — fast structural detection during ingestion + LLM-assisted deep analysis on demand.
- **Roofline estimator** — static count of FLOPs and global-memory bytes, then compares arithmetic intensity to A100's 156 FLOPs/byte ridge point to call the kernel compute-bound or memory-bound (Nsight Compute is still authoritative — this is for fast directional analysis).
- **CUDA-aware retrieval** — kernel chunks are boosted in hybrid search ranking, so questions like "which kernel does the matrix multiply" land on the kernel and not a launcher wrapper.

## Prerequisites

- Python 3.11+
- Node 18+
- [`uv`](https://docs.astral.sh/uv/) (recommended) or pip
- An Anthropic API key

## Setup

```bash
git clone <this repo>
cd cuda-agent

# Backend
uv venv
uv sync       # or: uv pip install -e .

# Frontend
cd frontend && npm install && cd ..

# Configure API key
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY
```

## Bootstrap the CUDA documentation knowledge base

```bash
uv run python scripts/download_cuda_docs.py
```

This attempts to download Nvidia's CUDA Programming Guide and Best Practices Guide from `docs.nvidia.com` and indexes them into ChromaDB. If the download fails (no internet / firewall), the script falls back to the in-process anti-pattern + concept knowledge base — the agent still works fully offline.

## Run the backend

```bash
uv run uvicorn backend.main:app --reload --port 8000
```

The API will be at `http://localhost:8000` (REST + WebSocket at `/ws/chat/{repo_id}`).

## Run the frontend

```bash
cd frontend && npm run dev
```

Open `http://localhost:3000` and paste a GitHub URL. **First repo to try: `https://github.com/NVIDIA/cuda-samples`.** That's a real Nvidia repo with real kernels.

## (Optional) Use as an MCP server in Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "cuda-agent": {
      "command": "uv",
      "args": ["run", "python", "-m", "backend.mcp_server"],
      "cwd": "/absolute/path/to/cuda-agent",
      "env": { "ANTHROPIC_API_KEY": "sk-ant-..." }
    }
  }
}
```

Restart Claude Desktop. You'll see tools: `ingest_cuda_repo`, `analyze_cuda_kernel`, `ask_cuda_agent`, `estimate_kernel_roofline`, `search_cuda_documentation`, `get_repo_gpu_health`.

## Example questions to try

- "Find all kernels with potential warp divergence."
- "Analyze the `matrixMul` kernel — is it memory or compute bound?"
- "Which kernel in this repo has the worst memory access pattern?"
- "Explain what the reduction kernel does and how its thread indexing works."
- "Are there any kernels missing CUDA error checking?"
- "Compare the `naive_matmul` and `tiled_matmul` kernels — what does the tiled one do better?"

## Anti-patterns detected

| ID | Severity | What it costs you |
|----|----------|-------------------|
| `WARP_DIVERGENCE` | critical | Up to 32× slowdown — branches in a warp serialize |
| `UNCOALESCED_MEMORY` | critical | Up to 32× memory bandwidth loss — strided / AoS access |
| `HOST_DEVICE_TRANSFER_IN_LOOP` | critical | PCIe (~16-32 GB/s) is 30× slower than HBM |
| `SHARED_MEMORY_BANK_CONFLICT` | warning | Up to 32× shared-mem slowdown — pad your tiles |
| `EXCESSIVE_KERNEL_LAUNCHES` | warning | 5-10 µs CPU overhead per launch — batch them |
| `OCCUPANCY_LIMITER` | warning | High register pressure → fewer warps/SM → worse latency hiding |
| `MISSING_RESTRICT` | info | 10-30% perf left on the table from compiler alias-analysis |
| `SYNC_OVERUSE` | info | Each `__syncthreads()` is a barrier — every thread waits |

## Roofline model in 30 seconds

Plot performance (TFLOP/s) vs arithmetic intensity (FLOPs/byte) on log-log axes. Two ceilings: a horizontal "peak compute" line and a diagonal "peak memory bandwidth" line. They meet at the **ridge point**. Kernels left of the ridge are **memory-bound** (you'll never reach peak compute because you can't feed the math units fast enough); right of the ridge are **compute-bound**. A100 reference: peak compute ≈ 312 TFLOP/s (FP16 tensor core), peak HBM ≈ 2.0 TB/s, ridge ≈ 156 FLOPs/byte.

## Known limitations

- The roofline estimator is **static** — it counts operators in source code, not what nvcc actually generates after optimization. Use Nsight Compute for ground truth.
- Register-pressure detection is a heuristic over local-variable count; only `nvcc --ptxas-options=-v` knows for sure.
- Tree-sitter's C++ grammar handles 99% of CUDA but very heavy template metaprogramming can fall through to a regex fallback.

## Architecture decisions

- **uv** for Python because lockfile + virtualenv is one command.
- **ChromaDB persistent client** — local-first, no service to run.
- **sentence-transformers `all-MiniLM-L6-v2`** — small, fast, runs without GPU.
- **rank-bm25 + RRF** for hybrid retrieval — kernel chunks get a 1.25× score boost in fusion.
- **LangGraph + langchain-anthropic** for the ReAct loop — gives us free tool-call streaming over WebSocket.

---

Built to demonstrate deep CUDA + GPU architecture knowledge, paired with modern agent tooling. Test repo: [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples).
