"""MCP server exposing CUDA agent capabilities over stdio.

Run: uv run python -m backend.mcp_server
Claude Desktop config snippet:

{
  "mcpServers": {
    "cuda-agent": {
      "command": "uv",
      "args": ["run", "python", "-m", "backend.mcp_server"],
      "cwd": "/absolute/path/to/cuda-agent",
      "env": { "ANTHROPIC_API_KEY": "..." }
    }
  }
}
"""
from __future__ import annotations
import asyncio
import json

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .ingestion.indexer import ingest_repo, load_summary
from .agent.tools import (
    search_codebase as t_search,
    search_cuda_documentation as t_docs,
    read_kernel as t_read,
    analyze_kernel as t_analyze,
    estimate_roofline as t_roofline,
    get_repo_cuda_summary as t_health,
)
from .agent.graph import stream_agent

server = Server("cuda-agent")


@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="ingest_cuda_repo",
            description=(
                "Clone and index a CUDA/GPU codebase for analysis. Automatically detects kernels, "
                "device functions, and common anti-patterns."
            ),
            inputSchema={
                "type": "object",
                "properties": {"github_url": {"type": "string"}},
                "required": ["github_url"],
            },
        ),
        Tool(
            name="analyze_cuda_kernel",
            description=(
                "Deep performance analysis of a specific CUDA kernel. Detects warp divergence, "
                "uncoalesced memory access, shared memory bank conflicts, occupancy limiters, and "
                "other GPU performance anti-patterns."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_id": {"type": "string"},
                    "kernel_name": {"type": "string"},
                },
                "required": ["repo_id", "kernel_name"],
            },
        ),
        Tool(
            name="ask_cuda_agent",
            description=(
                "Ask a GPU performance engineering question about an indexed CUDA codebase. The agent "
                "has deep knowledge of CUDA best practices and Nvidia GPU architecture."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_id": {"type": "string"},
                    "question": {"type": "string"},
                },
                "required": ["repo_id", "question"],
            },
        ),
        Tool(
            name="estimate_kernel_roofline",
            description=(
                "Estimate arithmetic intensity and roofline characteristics for a CUDA kernel via "
                "static analysis. Determines whether the kernel is compute-bound or memory-bound "
                "relative to A100 specs."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_id": {"type": "string"},
                    "kernel_name": {"type": "string"},
                },
                "required": ["repo_id", "kernel_name"],
            },
        ),
        Tool(
            name="search_cuda_documentation",
            description=(
                "Search Nvidia's CUDA Programming Guide and Best Practices Guide for authoritative "
                "information on GPU programming topics."
            ),
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
        Tool(
            name="get_repo_gpu_health",
            description=(
                "Get a high-level GPU performance health report for an entire CUDA codebase — total "
                "kernels, critical issues, warnings, and files that need the most attention."
            ),
            inputSchema={
                "type": "object",
                "properties": {"repo_id": {"type": "string"}},
                "required": ["repo_id"],
            },
        ),
    ]


def _txt(s: str):
    return [TextContent(type="text", text=s)]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "ingest_cuda_repo":
        loop = asyncio.get_running_loop()
        summary = await loop.run_in_executor(None, ingest_repo, arguments["github_url"])
        return _txt(json.dumps(summary, indent=2))

    if name == "analyze_cuda_kernel":
        out = t_analyze.invoke({"kernel_name": arguments["kernel_name"], "repo_id": arguments["repo_id"]})
        return _txt(out)

    if name == "ask_cuda_agent":
        chunks: list[str] = []
        async for ev in stream_agent(arguments["repo_id"], arguments["question"]):
            if ev["type"] == "final_answer":
                chunks.append(ev["content"])
            elif ev["type"] == "error":
                chunks.append(f"[error] {ev['message']}")
        return _txt("\n".join(chunks) or "(no answer)")

    if name == "estimate_kernel_roofline":
        out = t_roofline.invoke({"kernel_name": arguments["kernel_name"], "repo_id": arguments["repo_id"]})
        return _txt(out)

    if name == "search_cuda_documentation":
        return _txt(t_docs.invoke({"query": arguments["query"]}))

    if name == "get_repo_gpu_health":
        return _txt(t_health.invoke({"repo_id": arguments["repo_id"]}))

    return _txt(f"Unknown tool: {name}")


async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
