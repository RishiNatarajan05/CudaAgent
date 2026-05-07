"""REST endpoints."""
from __future__ import annotations
import asyncio
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException

from ..ingestion.indexer import (
    ingest_repo, list_repos, load_summary, load_state, load_kernels, load_parsed,
    REPOS_DIR,
)
from ..ingestion.cloner import repo_id_from_url
from ..analysis.pattern_detector import detect_patterns_static
from ..ingestion.cuda_parser import KernelFunction, HostFunction
from .schemas import IngestRequest, IngestResponse


router = APIRouter()

# in-memory ingestion progress tracker
_ingestion_states: dict[str, dict] = {}


def _do_ingest(github_url: str, repo_id: str):
    def progress(stage: str, payload: dict):
        _ingestion_states[repo_id] = {"stage": stage, **payload}
    try:
        summary = ingest_repo(github_url, progress=progress)
        _ingestion_states[repo_id] = {"stage": "done", **summary}
    except Exception as e:
        _ingestion_states[repo_id] = {"stage": "error", "message": str(e)}


@router.post("/api/repos/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest, bg: BackgroundTasks):
    rid = repo_id_from_url(req.github_url)
    _ingestion_states[rid] = {"stage": "queued"}
    bg.add_task(_do_ingest, req.github_url, rid)
    return IngestResponse(repo_id=rid, status="started")


@router.get("/api/repos")
async def repos():
    return list_repos()


@router.get("/api/repos/{repo_id}/status")
async def repo_status(repo_id: str):
    runtime = _ingestion_states.get(repo_id)
    state = load_state(repo_id) or {}
    summary = load_summary(repo_id)
    return {"runtime": runtime, "state": state, "summary": summary}


@router.get("/api/repos/{repo_id}/kernels")
async def repo_kernels(repo_id: str):
    out = []
    for k in load_kernels(repo_id):
        out.append({
            "name": k["name"],
            "filepath": k["filepath"],
            "start_line": k["start_line"],
            "end_line": k["end_line"],
            "uses_shared_memory": k.get("uses_shared_memory", False),
            "syncthreads_count": k.get("syncthreads_count", 0),
            "has_atomic_ops": k.get("has_atomic_ops", False),
            "estimated_register_pressure": k.get("estimated_register_pressure", "low"),
            "detected_patterns": k.get("detected_patterns", []),
        })
    return out


@router.get("/api/repos/{repo_id}/issues")
async def repo_issues(repo_id: str):
    """Returns all detected issues grouped by severity then by file."""
    parsed = load_parsed(repo_id)
    grouped: dict[str, dict[str, list[dict]]] = {"critical": {}, "warning": {}, "info": {}}
    for pf in parsed:
        for k_dict in pf["kernels"]:
            k = KernelFunction(**{**k_dict, "detected_patterns": k_dict.get("detected_patterns", [])})
            for issue in detect_patterns_static(k):
                d = issue.to_dict()
                grouped[issue.severity].setdefault(pf["filepath"], []).append(d)
        for h_dict in pf["host_functions"]:
            kls = [type("KL", (), x) for x in h_dict.get("kernel_launches", [])]
            h = HostFunction(
                name=h_dict["name"], body=h_dict["body"],
                contains_kernel_launches=h_dict.get("contains_kernel_launches", False),
                kernel_launches=[],
                cuda_api_calls=h_dict.get("cuda_api_calls", []),
                has_loop_with_memcpy=h_dict.get("has_loop_with_memcpy", False),
                has_loop_with_kernel_launch=h_dict.get("has_loop_with_kernel_launch", False),
                start_line=h_dict["start_line"], end_line=h_dict["end_line"],
                filepath=h_dict["filepath"],
            )
            for issue in detect_patterns_static(h):
                d = issue.to_dict()
                grouped[issue.severity].setdefault(pf["filepath"], []).append(d)
    return grouped


@router.get("/api/repos/{repo_id}/tree")
async def repo_tree(repo_id: str):
    s = load_summary(repo_id)
    if not s:
        raise HTTPException(404, "repo not indexed")
    parsed = load_parsed(repo_id)
    src_root = Path(s["src_path"])
    by_dir: dict[str, list[dict]] = defaultdict(list)
    for pf in parsed:
        p = Path(pf["filepath"])
        try:
            rel = p.relative_to(src_root)
        except Exception:
            rel = p
        by_dir[str(rel.parent)].append({"name": rel.name, "kernel_count": len(pf["kernels"])})
    return [{"dir": d, "files": sorted(by_dir[d], key=lambda x: x["name"])} for d in sorted(by_dir)]
