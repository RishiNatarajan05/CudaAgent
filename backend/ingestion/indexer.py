"""Pipeline orchestration: clone -> parse -> chunk -> detect -> embed -> index."""
from __future__ import annotations
import json
import os
import pickle
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable, Optional

from dotenv import load_dotenv

from .cloner import clone_repo, list_cuda_files, repo_id_from_url
from .cuda_parser import parse_file, ParsedFile, KernelFunction
from .chunker import chunk_parsed_file, CudaChunk
from .embedder import embed
from ..analysis.pattern_detector import detect_patterns_static
from ..retrieval.vector_search import upsert_chunks, get_or_create_collection
from ..retrieval.keyword_search import build_bm25_index

load_dotenv(override=True)

REPOS_DIR = Path(os.environ.get("REPOS_DIR", "./data/repos"))
CHROMA_DIR = Path(os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma"))


ProgressCB = Callable[[str, dict], None]


def _noop(*_a, **_kw): pass


def _save_state(repo_dir: Path, state: dict) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "ingestion_state.json").write_text(json.dumps(state, indent=2))


def _save_summary(repo_dir: Path, summary: dict) -> None:
    (repo_dir / "cuda_summary.json").write_text(json.dumps(summary, indent=2))


def _save_chunks(repo_dir: Path, chunks: list[CudaChunk]) -> None:
    (repo_dir / "chunks.json").write_text(
        json.dumps([c.to_dict() for c in chunks], indent=2)
    )


def _save_parsed(repo_dir: Path, parsed: list[ParsedFile]) -> None:
    (repo_dir / "parsed.json").write_text(
        json.dumps([p.to_dict() for p in parsed], indent=2)
    )


def ingest_repo(
    github_url: str,
    progress: ProgressCB = _noop,
) -> dict:
    """End-to-end ingestion. Returns a summary dict."""
    t0 = time.time()
    progress("clone", {"url": github_url})
    repo_id, src_path = clone_repo(github_url, REPOS_DIR)
    repo_dir = REPOS_DIR / repo_id
    repo_dir.mkdir(parents=True, exist_ok=True)

    state = {"repo_id": repo_id, "url": github_url, "stage": "cloning", "started": t0}
    _save_state(repo_dir, state)

    files = list_cuda_files(src_path)
    progress("parse", {"file_count": len(files)})
    state["stage"] = "parsing"
    state["file_count"] = len(files)
    _save_state(repo_dir, state)

    parsed_files: list[ParsedFile] = []
    for i, f in enumerate(files):
        if i % 25 == 0:
            progress("parse_progress", {"i": i, "total": len(files), "file": str(f)})
        pf = parse_file(f)
        parsed_files.append(pf)

    # Run static pattern detection on each kernel and stamp chunks with pattern ids
    issues_by_severity = Counter()
    files_with_issues: set[str] = set()
    cuda_api_counter = Counter()
    languages = set()
    total_kernels = 0
    total_device_functions = 0
    total_kernel_launches = 0

    for pf in parsed_files:
        for k in pf.kernels:
            issues = detect_patterns_static(k, parsed_file=pf)
            k.detected_patterns = [i.pattern_id for i in issues]
            for i in issues:
                issues_by_severity[i.severity] += 1
            if issues:
                files_with_issues.add(pf.filepath)
        for h in pf.host_functions:
            issues = detect_patterns_static(h, parsed_file=pf)
            for i in issues:
                issues_by_severity[i.severity] += 1
            if issues:
                files_with_issues.add(pf.filepath)
            total_kernel_launches += len(h.kernel_launches)
        for api in pf.cuda_api_calls:
            cuda_api_counter[api] += 1
        ext = Path(pf.filepath).suffix.lower()
        if ext in {".cu", ".cuh"}:
            languages.add("CUDA")
        elif ext in {".cpp", ".cxx", ".cc"}:
            languages.add("C++")
        elif ext in {".c"}:
            languages.add("C")
        elif ext in {".h", ".hpp", ".hxx"}:
            languages.add("Headers")
        total_kernels += len(pf.kernels)
        total_device_functions += len(pf.device_functions)

    _save_parsed(repo_dir, parsed_files)

    progress("chunk", {})
    state["stage"] = "chunking"
    _save_state(repo_dir, state)

    all_chunks: list[CudaChunk] = []
    for pf in parsed_files:
        all_chunks.extend(chunk_parsed_file(pf, src_path))

    _save_chunks(repo_dir, all_chunks)

    progress("embed", {"chunk_count": len(all_chunks)})
    state["stage"] = "embedding"
    _save_state(repo_dir, state)

    if all_chunks:
        embeddings = embed([c.text for c in all_chunks])
    else:
        embeddings = []

    progress("index", {})
    state["stage"] = "indexing"
    _save_state(repo_dir, state)

    if all_chunks:
        coll = get_or_create_collection(f"repo_{repo_id}")
        upsert_chunks(coll, all_chunks, embeddings)
        bm25 = build_bm25_index(all_chunks)
        with open(repo_dir / "bm25.pkl", "wb") as f:
            pickle.dump(bm25, f)

    summary = {
        "repo_id": repo_id,
        "url": github_url,
        "src_path": str(src_path),
        "total_kernels": total_kernels,
        "total_device_functions": total_device_functions,
        "total_kernel_launches": total_kernel_launches,
        "languages_detected": sorted(languages),
        "cuda_api_calls": dict(cuda_api_counter.most_common(50)),
        "files_with_issues": sorted(files_with_issues),
        "total_issues_by_severity": {
            "critical": issues_by_severity["critical"],
            "warning": issues_by_severity["warning"],
            "info": issues_by_severity["info"],
        },
        "chunk_count": len(all_chunks),
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    _save_summary(repo_dir, summary)

    state["stage"] = "done"
    state["finished"] = time.time()
    _save_state(repo_dir, state)

    progress("done", summary)
    return summary


def list_repos() -> list[dict]:
    if not REPOS_DIR.exists():
        return []
    out = []
    for d in REPOS_DIR.iterdir():
        if not d.is_dir():
            continue
        s = d / "cuda_summary.json"
        if s.exists():
            try:
                out.append(json.loads(s.read_text()))
            except Exception:
                pass
    return out


def load_summary(repo_id: str) -> Optional[dict]:
    s = REPOS_DIR / repo_id / "cuda_summary.json"
    if not s.exists():
        return None
    return json.loads(s.read_text())


def load_state(repo_id: str) -> Optional[dict]:
    s = REPOS_DIR / repo_id / "ingestion_state.json"
    if not s.exists():
        return None
    return json.loads(s.read_text())


def load_chunks(repo_id: str) -> list[CudaChunk]:
    p = REPOS_DIR / repo_id / "chunks.json"
    if not p.exists():
        return []
    return [CudaChunk(**c) for c in json.loads(p.read_text())]


def load_parsed(repo_id: str) -> list[dict]:
    p = REPOS_DIR / repo_id / "parsed.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())


def load_kernels(repo_id: str) -> list[dict]:
    parsed = load_parsed(repo_id)
    out = []
    for pf in parsed:
        for k in pf["kernels"]:
            out.append(k)
    return out


def find_kernel(repo_id: str, name: str) -> Optional[dict]:
    for k in load_kernels(repo_id):
        if k["name"] == name:
            return k
    return None
