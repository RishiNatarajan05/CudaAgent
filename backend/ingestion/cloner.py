"""Git repository cloning utilities."""
from __future__ import annotations
import hashlib
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse

from git import Repo


def repo_id_from_url(github_url: str) -> str:
    """Stable id derived from the URL: <owner>__<name>__<short-hash>."""
    parsed = urlparse(github_url.rstrip("/"))
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 2:
        owner, name = parts[-2], parts[-1].removesuffix(".git")
    else:
        owner, name = "repo", parts[-1] if parts else "unknown"
    short = hashlib.sha1(github_url.encode()).hexdigest()[:8]
    return f"{owner}__{name}__{short}"


def clone_repo(github_url: str, repos_dir: str | Path) -> tuple[str, Path]:
    """Clone (or reuse) a repo. Returns (repo_id, local_path)."""
    repos_dir = Path(repos_dir)
    repos_dir.mkdir(parents=True, exist_ok=True)

    rid = repo_id_from_url(github_url)
    target = repos_dir / rid / "src"

    auth_url = github_url
    token = os.environ.get("GITHUB_TOKEN")
    if token and "github.com" in github_url and "@" not in github_url:
        auth_url = github_url.replace("https://", f"https://{token}@")

    if target.exists() and any(target.iterdir()):
        return rid, target

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    Repo.clone_from(auth_url, target, depth=1)
    return rid, target


def list_cuda_files(root: Path) -> list[Path]:
    """Files we consider for CUDA parsing."""
    exts = {".cu", ".cuh", ".cpp", ".cxx", ".cc", ".c", ".h", ".hpp", ".hxx"}
    skip_dirs = {".git", "node_modules", "build", "cmake-build-debug", "third_party", "extern"}
    out = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in skip_dirs for part in p.parts):
            continue
        if p.suffix.lower() in exts:
            out.append(p)
    return out
