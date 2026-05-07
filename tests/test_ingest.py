"""Full ingestion test against a tiny synthetic 'repo' (just our sample.cu)."""
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Force isolated data dirs for the test
import os
TEST_DATA = ROOT / "tests" / "_data"
if TEST_DATA.exists():
    shutil.rmtree(TEST_DATA)
TEST_DATA.mkdir(parents=True)
os.environ["REPOS_DIR"] = str(TEST_DATA / "repos")
os.environ["CHROMA_PERSIST_DIR"] = str(TEST_DATA / "chroma")

# Set up a fake repo dir mirroring what cloner produces
fake_url = "https://github.com/test/sample"
from backend.ingestion.cloner import repo_id_from_url
rid = repo_id_from_url(fake_url)
repo_src = Path(os.environ["REPOS_DIR"]) / rid / "src"
repo_src.mkdir(parents=True, exist_ok=True)
shutil.copy(ROOT / "tests" / "sample.cu", repo_src / "sample.cu")

# Monkey-patch clone_repo to skip git
import backend.ingestion.cloner as cloner_mod
cloner_mod.clone_repo = lambda url, repos_dir: (rid, repo_src)

# Reload indexer to pick up env vars
import importlib
import backend.ingestion.indexer as indexer_mod
importlib.reload(indexer_mod)
import backend.retrieval.vector_search as vs
importlib.reload(vs)
import backend.retrieval.hybrid as hyb
importlib.reload(hyb)

# Re-import after reload
from backend.ingestion.indexer import ingest_repo, load_summary, find_kernel
from backend.retrieval.hybrid import hybrid_search

print("Running ingest_repo...")
summary = ingest_repo(fake_url, progress=lambda s, p: print(f"  [{s}] {p}"))
print("\n--- summary ---")
for k, v in summary.items():
    print(f"  {k}: {v}")

assert summary["total_kernels"] == 3, f"expected 3 kernels, got {summary['total_kernels']}"
assert summary["total_issues_by_severity"]["critical"] >= 2
assert summary["chunk_count"] >= 4

print("\n--- find_kernel('tiled_matmul') ---")
k = find_kernel(rid, "tiled_matmul")
assert k is not None
print(f"  found at {k['filepath']}:{k['start_line']} with patterns {k['detected_patterns']}")

print("\n--- hybrid search 'shared memory tile matmul' ---")
hits = hybrid_search(rid, "shared memory tile matmul", k=3)
for h in hits:
    print(f"  [{h['metadata'].get('chunk_type'):10s}] {h['metadata'].get('name'):20s} score={h['score']:.4f}")
assert any(h["metadata"].get("name") == "tiled_matmul" for h in hits), "hybrid search should surface tiled_matmul"

print("\nALL INGESTION ASSERTIONS PASSED ✓")
