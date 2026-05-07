"""Pydantic schemas for FastAPI endpoints."""
from __future__ import annotations
from typing import Any, List, Optional

from pydantic import BaseModel


class IngestRequest(BaseModel):
    github_url: str


class IngestResponse(BaseModel):
    repo_id: str
    status: str


class RepoSummary(BaseModel):
    repo_id: str
    url: str
    total_kernels: int
    total_device_functions: int
    total_kernel_launches: int
    languages_detected: List[str]
    cuda_api_calls: dict
    files_with_issues: List[str]
    total_issues_by_severity: dict
    chunk_count: int = 0
    elapsed_seconds: float = 0.0


class KernelDigest(BaseModel):
    name: str
    filepath: str
    start_line: int
    end_line: int
    uses_shared_memory: bool
    syncthreads_count: int
    has_atomic_ops: bool
    estimated_register_pressure: str
    detected_patterns: List[str]


class IssuesByFile(BaseModel):
    filepath: str
    issues: List[dict]


class WSMessage(BaseModel):
    type: str
    data: Any = None
