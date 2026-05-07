"""FastAPI entrypoint. Run with: uv run uvicorn backend.main:app --reload --port 8000"""
from __future__ import annotations
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router as rest_router
from .api.websocket import ws_router

load_dotenv(override=True)

app = FastAPI(title="CUDA Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(rest_router)
app.include_router(ws_router)


@app.get("/health")
async def health():
    return {"ok": True, "model": "claude-sonnet-4-5"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
