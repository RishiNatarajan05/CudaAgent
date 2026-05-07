"""WebSocket streaming for the chat agent."""
from __future__ import annotations
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..agent.graph import stream_agent

ws_router = APIRouter()


@ws_router.websocket("/ws/chat/{repo_id}")
async def chat_ws(ws: WebSocket, repo_id: str):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            try:
                payload = json.loads(data)
            except Exception:
                payload = {"question": data}
            question = payload.get("question", "")
            if not question.strip():
                continue
            async for event in stream_agent(repo_id, question):
                await ws.send_text(json.dumps(event))
    except WebSocketDisconnect:
        return
    except Exception as e:  # pragma: no cover
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass
