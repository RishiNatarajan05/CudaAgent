"""Quick WS smoke test against a running backend."""
import asyncio
import json
import sys

import websockets


async def main(repo_id: str, question: str):
    url = f"ws://127.0.0.1:8000/ws/chat/{repo_id}"
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"question": question}))
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=120)
            except asyncio.TimeoutError:
                print("(timeout)")
                return
            ev = json.loads(raw)
            t = ev.get("type")
            if t == "agent_step":
                print(f"  -> {ev['tool']}({json.dumps(ev.get('input', {}))[:80]})")
                if ev.get("output"):
                    print(f"     {ev['output'][:160].replace(chr(10), ' / ')}")
            elif t == "roofline_data":
                print(f"  ★ roofline: AI={ev['data']['estimated_ai']} bound={'compute' if ev['data']['is_compute_bound'] else 'memory'}")
            elif t == "issues_data":
                print(f"  ★ issues: {len(ev['issues'])} finding(s)")
            elif t == "final_answer":
                print("\n=== FINAL ANSWER ===")
                print(ev["content"][:1500])
                return
            elif t == "error":
                print("ERROR:", ev["message"])
                return


if __name__ == "__main__":
    repo = sys.argv[1] if len(sys.argv) > 1 else "test__sample__dff52b33"
    q = sys.argv[2] if len(sys.argv) > 2 else "Analyze the tiled_matmul kernel and estimate its roofline."
    asyncio.run(main(repo, q))
