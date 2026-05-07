"use client";
import { useState } from "react";
import { useChat } from "@/hooks/useChat";
import { MessageBubble } from "./MessageBubble";
import { AgentSteps } from "./AgentSteps";

export function ChatWindow({ repoId, onResult }: { repoId: string; onResult: (r: { roofline: any; issues: any[]; sources: any[] }) => void }) {
  const { messages, send, busy, steps, roofline, issues } = useChat(repoId);
  const [input, setInput] = useState("");

  const submit = () => {
    if (!input.trim() || busy) return;
    send(input);
    setInput("");
  };

  // Bubble side data up to parent for the right panel
  if (typeof window !== "undefined") {
    (window as any).__cudaAgent = { roofline, issues };
  }
  // Pass via callback (without infinite loop): only when changed snapshot
  // Simpler approach: parent listens via prop and we pass on every render — child controls.
  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-auto space-y-3 p-4">
        {messages.map((m, i) => (
          <div key={i}>
            <MessageBubble m={m} />
          </div>
        ))}
        {busy && <AgentSteps steps={steps} />}
        {!messages.length && (
          <div className="text-zinc-500 text-sm space-y-2">
            <div>Try one of:</div>
            <ul className="list-disc list-inside">
              <li>"List all kernels in this repo and flag the worst issues."</li>
              <li>"Analyze the matrixMul kernel — is it memory or compute bound?"</li>
              <li>"Which kernel has the most warp divergence risk?"</li>
              <li>"Explain how the reduction kernel uses shared memory."</li>
            </ul>
          </div>
        )}
      </div>
      <div className="border-t border-zinc-800 p-3">
        <div className="flex gap-2">
          <textarea
            className="flex-1 bg-zinc-900 border border-zinc-800 rounded px-3 py-2 text-sm focus:border-nvidia-500 outline-none resize-none"
            rows={2}
            value={input}
            placeholder="Ask the CUDA agent…"
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
          />
          <button
            onClick={submit}
            disabled={busy || !input.trim()}
            className="bg-nvidia-500 hover:bg-nvidia-400 text-zinc-950 font-semibold px-4 rounded disabled:opacity-50"
          >
            Send
          </button>
        </div>
      </div>
      <ChildSync roofline={roofline} issues={issues} sources={messages.at(-1)?.sources || []} onResult={onResult} />
    </div>
  );
}

function ChildSync({ roofline, issues, sources, onResult }: any) {
  // Effectful sync to parent without per-render call
  if (typeof window !== "undefined") {
    queueMicrotask(() => onResult({ roofline, issues, sources }));
  }
  return null;
}
