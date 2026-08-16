"use client";
import { useEffect, useRef, useState } from "react";
import { useChat } from "@/hooks/useChat";
import { MessageBubble } from "./MessageBubble";
import { AgentSteps } from "./AgentSteps";

interface SideData {
  roofline: any;
  issues: any[];
  sources: any[];
}

export function ChatWindow({
  repoId,
  onResult,
  seedPrompt,
}: {
  repoId: string;
  onResult: (r: SideData) => void;
  seedPrompt?: string | null;
}) {
  const { messages, send, busy, steps, roofline, issues } = useChat(repoId);
  const [input, setInput] = useState("");
  const lastSyncRef = useRef<string>("");

  const submit = () => {
    if (!input.trim() || busy) return;
    send(input);
    setInput("");
  };

  // If a seed prompt comes in from the sidebar, pre-fill the textarea
  useEffect(() => {
    if (seedPrompt) setInput(seedPrompt);
  }, [seedPrompt]);

  // Sync side-panel state to parent ONLY when it actually changes.
  const sources = messages.at(-1)?.sources || [];
  useEffect(() => {
    const sig = JSON.stringify({
      r: roofline ? roofline.estimated_ai : null,
      i: issues.length,
      s: sources.length,
    });
    if (sig === lastSyncRef.current) return;
    lastSyncRef.current = sig;
    onResult({ roofline, issues, sources });
  }, [roofline, issues, sources, onResult]);

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
              <li>"Analyze the MatrixMulNaive kernel."</li>
              <li>"Which kernel has the most warp divergence risk?"</li>
              <li>"Compare MatrixMulNaive vs MatrixMulAsyncCopyMultiStageSharedState."</li>
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
    </div>
  );
}
