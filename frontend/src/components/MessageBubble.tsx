"use client";
import type { ChatMessage } from "@/hooks/useChat";

export function MessageBubble({ m }: { m: ChatMessage }) {
  const isUser = m.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-3xl whitespace-pre-wrap rounded-lg px-4 py-2 text-sm leading-relaxed ${
          isUser ? "bg-nvidia-500 text-zinc-950" : "bg-zinc-900 border border-zinc-800 text-zinc-100"
        }`}
      >
        {m.content}
      </div>
    </div>
  );
}
