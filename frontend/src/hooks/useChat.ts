"use client";
import { useEffect, useRef, useState } from "react";
import { chatWebSocketUrl } from "@/lib/api";

export type AgentEvent =
  | { type: "agent_step"; tool: string; input: any; output: string }
  | { type: "final_answer"; content: string; sources: any[] }
  | { type: "roofline_data"; data: any }
  | { type: "issues_data"; issues: any[] }
  | { type: "error"; message: string };

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  steps?: { tool: string; output: string }[];
  sources?: any[];
}

export function useChat(repoId: string) {
  const wsRef = useRef<WebSocket | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const [steps, setSteps] = useState<{ tool: string; output: string }[]>([]);
  const [roofline, setRoofline] = useState<any | null>(null);
  const [issues, setIssues] = useState<any[]>([]);

  useEffect(() => {
    const ws = new WebSocket(chatWebSocketUrl(repoId));
    wsRef.current = ws;
    ws.onmessage = (e) => {
      const ev: AgentEvent = JSON.parse(e.data);
      if (ev.type === "agent_step") {
        setSteps((s) => [...s, { tool: ev.tool, output: ev.output }]);
      } else if (ev.type === "roofline_data") {
        setRoofline(ev.data);
      } else if (ev.type === "issues_data") {
        setIssues(ev.issues);
      } else if (ev.type === "final_answer") {
        setMessages((m) => [
          ...m,
          { role: "assistant", content: ev.content, steps: [], sources: ev.sources },
        ]);
        setBusy(false);
      } else if (ev.type === "error") {
        setMessages((m) => [...m, { role: "assistant", content: `error: ${ev.message}` }]);
        setBusy(false);
      }
    };
    ws.onclose = () => setBusy(false);
    return () => ws.close();
  }, [repoId]);

  const send = (question: string) => {
    setMessages((m) => [...m, { role: "user", content: question }]);
    setSteps([]);
    setBusy(true);
    wsRef.current?.send(JSON.stringify({ question }));
  };

  return { messages, send, busy, steps, roofline, issues };
}
