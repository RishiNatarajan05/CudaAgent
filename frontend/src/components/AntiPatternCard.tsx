"use client";
import { useState } from "react";

interface Issue {
  pattern_id: string;
  pattern_name: string;
  severity: "critical" | "warning" | "info";
  confidence: string;
  evidence: string;
  line_numbers: number[];
  explanation: string;
  filepath?: string;
  kernel_name?: string;
}

const SEV_COLOR = {
  critical: "border-red-700 bg-red-950/40 text-red-300",
  warning: "border-amber-700 bg-amber-950/40 text-amber-300",
  info: "border-sky-700 bg-sky-950/40 text-sky-300",
} as const;

export function AntiPatternCard({ issue }: { issue: Issue }) {
  const [open, setOpen] = useState(false);
  return (
    <div className={`border rounded p-3 ${SEV_COLOR[issue.severity]}`}>
      <div className="flex items-center justify-between gap-2">
        <div className="font-semibold">{issue.pattern_name}</div>
        <div className="text-[10px] uppercase tracking-wider">
          {issue.severity} · {issue.confidence}
        </div>
      </div>
      {issue.kernel_name && (
        <div className="text-xs text-zinc-400 mt-0.5">
          {issue.kernel_name} {issue.filepath ? `· ${issue.filepath}` : ""}
        </div>
      )}
      <div className="flex flex-wrap gap-1 mt-2">
        {issue.line_numbers.map((l) => (
          <span key={l} className="text-[10px] bg-zinc-800 text-zinc-200 px-1.5 py-0.5 rounded">
            L{l}
          </span>
        ))}
      </div>
      {issue.evidence && (
        <pre className="mt-2 text-xs whitespace-pre-wrap bg-zinc-900 border border-zinc-800 rounded p-2 text-zinc-300">
          {issue.evidence}
        </pre>
      )}
      <div className="text-sm mt-2 text-zinc-200">{issue.explanation}</div>
      <button
        className="text-xs underline mt-2 text-zinc-400"
        onClick={() => setOpen((o) => !o)}
      >
        {open ? "Hide fix" : "How to fix"}
      </button>
      {open && (
        <div className="text-sm mt-1 text-zinc-300">
          See registry pattern <code>{issue.pattern_id}</code> in the agent's knowledge base for the full fix description.
        </div>
      )}
    </div>
  );
}
