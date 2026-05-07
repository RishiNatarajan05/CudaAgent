"use client";
export function AgentSteps({ steps }: { steps: { tool: string; output: string }[] }) {
  if (!steps.length) return null;
  return (
    <div className="text-xs text-zinc-400 border border-zinc-800 rounded p-2 bg-zinc-950 space-y-1">
      <div className="text-zinc-500 uppercase tracking-wider">agent steps</div>
      {steps.map((s, i) => (
        <div key={i}>
          <span className="text-nvidia-400">→ {s.tool}</span>
          {s.output && (
            <pre className="mt-0.5 max-h-32 overflow-auto text-[11px] text-zinc-400 whitespace-pre-wrap">
              {s.output.slice(0, 400)}
              {s.output.length > 400 ? "…" : ""}
            </pre>
          )}
        </div>
      ))}
    </div>
  );
}
