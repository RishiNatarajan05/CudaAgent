"use client";
export function SourcePanel({ sources }: { sources: any[] }) {
  if (!sources?.length) return null;
  return (
    <div className="border border-zinc-800 rounded p-2 bg-zinc-900">
      <div className="text-xs uppercase tracking-wider text-zinc-500 mb-1">sources</div>
      <div className="space-y-2">
        {sources.map((s, i) => (
          <pre key={i} className="text-[11px] text-zinc-400 whitespace-pre-wrap max-h-40 overflow-auto">
            {s.preview}
          </pre>
        ))}
      </div>
    </div>
  );
}
