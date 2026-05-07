"use client";

interface Kernel {
  name: string;
  filepath: string;
  start_line: number;
  end_line: number;
  uses_shared_memory: boolean;
  syncthreads_count: number;
  estimated_register_pressure: string;
  detected_patterns: string[];
}

const PRESSURE_COLOR: Record<string, string> = {
  low: "bg-emerald-900 text-emerald-300",
  medium: "bg-amber-900 text-amber-300",
  high: "bg-red-900 text-red-300",
};

export function KernelCard({ k, onClick }: { k: Kernel; onClick: () => void }) {
  const issueCount = k.detected_patterns.length;
  return (
    <button
      onClick={onClick}
      className="w-full text-left p-2 rounded border border-zinc-800 bg-zinc-900 hover:border-nvidia-500"
    >
      <div className="flex items-center justify-between gap-2">
        <div className="font-mono text-sm text-nvidia-400 truncate">{k.name}</div>
        {issueCount > 0 && (
          <span className="text-[10px] bg-red-900 text-red-200 px-1.5 py-0.5 rounded">
            {issueCount} issue{issueCount > 1 ? "s" : ""}
          </span>
        )}
      </div>
      <div className="text-[11px] text-zinc-500 truncate">
        {k.filepath.split("/").slice(-2).join("/")}:{k.start_line}-{k.end_line}
      </div>
      <div className="flex flex-wrap gap-1 mt-1">
        {k.uses_shared_memory && (
          <span className="text-[10px] bg-zinc-800 text-zinc-300 px-1.5 py-0.5 rounded">smem</span>
        )}
        {k.syncthreads_count > 0 && (
          <span className="text-[10px] bg-zinc-800 text-zinc-300 px-1.5 py-0.5 rounded">
            sync×{k.syncthreads_count}
          </span>
        )}
        <span
          className={`text-[10px] px-1.5 py-0.5 rounded ${
            PRESSURE_COLOR[k.estimated_register_pressure] || "bg-zinc-800 text-zinc-300"
          }`}
        >
          reg:{k.estimated_register_pressure}
        </span>
      </div>
    </button>
  );
}
