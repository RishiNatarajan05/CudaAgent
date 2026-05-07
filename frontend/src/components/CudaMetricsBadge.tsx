"use client";
export function CudaMetricsBadge({ label, value, accent = false }: { label: string; value: string | number; accent?: boolean }) {
  return (
    <div className={`px-2 py-1 rounded text-xs ${accent ? "bg-nvidia-500 text-zinc-950" : "bg-zinc-800 text-zinc-200"}`}>
      <span className="opacity-70 mr-1">{label}:</span>
      <span className="font-semibold">{value}</span>
    </div>
  );
}
