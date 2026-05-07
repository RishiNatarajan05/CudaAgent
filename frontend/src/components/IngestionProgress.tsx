"use client";
import { useIngestion } from "@/hooks/useIngestion";

const STAGE_MSG: Record<string, string> = {
  queued: "Queued…",
  clone: "Cloning repository…",
  cloning: "Cloning repository…",
  parse: "Parsing CUDA kernels…",
  parsing: "Parsing CUDA kernels…",
  parse_progress: "Parsing CUDA kernels…",
  chunk: "Chunking by kernel/device function…",
  chunking: "Chunking by kernel/device function…",
  embed: "Generating embeddings…",
  embedding: "Generating embeddings…",
  index: "Indexing for hybrid search…",
  indexing: "Indexing for hybrid search…",
  done: "Done.",
  error: "Error.",
};

export function IngestionProgress({ repoId }: { repoId: string }) {
  const status = useIngestion(repoId);
  const stage = status?.runtime?.stage || status?.state?.stage;
  const msg = STAGE_MSG[stage || ""] || stage || "Working…";
  const summary = status?.summary;
  return (
    <div className="rounded border border-zinc-800 bg-zinc-900 p-4">
      <div className="text-sm text-zinc-400">{msg}</div>
      {summary && (
        <div className="mt-2 text-zinc-200">
          Found <span className="text-nvidia-400 font-semibold">{summary.total_kernels}</span> kernels,{" "}
          <span className="text-amber-400 font-semibold">
            {(summary.total_issues_by_severity?.critical || 0) +
              (summary.total_issues_by_severity?.warning || 0) +
              (summary.total_issues_by_severity?.info || 0)}
          </span>{" "}
          potential issues.
        </div>
      )}
    </div>
  );
}
