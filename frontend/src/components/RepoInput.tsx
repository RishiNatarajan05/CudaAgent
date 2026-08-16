"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ingestRepo, listRepos } from "@/lib/api";

const SUGGESTED = [
  "https://github.com/NVIDIA/cuda-samples",
  "https://github.com/NVIDIA/CUDALibrarySamples",
];

export function RepoInput() {
  const router = useRouter();
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [existing, setExisting] = useState<any[]>([]);

  useEffect(() => {
    listRepos().then(setExisting).catch(() => setExisting([]));
  }, []);

  const submit = async (u: string) => {
    setErr(null);
    setBusy(true);
    try {
      const { repo_id } = await ingestRepo(u);
      router.push(`/chat/${encodeURIComponent(repo_id)}`);
    } catch (e: any) {
      setErr(e.message || String(e));
      setBusy(false);
    }
  };

  return (
    <div className="w-full max-w-2xl space-y-6">
      {existing.length > 0 && (
        <div className="space-y-2">
          <div className="text-sm uppercase tracking-wider text-nvidia-400">
            Previously indexed
          </div>
          <div className="space-y-2">
            {existing.map((r) => {
              const sev = r.total_issues_by_severity || {};
              return (
                <Link
                  key={r.repo_id}
                  href={`/chat/${encodeURIComponent(r.repo_id)}`}
                  className="block bg-zinc-900 border border-zinc-800 hover:border-nvidia-500 rounded px-3 py-2"
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="font-mono text-sm text-zinc-100 truncate">{r.repo_id}</div>
                    <div className="flex gap-1 text-[10px]">
                      <span className="bg-red-950 text-red-300 px-1.5 py-0.5 rounded">
                        {sev.critical || 0}C
                      </span>
                      <span className="bg-amber-950 text-amber-300 px-1.5 py-0.5 rounded">
                        {sev.warning || 0}W
                      </span>
                      <span className="bg-sky-950 text-sky-300 px-1.5 py-0.5 rounded">
                        {sev.info || 0}I
                      </span>
                    </div>
                  </div>
                  <div className="text-xs text-zinc-500 truncate">
                    {r.total_kernels} kernels · {r.url}
                  </div>
                </Link>
              );
            })}
          </div>
        </div>
      )}

      <div className="space-y-2">
        <div className="text-sm uppercase tracking-wider text-zinc-400">
          Or index a new repo
        </div>
        <div className="flex gap-2">
          <input
            className="flex-1 rounded bg-zinc-900 border border-zinc-800 px-3 py-2 outline-none focus:border-nvidia-500"
            placeholder="https://github.com/NVIDIA/cuda-samples"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && url.trim()) submit(url);
            }}
          />
          <button
            className="bg-nvidia-500 hover:bg-nvidia-400 text-zinc-950 font-semibold px-4 py-2 rounded disabled:opacity-50"
            onClick={() => url && submit(url)}
            disabled={busy || !url.trim()}
          >
            {busy ? "Ingesting…" : "Analyze"}
          </button>
        </div>
        {err && <div className="text-red-400 text-sm">{err}</div>}
      </div>

      <div className="space-y-2">
        <div className="text-xs uppercase tracking-wider text-zinc-500">Suggested</div>
        <div className="grid grid-cols-1 gap-2">
          {SUGGESTED.map((s) => (
            <button
              key={s}
              onClick={() => submit(s)}
              className="text-left bg-zinc-900 border border-zinc-800 hover:border-nvidia-500 rounded px-3 py-2 text-sm"
            >
              {s}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
