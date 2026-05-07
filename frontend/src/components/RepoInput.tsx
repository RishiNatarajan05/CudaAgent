"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { ingestRepo } from "@/lib/api";

const SUGGESTED = [
  "https://github.com/NVIDIA/cuda-samples",
  "https://github.com/NVIDIA/CUDALibrarySamples",
];

export function RepoInput() {
  const router = useRouter();
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

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
    <div className="w-full max-w-2xl space-y-4">
      <div className="flex gap-2">
        <input
          className="flex-1 rounded bg-zinc-900 border border-zinc-800 px-3 py-2 outline-none focus:border-nvidia-500"
          placeholder="https://github.com/NVIDIA/cuda-samples"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
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
      <div className="text-sm text-zinc-400">Suggested:</div>
      <div className="grid grid-cols-1 gap-2">
        {SUGGESTED.map((s) => (
          <button
            key={s}
            onClick={() => submit(s)}
            className="text-left bg-zinc-900 border border-zinc-800 hover:border-nvidia-500 rounded px-3 py-2"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}
