"use client";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { listKernels, repoStatus, listIssues } from "@/lib/api";
import { IngestionProgress } from "@/components/IngestionProgress";
import { KernelCard } from "@/components/KernelCard";
import { ChatWindow } from "@/components/ChatWindow";
import { RooflinePlot } from "@/components/RooflinePlot";
import { AntiPatternCard } from "@/components/AntiPatternCard";
import { CudaMetricsBadge } from "@/components/CudaMetricsBadge";
import { SourcePanel } from "@/components/SourcePanel";

export default function ChatPage() {
  const params = useParams();
  const repoId = decodeURIComponent(String(params.repoId));
  const [kernels, setKernels] = useState<any[]>([]);
  const [summary, setSummary] = useState<any>(null);
  const [ready, setReady] = useState(false);
  const [seed, setSeed] = useState<string | null>(null);
  const [side, setSide] = useState<{ roofline: any; issues: any[]; sources: any[] }>({
    roofline: null,
    issues: [],
    sources: [],
  });
  const [issuesByFile, setIssuesByFile] = useState<any>(null);

  useEffect(() => {
    let cancel = false;
    const tick = async () => {
      const s = await repoStatus(repoId);
      if (cancel) return;
      const stage = s?.runtime?.stage || s?.state?.stage;
      setSummary(s?.summary);
      if (stage === "done" || s?.summary) {
        setReady(true);
        const ks = await listKernels(repoId);
        setKernels(ks);
        try {
          setIssuesByFile(await listIssues(repoId));
        } catch {}
      } else {
        setTimeout(tick, 1500);
      }
    };
    tick();
    return () => {
      cancel = true;
    };
  }, [repoId]);

  if (!ready) {
    return (
      <main className="min-h-screen p-8 max-w-3xl mx-auto space-y-4">
        <h1 className="text-xl font-semibold">Indexing {repoId}</h1>
        <IngestionProgress repoId={repoId} />
      </main>
    );
  }

  const sev = summary?.total_issues_by_severity || { critical: 0, warning: 0, info: 0 };

  return (
    <main className="grid grid-cols-12 h-screen">
      {/* Left sidebar */}
      <aside className="col-span-3 border-r border-zinc-800 bg-zinc-950 overflow-auto p-3 space-y-3">
        <div>
          <div className="text-xs uppercase tracking-wider text-zinc-500">repo</div>
          <div className="text-sm text-zinc-200 truncate">{repoId}</div>
        </div>
        <div className="flex flex-wrap gap-1.5">
          <CudaMetricsBadge label="kernels" value={summary?.total_kernels ?? 0} accent />
          <CudaMetricsBadge label="device fns" value={summary?.total_device_functions ?? 0} />
          <CudaMetricsBadge label="launches" value={summary?.total_kernel_launches ?? 0} />
        </div>
        <div className="grid grid-cols-3 gap-1 text-center text-xs">
          <div className="rounded bg-red-950 text-red-300 py-1">{sev.critical} crit</div>
          <div className="rounded bg-amber-950 text-amber-300 py-1">{sev.warning} warn</div>
          <div className="rounded bg-sky-950 text-sky-300 py-1">{sev.info} info</div>
        </div>
        <div>
          <div className="text-xs uppercase tracking-wider text-zinc-500 mb-1">kernels</div>
          <div className="space-y-1.5">
            {kernels.map((k) => (
              <KernelCard
                key={k.name + k.start_line}
                k={k}
                onClick={() => setSeed(`Analyze the ${k.name} kernel — performance findings, roofline, and recommended fixes.`)}
              />
            ))}
            {!kernels.length && <div className="text-xs text-zinc-500">No kernels found.</div>}
          </div>
        </div>
      </aside>

      {/* Chat */}
      <section className="col-span-6 border-r border-zinc-800 flex flex-col">
        <ChatWindow repoId={repoId} onResult={setSide} />
        {seed && (
          <div className="px-3 pb-2 text-[11px] text-zinc-500">
            Click-suggested prompt: <code className="text-zinc-300">{seed}</code> — type Send.
            <button
              className="ml-2 underline"
              onClick={() => {
                navigator.clipboard?.writeText(seed);
              }}
            >
              copy
            </button>
          </div>
        )}
      </section>

      {/* Right panel */}
      <aside className="col-span-3 overflow-auto p-3 space-y-3">
        {side.roofline && <RooflinePlot data={side.roofline} />}
        {side.issues?.length > 0 && (
          <div className="space-y-2">
            <div className="text-xs uppercase tracking-wider text-zinc-500">findings</div>
            {side.issues.map((i: any, idx: number) => (
              <AntiPatternCard key={idx} issue={i} />
            ))}
          </div>
        )}
        {side.sources?.length > 0 && <SourcePanel sources={side.sources} />}
        {!side.roofline && !side.issues?.length && issuesByFile && (
          <div className="space-y-2">
            <div className="text-xs uppercase tracking-wider text-zinc-500">repo issues</div>
            {(["critical", "warning", "info"] as const).map((sev) => {
              const files = issuesByFile?.[sev] || {};
              const flat: any[] = [];
              for (const f of Object.keys(files)) {
                for (const i of files[f]) flat.push(i);
              }
              if (!flat.length) return null;
              return (
                <div key={sev} className="space-y-1.5">
                  {flat.slice(0, 5).map((i, idx) => (
                    <AntiPatternCard key={sev + idx} issue={i} />
                  ))}
                </div>
              );
            })}
          </div>
        )}
      </aside>
    </main>
  );
}
