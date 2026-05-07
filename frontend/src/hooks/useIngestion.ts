"use client";
import { useEffect, useState } from "react";
import { repoStatus } from "@/lib/api";

export function useIngestion(repoId: string | null) {
  const [status, setStatus] = useState<any>(null);

  useEffect(() => {
    if (!repoId) return;
    let cancel = false;
    const tick = async () => {
      try {
        const s = await repoStatus(repoId);
        if (!cancel) setStatus(s);
        const stage = s?.runtime?.stage || s?.state?.stage;
        if (stage !== "done" && stage !== "error" && !cancel) {
          setTimeout(tick, 1500);
        }
      } catch {
        if (!cancel) setTimeout(tick, 2500);
      }
    };
    tick();
    return () => {
      cancel = true;
    };
  }, [repoId]);

  return status;
}
