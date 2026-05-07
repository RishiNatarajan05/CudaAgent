import { RepoInput } from "@/components/RepoInput";

export default function Home() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center p-8 space-y-6">
      <div className="text-center">
        <h1 className="text-4xl font-bold">
          CUDA <span className="text-nvidia-500">Agent</span>
        </h1>
        <p className="text-zinc-400 mt-2">
          GPU-aware code intelligence — kernels, anti-patterns, roofline. Powered by Claude.
        </p>
      </div>
      <RepoInput />
    </main>
  );
}
