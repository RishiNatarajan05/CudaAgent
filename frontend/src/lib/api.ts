export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export async function ingestRepo(github_url: string): Promise<{ repo_id: string; status: string }> {
  const r = await fetch(`${API_BASE}/api/repos/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ github_url }),
  });
  if (!r.ok) throw new Error(`ingest failed: ${r.status}`);
  return r.json();
}

export async function repoStatus(repoId: string) {
  const r = await fetch(`${API_BASE}/api/repos/${repoId}/status`);
  return r.json();
}

export async function listRepos() {
  const r = await fetch(`${API_BASE}/api/repos`);
  return r.json();
}

export async function listKernels(repoId: string) {
  const r = await fetch(`${API_BASE}/api/repos/${repoId}/kernels`);
  return r.json();
}

export async function listIssues(repoId: string) {
  const r = await fetch(`${API_BASE}/api/repos/${repoId}/issues`);
  return r.json();
}

export function chatWebSocketUrl(repoId: string): string {
  const base = API_BASE.replace(/^http/, "ws");
  return `${base}/ws/chat/${repoId}`;
}
