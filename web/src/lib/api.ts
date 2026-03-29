export type MemoryRecord = {
  id: string;
  memory_type: string;
  modality: string;
  content: string;
  created_at: string;
  last_accessed_at: string | null;
  access_count: number;
  importance: number;
  media_ref: string | null;
  session_id?: string;
  turn_number?: number | null;
  participants?: string[];
  summary?: string | null;
  source_mime_type?: string | null;
  category?: string;
  confidence?: number;
  steps?: string[];
  preconditions?: string[];
  success_count?: number;
  failure_count?: number;
  total_outcomes?: number;
  success_rate?: number;
  wilson_score?: number;
};

export type RankedQueryResult = {
  record: MemoryRecord;
  raw_similarity: number;
  recency_score: number;
  importance_score: number;
  final_score: number;
};

export type PlaygroundEvent = {
  event_type: string;
  timestamp: string;
  data: Record<string, unknown>;
};

export type Overview = {
  semantic_count: number;
  episodic_count: number;
  procedural_count: number;
  recent_sessions: string[];
  latest_events: PlaygroundEvent[];
};

export type ProceduralMatchResult = {
  record: MemoryRecord;
  similarity: number;
  wilson_score: number;
  combined_score: number;
};

export type ContradictionCandidate = {
  record: MemoryRecord;
  similarity: number;
};

export type ForgettingDecision = {
  record_id: string;
  memory_type: string;
  action: string;
  reason: string | null;
  score: number;
  media_deleted: boolean;
  executed: boolean;
  record_skip_reason: string | null;
  media_skip_reason: string | null;
  old_importance: number | null;
  new_importance: number | null;
};

export type ForgettingReport = {
  dry_run: boolean;
  scanned: number;
  kept: number;
  faded: number;
  pruned: number;
  media_deleted: number;
  duplicates_flagged: number;
  skipped_records: number;
  skipped_media: number;
  by_type: Record<string, Record<string, number>>;
  decisions: ForgettingDecision[];
};

const API_BASE = process.env.NEXT_PUBLIC_MEMORY_API_BASE_URL ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...init?.headers,
    },
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || "Request failed");
  }

  return (await response.json()) as T;
}

export function getOverview() {
  return request<Overview>("/api/overview");
}

export function getEvents(limit = 40) {
  return request<{ events: PlaygroundEvent[] }>(`/api/events?limit=${limit}`);
}

export function checkHealth() {
  return fetch(`${API_BASE}/health`).then((r) => r.ok);
}

export function createSemanticMemory(input: {
  content: string;
  importance?: number;
  category?: string;
  confidence?: number;
}) {
  return request<{ record: MemoryRecord; potential_contradictions: ContradictionCandidate[] }>(
    "/api/memories/semantic",
    {
      method: "POST",
      body: JSON.stringify(input),
    },
  );
}

export function createTextEpisode(input: {
  session_id: string;
  text: string;
  summary?: string;
  turn_number?: number;
  importance?: number;
}) {
  return request<{ record: MemoryRecord }>("/api/memories/episodic/text", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function createFileEpisode(input: {
  session_id: string;
  file: File;
  content?: string;
  summary?: string;
  importance?: number;
}) {
  const formData = new FormData();
  formData.append("session_id", input.session_id);
  formData.append("file", input.file);
  if (input.content) formData.append("content", input.content);
  if (input.summary) formData.append("summary", input.summary);
  if (input.importance !== undefined) formData.append("importance", String(input.importance));

  return request<{ record: MemoryRecord }>("/api/memories/episodic/file", {
    method: "POST",
    body: formData,
  });
}

export type MediaQueryResult = {
  query_type: "vector";
  source_modality: string;
  results: RankedQueryResult[];
};

export function queryMemories(input: { query: string; top_k?: number; memory_types?: string[] }) {
  return request<{ results: RankedQueryResult[] }>("/api/retrieval/query", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function queryByImage(input: { file: File; top_k?: number; memory_types?: string[] }) {
  const formData = new FormData();
  formData.append("file", input.file);
  if (input.top_k !== undefined) formData.append("top_k", String(input.top_k));
  if (input.memory_types?.length) formData.append("memory_types", input.memory_types.join(","));

  return request<MediaQueryResult>("/api/retrieval/query-by-image", {
    method: "POST",
    body: formData,
  });
}

export function queryByAudio(input: { file: File; top_k?: number; memory_types?: string[] }) {
  const formData = new FormData();
  formData.append("file", input.file);
  if (input.top_k !== undefined) formData.append("top_k", String(input.top_k));
  if (input.memory_types?.length) formData.append("memory_types", input.memory_types.join(","));

  return request<MediaQueryResult>("/api/retrieval/query-by-audio", {
    method: "POST",
    body: formData,
  });
}

export function createProcedure(input: {
  content: string;
  steps: string[];
  preconditions?: string[];
  importance?: number;
}) {
  return request<{ record: MemoryRecord }>("/api/memories/procedural", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function recordOutcome(recordId: string, success: boolean) {
  return request<{ record: MemoryRecord }>(
    `/api/memories/procedural/${encodeURIComponent(recordId)}/outcome`,
    {
      method: "POST",
      body: JSON.stringify({ success }),
    },
  );
}

export function getBestProcedures(input: { task: string; top_k?: number }) {
  return request<{ results: ProceduralMatchResult[] }>("/api/retrieval/best-procedures", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function getRecentEpisodes(n: number) {
  return request<{ records: MemoryRecord[] }>(`/api/episodes/recent?n=${n}`);
}

export function getSessionEpisodes(sessionId: string) {
  return request<{ records: MemoryRecord[] }>(`/api/episodes/session/${encodeURIComponent(sessionId)}`);
}

export function getTimeRangeEpisodes(startIso: string, endIso: string) {
  const params = new URLSearchParams({ start: startIso, end: endIso });
  return request<{ records: MemoryRecord[] }>(`/api/episodes/time-range?${params.toString()}`);
}

export function forgettingPreview() {
  return request<ForgettingReport>("/api/forgetting/preview", { method: "POST" });
}

export function forgettingRun() {
  return request<ForgettingReport>("/api/forgetting/run", { method: "POST" });
}

export function forgettingResolve(input: { keep_id: string; supersede_id: string }) {
  return request<{ superseded_id: string; kept_id: string; status: string }>(
    "/api/forgetting/resolve",
    {
      method: "POST",
      body: JSON.stringify(input),
    },
  );
}
