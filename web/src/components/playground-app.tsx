"use client";

import { FormEvent, useEffect, useEffectEvent, useState } from "react";
import {
  ArrowRight,
  Binary,
  Clock3,
  Database,
  FileUp,
  History,
  Layers3,
  RefreshCcw,
  Search,
  Signal,
  TimerReset,
} from "lucide-react";
import {
  createFileEpisode,
  createSemanticMemory,
  createTextEpisode,
  getEvents,
  getOverview,
  getRecentEpisodes,
  getSessionEpisodes,
  getTimeRangeEpisodes,
  MemoryRecord,
  Overview,
  PlaygroundEvent,
  queryMemories,
  RankedQueryResult,
} from "@/lib/api";

type Notice = { tone: "ok" | "error"; text: string } | null;

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="mb-6 text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-4)]">
      {children}
    </p>
  );
}

function WindowPanel({
  title,
  context,
  children,
}: {
  title: string;
  context?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="glass-border surface-panel overflow-hidden rounded-[18px]">
      <header className="flex items-center justify-between border-b border-[var(--border)] bg-[var(--surface)] px-4 py-3">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="window-dot" />
            <span className="window-dot" />
            <span className="window-dot" />
          </div>
          <span className="mono-numeric text-xs text-[var(--text-3)]">{title}</span>
        </div>
        {context ? <span className="mono-numeric text-[11px] text-[var(--text-4)]">{context}</span> : null}
      </header>
      <div className="p-4 sm:p-5">{children}</div>
    </section>
  );
}

function StatusBadge({ children }: { children: React.ReactNode }) {
  return (
    <span className="mono-numeric inline-flex items-center rounded-[6px] bg-white/[0.06] px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.08em] text-[var(--accent)]">
      {children}
    </span>
  );
}

function StatTile({
  label,
  value,
  icon,
}: {
  label: string;
  value: React.ReactNode;
  icon: React.ReactNode;
}) {
  return (
    <div className="glass-border surface-panel rounded-2xl p-4">
      <div className="mb-4 flex items-center justify-between text-[var(--text-3)]">
        {icon}
        <span className="text-[11px] uppercase tracking-[0.14em]">{label}</span>
      </div>
      <div className="mono-numeric text-3xl text-[var(--text-1)]">{value}</div>
    </div>
  );
}

function FormField({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <label className="block">
      <span className="mb-2 block text-sm text-[var(--text-3)]">{label}</span>
      {children}
    </label>
  );
}

function textInputClass() {
  return "glass-border h-11 w-full rounded-xl bg-white/[0.02] px-4 text-sm text-[var(--text-1)] outline-none transition-colors placeholder:text-[var(--text-4)] focus:border-white/[0.2]";
}

function textareaClass() {
  return "glass-border min-h-[108px] w-full rounded-xl bg-white/[0.02] px-4 py-3 text-sm text-[var(--text-1)] outline-none transition-colors placeholder:text-[var(--text-4)] focus:border-white/[0.2]";
}

function PrimaryButton({
  children,
  disabled,
  type = "button",
}: {
  children: React.ReactNode;
  disabled?: boolean;
  type?: "button" | "submit";
}) {
  return (
    <button
      type={type}
      disabled={disabled}
      className="inline-flex h-11 items-center justify-center gap-2 rounded-xl bg-[var(--text-1)] px-5 text-sm font-medium text-[var(--bg)] transition hover:bg-white disabled:cursor-not-allowed disabled:opacity-50"
    >
      {children}
    </button>
  );
}

function GhostButton({
  children,
  onClick,
  active = false,
}: {
  children: React.ReactNode;
  onClick: () => void;
  active?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`glass-border inline-flex h-11 items-center justify-center gap-2 rounded-xl px-4 text-sm transition ${
        active
          ? "bg-white/[0.08] text-[var(--text-1)]"
          : "bg-transparent text-[var(--text-2)] hover:bg-white/[0.03] hover:text-[var(--text-1)]"
      }`}
    >
      {children}
    </button>
  );
}

function RecordTable({
  title,
  records,
  emptyText,
}: {
  title: string;
  records: MemoryRecord[];
  emptyText: string;
}) {
  return (
    <WindowPanel title={title} context={`${records.length} rows`}>
      {records.length === 0 ? (
        <p className="text-sm text-[var(--text-3)]">{emptyText}</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-left mono-numeric text-[13px] text-[var(--text-2)]">
            <thead>
              <tr className="border-b border-[var(--border)] text-[11px] uppercase tracking-[0.12em] text-[var(--text-4)]">
                <th className="px-0 py-3 font-medium">Memory</th>
                <th className="px-3 py-3 font-medium">Type</th>
                <th className="px-3 py-3 font-medium">Session</th>
                <th className="px-3 py-3 font-medium">Stored</th>
                <th className="px-3 py-3 font-medium">Accessed</th>
              </tr>
            </thead>
            <tbody>
              {records.map((record) => (
                <tr key={record.id} className="border-b border-[var(--border)] transition hover:bg-white/[0.02] last:border-b-0">
                  <td className="py-3 pr-3 align-top">
                    <div className="font-[family-name:var(--font-body)] text-sm leading-6 text-[var(--text-1)]">
                      {record.content}
                    </div>
                    <div className="mt-1 text-[11px] text-[var(--text-4)]">
                      {record.modality}
                      {record.media_ref ? ` · ${shortMediaRef(record.media_ref)}` : ""}
                    </div>
                  </td>
                  <td className="px-3 py-3 align-top">{record.memory_type}</td>
                  <td className="px-3 py-3 align-top break-all">{record.session_id ?? "—"}</td>
                  <td className="px-3 py-3 align-top">{formatDate(record.created_at)}</td>
                  <td className="px-3 py-3 align-top">{record.access_count}x</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </WindowPanel>
  );
}

function QueryTable({ results }: { results: RankedQueryResult[] }) {
  return (
    <WindowPanel title="mixed query results" context={`${results.length} hits`}>
      {results.length === 0 ? (
        <p className="text-sm text-[var(--text-3)]">Run a search to compare semantic and episodic results together.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full border-collapse mono-numeric text-[13px] text-[var(--text-2)]">
            <thead>
              <tr className="border-b border-[var(--border)] text-[11px] uppercase tracking-[0.12em] text-[var(--text-4)]">
                <th className="px-0 py-3 text-left font-medium">Memory</th>
                <th className="px-3 py-3 text-left font-medium">Type</th>
                <th className="px-3 py-3 text-right font-medium">Final</th>
                <th className="px-3 py-3 text-right font-medium">Raw</th>
                <th className="px-3 py-3 text-right font-medium">Recency</th>
                <th className="px-3 py-3 text-right font-medium">Importance</th>
              </tr>
            </thead>
            <tbody>
              {results.map((result) => (
                <tr key={result.record.id} className="border-b border-[var(--border)] transition hover:bg-white/[0.02] last:border-b-0">
                  <td className="py-3 pr-3 align-top">
                    <div className="font-[family-name:var(--font-body)] text-sm leading-6 text-[var(--text-1)]">
                      {result.record.content}
                    </div>
                    <div className="mt-1 text-[11px] text-[var(--text-4)]">
                      {result.record.modality}
                      {result.record.session_id ? ` · ${result.record.session_id}` : ""}
                    </div>
                  </td>
                  <td className="px-3 py-3 align-top">{result.record.memory_type}</td>
                  <td className="px-3 py-3 text-right">{result.final_score.toFixed(4)}</td>
                  <td className="px-3 py-3 text-right">{result.raw_similarity.toFixed(4)}</td>
                  <td className="px-3 py-3 text-right">{result.recency_score.toFixed(4)}</td>
                  <td className="px-3 py-3 text-right">{result.importance_score.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </WindowPanel>
  );
}

function EventTable({ events }: { events: PlaygroundEvent[] }) {
  return (
    <WindowPanel title="event stream" context={`${events.length} latest`}>
      {events.length === 0 ? (
        <p className="text-sm text-[var(--text-3)]">No events yet. The stream will populate as the playground interacts with the API.</p>
      ) : (
        <div className="space-y-3">
          {events.map((event) => (
            <div key={`${event.timestamp}-${event.event_type}`} className="glass-border rounded-2xl bg-[var(--surface)] p-4">
              <div className="mb-2 flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-3">
                  <StatusBadge>{event.event_type}</StatusBadge>
                  <span className="mono-numeric text-[11px] text-[var(--text-4)]">{formatDate(event.timestamp)}</span>
                </div>
              </div>
              <pre className="mono-numeric overflow-x-auto whitespace-pre-wrap text-[12px] leading-6 text-[var(--text-3)]">
                {JSON.stringify(event.data, null, 2)}
              </pre>
            </div>
          ))}
        </div>
      )}
    </WindowPanel>
  );
}

function MicroCopy({ children }: { children: React.ReactNode }) {
  return <p className="text-[12px] leading-5 text-[var(--text-4)]">{children}</p>;
}

function formatDate(value: string) {
  return new Date(value).toLocaleString([], {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function shortMediaRef(value: string) {
  return value.split("/").slice(-2).join("/");
}

function initialTimeInput(offsetMinutes: number) {
  const date = new Date(Date.now() + offsetMinutes * 60_000);
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}T${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
}

export function PlaygroundApp() {
  const apiBaseUrl = process.env.NEXT_PUBLIC_MEMORY_API_BASE_URL ?? "http://localhost:8000";
  const apiDocsUrl = `${apiBaseUrl.replace(/\/$/, "")}/docs`;

  const [overview, setOverview] = useState<Overview | null>(null);
  const [events, setEvents] = useState<PlaygroundEvent[]>([]);
  const [queryResults, setQueryResults] = useState<RankedQueryResult[]>([]);
  const [recentRecords, setRecentRecords] = useState<MemoryRecord[]>([]);
  const [sessionRecords, setSessionRecords] = useState<MemoryRecord[]>([]);
  const [timeRecords, setTimeRecords] = useState<MemoryRecord[]>([]);
  const [notice, setNotice] = useState<Notice>(null);
  const [busyLabel, setBusyLabel] = useState<string | null>(null);

  const [semanticContent, setSemanticContent] = useState("");
  const [episodeMode, setEpisodeMode] = useState<"text" | "file">("text");
  const [episodeSession, setEpisodeSession] = useState("session-playground");
  const [episodeText, setEpisodeText] = useState("");
  const [episodeContent, setEpisodeContent] = useState("");
  const [episodeSummary, setEpisodeSummary] = useState("");
  const [episodeFile, setEpisodeFile] = useState<File | null>(null);
  const [queryText, setQueryText] = useState("retrieval");
  const [recentCount, setRecentCount] = useState("5");
  const [sessionFilter, setSessionFilter] = useState("session-playground");
  const [timeStart, setTimeStart] = useState(initialTimeInput(-120));
  const [timeEnd, setTimeEnd] = useState(initialTimeInput(0));

  const loadInitialData = useEffectEvent(async () => {
    await refreshChrome();
    await runRecent(5, true);
  });

  useEffect(() => {
    void loadInitialData();
  }, []);

  async function refreshChrome() {
    try {
      const [nextOverview, nextEvents] = await Promise.all([getOverview(), getEvents(30)]);
      setOverview(nextOverview);
      setEvents(nextEvents.events);
    } catch (error) {
      setNotice({
        tone: "error",
        text: error instanceof Error ? error.message : "Could not reach the memory API.",
      });
    }
  }

  async function withBusy<T>(label: string, work: () => Promise<T>) {
    setBusyLabel(label);
    setNotice(null);
    try {
      const result = await work();
      await refreshChrome();
      return result;
    } catch (error) {
      setNotice({
        tone: "error",
        text: error instanceof Error ? error.message : "Request failed.",
      });
      throw error;
    } finally {
      setBusyLabel(null);
    }
  }

  async function handleSemanticSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!semanticContent.trim()) return;
    try {
      const result = await withBusy("Storing semantic memory", () =>
        createSemanticMemory({ content: semanticContent.trim() }),
      );
      setSemanticContent("");
      setNotice({ tone: "ok", text: `Stored semantic memory ${result.record.id.slice(0, 8)}.` });
    } catch {}
  }

  async function handleEpisodeSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!episodeSession.trim()) return;

    try {
      if (episodeMode === "text") {
        if (!episodeText.trim()) return;
        const result = await withBusy("Storing text episode", () =>
          createTextEpisode({
            session_id: episodeSession.trim(),
            text: episodeText.trim(),
            summary: episodeSummary.trim() || undefined,
          }),
        );
        setEpisodeText("");
        setEpisodeSummary("");
        setNotice({ tone: "ok", text: `Stored episode ${result.record.id.slice(0, 8)}.` });
      } else {
        if (!episodeFile) return;
        const result = await withBusy("Uploading file-backed episode", () =>
          createFileEpisode({
            session_id: episodeSession.trim(),
            file: episodeFile,
            content: episodeContent.trim() || undefined,
            summary: episodeSummary.trim() || undefined,
          }),
        );
        setEpisodeFile(null);
        setEpisodeContent("");
        setEpisodeSummary("");
        setNotice({ tone: "ok", text: `Stored file-backed episode ${result.record.id.slice(0, 8)}.` });
      }
    } catch {}
  }

  async function handleQuerySubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!queryText.trim()) return;
    try {
      const result = await withBusy("Running mixed query", () =>
        queryMemories({ query: queryText.trim(), top_k: 6 }),
      );
      setQueryResults(result.results);
      setNotice({ tone: "ok", text: `Returned ${result.results.length} mixed memory hits.` });
    } catch {}
  }

  async function runRecent(count: number, silent = false) {
    try {
      const result = await withBusy("Loading recent episodes", () => getRecentEpisodes(count));
      setRecentRecords(result.records);
      if (!silent) {
        setNotice({ tone: "ok", text: `Loaded ${result.records.length} recent episodic memories.` });
      }
    } catch {}
  }

  async function handleRecentSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const parsed = Math.max(1, Number.parseInt(recentCount, 10) || 5);
    await runRecent(parsed);
  }

  async function handleSessionSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!sessionFilter.trim()) return;
    try {
      const result = await withBusy("Loading session timeline", () =>
        getSessionEpisodes(sessionFilter.trim()),
      );
      setSessionRecords(result.records);
      setNotice({ tone: "ok", text: `Loaded ${result.records.length} records for ${sessionFilter.trim()}.` });
    } catch {}
  }

  async function handleTimeRangeSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!timeStart || !timeEnd) return;
    try {
      const result = await withBusy("Loading time window", () =>
        getTimeRangeEpisodes(new Date(timeStart).toISOString(), new Date(timeEnd).toISOString()),
      );
      setTimeRecords(result.records);
      setNotice({ tone: "ok", text: `Loaded ${result.records.length} episodic memories in range.` });
    } catch {}
  }

  return (
    <main className="pb-24">
      <nav className="fixed top-0 right-0 z-50 p-4">
        <a
          href="https://github.com/agentclash/agentic-memory"
          target="_blank"
          rel="noreferrer"
          className="glass-border inline-flex items-center gap-2 rounded-xl bg-white/[0.03] px-3 py-2 text-xs text-[var(--text-3)] transition hover:text-[var(--text-1)]"
        >
          <Layers3 className="size-3.5" />
          View repository
        </a>
      </nav>

      <section className="relative overflow-hidden px-6 pb-18 pt-24 sm:px-8 lg:px-10">
        <div className="relative mx-auto max-w-6xl">
          <SectionLabel>Agentic Memory Playground</SectionLabel>
          <div className="max-w-4xl">
            <h1 className="max-w-3xl font-[family-name:var(--font-display)] text-5xl leading-[1.03] tracking-[-0.03em] text-[var(--text-1)] sm:text-6xl">
              A real workbench for semantic and episodic memory.
            </h1>
            <p className="mt-6 max-w-2xl text-base leading-8 text-[var(--text-2)] sm:text-lg">
              This UI talks to the Python backend directly. Store facts, upload episodic media, query mixed
              retrieval, inspect recent/session/time-range behavior, and watch the event stream the memory system emits.
            </p>
            <div className="mt-10 flex flex-wrap items-center gap-3">
              <StatusBadge>real API mode</StatusBadge>
              <StatusBadge>chroma-backed</StatusBadge>
              <span className="mono-numeric text-xs text-[var(--text-4)]">
                NEXT_PUBLIC_MEMORY_API_BASE_URL = {apiBaseUrl}
              </span>
            </div>
          </div>
        </div>
      </section>

      <section className="px-6 sm:px-8 lg:px-10">
        <div className="mx-auto grid max-w-6xl gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatTile label="Semantic Memories" value={overview?.semantic_count ?? "—"} icon={<Database className="size-4" />} />
          <StatTile label="Episodic Memories" value={overview?.episodic_count ?? "—"} icon={<History className="size-4" />} />
          <StatTile label="Tracked Sessions" value={overview?.recent_sessions.length ?? "—"} icon={<Clock3 className="size-4" />} />
          <StatTile label="Live Event Buffer" value={events.length} icon={<Signal className="size-4" />} />
        </div>
      </section>

      <section className="px-6 pt-12 sm:px-8 lg:px-10">
        <div className="mx-auto max-w-6xl">
          <SectionLabel>Workbench</SectionLabel>
          <div className="grid gap-4 lg:grid-cols-[1.05fr_1fr]">
            <WindowPanel title="store semantic memory" context="fact ingestion">
              <form className="space-y-4" onSubmit={handleSemanticSubmit}>
                <FormField label="Semantic fact">
                  <textarea
                    className={textareaClass()}
                    placeholder="The capital of France is Paris."
                    value={semanticContent}
                    onChange={(event) => setSemanticContent(event.target.value)}
                  />
                </FormField>
                <MicroCopy>Stores a factual memory in the semantic collection and emits `memory.stored`.</MicroCopy>
                <PrimaryButton type="submit" disabled={busyLabel !== null}>
                  <Database className="size-4" />
                  Store fact
                </PrimaryButton>
              </form>
            </WindowPanel>

            <WindowPanel title="store episodic memory" context="text or file-backed">
              <div className="mb-4 flex items-center gap-2">
                <GhostButton active={episodeMode === "text"} onClick={() => setEpisodeMode("text")}>text episode</GhostButton>
                <GhostButton active={episodeMode === "file"} onClick={() => setEpisodeMode("file")}>file-backed episode</GhostButton>
              </div>
              <form className="space-y-4" onSubmit={handleEpisodeSubmit}>
                <FormField label="Session id">
                  <input
                    className={textInputClass()}
                    value={episodeSession}
                    onChange={(event) => setEpisodeSession(event.target.value)}
                  />
                </FormField>

                {episodeMode === "text" ? (
                  <FormField label="Episode text">
                    <textarea
                      className={textareaClass()}
                      placeholder="We debugged retrieval after a failed memory lookup."
                      value={episodeText}
                      onChange={(event) => setEpisodeText(event.target.value)}
                    />
                  </FormField>
                ) : (
                  <div className="grid gap-4">
                    <FormField label="File">
                      <input
                        className={`${textInputClass()} pt-3`}
                        type="file"
                        onChange={(event) => setEpisodeFile(event.target.files?.[0] ?? null)}
                      />
                    </FormField>
                    <FormField label="Content label">
                      <input
                        className={textInputClass()}
                        placeholder="Optional human-readable description"
                        value={episodeContent}
                        onChange={(event) => setEpisodeContent(event.target.value)}
                      />
                    </FormField>
                  </div>
                )}

                <FormField label="Summary">
                  <input
                    className={textInputClass()}
                    placeholder="Optional summary shown in retrieval context"
                    value={episodeSummary}
                    onChange={(event) => setEpisodeSummary(event.target.value)}
                  />
                </FormField>
                <MicroCopy>Use file-backed mode for image, audio, video, and PDF episodes. Modality is inferred from the upload.</MicroCopy>

                <PrimaryButton type="submit" disabled={busyLabel !== null}>
                  <FileUp className="size-4" />
                  Store episode
                </PrimaryButton>
              </form>
            </WindowPanel>
          </div>
        </div>
      </section>

      <section className="px-6 pt-12 sm:px-8 lg:px-10">
        <div className="mx-auto max-w-6xl">
          <SectionLabel>Retrieval</SectionLabel>
          <div className="grid gap-4 lg:grid-cols-[1.3fr_0.7fr]">
            <WindowPanel title="mixed query" context="semantic + episodic">
              <form className="flex flex-col gap-4 sm:flex-row" onSubmit={handleQuerySubmit}>
                <input
                  className={textInputClass()}
                  placeholder="retrieval engine"
                  value={queryText}
                  onChange={(event) => setQueryText(event.target.value)}
                />
                <PrimaryButton type="submit" disabled={busyLabel !== null}>
                  <Search className="size-4" />
                  Search memory
                </PrimaryButton>
              </form>
              <div className="mt-4">
                <MicroCopy>Runs the unified retriever and shows reranked results across semantic and episodic stores.</MicroCopy>
              </div>
            </WindowPanel>

            <WindowPanel title="refresh chrome" context="overview + events">
              <div className="flex h-full items-center">
                <GhostButton onClick={() => void refreshChrome()}>
                  <RefreshCcw className="size-4" />
                  Refresh
                </GhostButton>
              </div>
            </WindowPanel>
          </div>
        </div>
      </section>

      <section className="px-6 pt-12 sm:px-8 lg:px-10">
        <div className="mx-auto max-w-6xl">
          <SectionLabel>Direct Episodic Queries</SectionLabel>
          <div className="grid gap-4 lg:grid-cols-3">
            <WindowPanel title="recent episodes" context="episodic only">
              <form className="space-y-4" onSubmit={handleRecentSubmit}>
                <FormField label="How many">
                  <input
                    className={textInputClass()}
                    value={recentCount}
                    onChange={(event) => setRecentCount(event.target.value)}
                  />
                </FormField>
                <PrimaryButton type="submit" disabled={busyLabel !== null}>
                  <History className="size-4" />
                  Load recent
                </PrimaryButton>
              </form>
            </WindowPanel>

            <WindowPanel title="session timeline" context="ordered by turn">
              <form className="space-y-4" onSubmit={handleSessionSubmit}>
                <FormField label="Session id">
                  <input
                    className={textInputClass()}
                    value={sessionFilter}
                    onChange={(event) => setSessionFilter(event.target.value)}
                  />
                </FormField>
                <PrimaryButton type="submit" disabled={busyLabel !== null}>
                  <Binary className="size-4" />
                  Load session
                </PrimaryButton>
              </form>
            </WindowPanel>

            <WindowPanel title="time range" context="inclusive window">
              <form className="space-y-4" onSubmit={handleTimeRangeSubmit}>
                <FormField label="Start">
                  <input
                    className={textInputClass()}
                    type="datetime-local"
                    value={timeStart}
                    onChange={(event) => setTimeStart(event.target.value)}
                  />
                </FormField>
                <FormField label="End">
                  <input
                    className={textInputClass()}
                    type="datetime-local"
                    value={timeEnd}
                    onChange={(event) => setTimeEnd(event.target.value)}
                  />
                </FormField>
                <PrimaryButton type="submit" disabled={busyLabel !== null}>
                  <TimerReset className="size-4" />
                  Load range
                </PrimaryButton>
              </form>
            </WindowPanel>
          </div>
        </div>
      </section>

      <section className="px-6 pt-12 sm:px-8 lg:px-10">
        <div className="mx-auto max-w-6xl space-y-4">
          {notice ? (
            <div
              className={`glass-border rounded-2xl px-4 py-3 text-sm ${
                notice.tone === "error"
                  ? "bg-white/[0.06] text-[var(--text-1)]"
                  : "bg-white/[0.03] text-[var(--text-2)]"
              }`}
            >
              {busyLabel ? `${busyLabel}... ` : ""}
              {notice.text}
            </div>
          ) : busyLabel ? (
            <div className="glass-border rounded-2xl bg-white/[0.03] px-4 py-3 text-sm text-[var(--text-2)]">
              {busyLabel}...
            </div>
          ) : null}

          <QueryTable results={queryResults} />
          <RecordTable title="recent episodic results" records={recentRecords} emptyText="No recent episodic results loaded yet." />
          <RecordTable title="session results" records={sessionRecords} emptyText="Load a session to inspect ordered episodic playback." />
          <RecordTable title="time-range results" records={timeRecords} emptyText="Load a time window to inspect episodic memories inside it." />
          <EventTable events={events} />
        </div>
      </section>

      <footer className="px-6 pt-12 sm:px-8 lg:px-10">
        <div className="mx-auto flex max-w-6xl flex-col gap-4 border-t border-[var(--border)] py-8 text-[12px] text-[var(--text-3)] sm:flex-row sm:items-center sm:justify-between">
          <span className="mono-numeric">memory.agentclash.dev · powered by the Python memory backend</span>
          <div className="flex flex-wrap items-center gap-3">
            <a href={apiDocsUrl} target="_blank" rel="noreferrer" className="hover:text-[var(--text-1)]">
              API docs
            </a>
            <a href="https://github.com/agentclash/agentic-memory" target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 hover:text-[var(--text-1)]">
              Source
              <ArrowRight className="size-3.5" />
            </a>
          </div>
        </div>
      </footer>
    </main>
  );
}
