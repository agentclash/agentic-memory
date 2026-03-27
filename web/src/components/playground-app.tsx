"use client";

import {
  type DragEvent,
  type FormEvent,
  useEffect,
  useEffectEvent,
  useRef,
  useState,
} from "react";
import {
  ChevronDown,
  Clock3,
  Database,
  FileUp,
  History,
  Image,
  Layers3,
  Music,
  RefreshCcw,
  Search,
  Signal,
  Type,
  Zap,
} from "lucide-react";
import {
  checkHealth,
  createFileEpisode,
  createSemanticMemory,
  createTextEpisode,
  getEvents,
  getOverview,
  getRecentEpisodes,
  getSessionEpisodes,
  getTimeRangeEpisodes,
  type MemoryRecord,
  type Overview,
  type PlaygroundEvent,
  queryByAudio,
  queryByImage,
  queryMemories,
  type RankedQueryResult,
} from "@/lib/api";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

type Notice = { tone: "ok" | "error"; text: string } | null;
type SidebarTab = "store" | "explore";
type StoreMode = "semantic" | "text" | "file";
type ExploreMode = "recent" | "session" | "time";
type QueryMode = "text" | "image" | "audio";

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

const EXAMPLE_FACTS = [
  { content: "The capital of France is Paris", category: "fact" },
  { content: "Transformers use self-attention to process sequences in parallel", category: "definition" },
  { content: "Python was created by Guido van Rossum and first released in 1991", category: "fact" },
];

const EXAMPLE_EPISODES = [
  { content: "We debugged a retrieval bug where recency scores were inverted", session: "session-debug-01" },
  { content: "User asked about multimodal embeddings and we walked through the Gemini pipeline", session: "session-onboard-01" },
];

const SUGGESTED_QUERIES = ["capital city", "attention mechanism", "debug session", "embedding pipeline"];

const CATEGORIES = ["general", "fact", "definition", "relationship", "preference", "procedure"];

/* ------------------------------------------------------------------ */
/*  Utilities                                                          */
/* ------------------------------------------------------------------ */

function formatDate(value: string) {
  return new Date(value).toLocaleString([], {
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
  const d = new Date(Date.now() + offsetMinutes * 60_000);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/* ------------------------------------------------------------------ */
/*  Micro-components                                                   */
/* ------------------------------------------------------------------ */

function HealthDot({ status }: { status: "checking" | "ok" | "error" }) {
  const color =
    status === "ok" ? "bg-emerald-400" : status === "error" ? "bg-red-400" : "bg-amber-400";
  return (
    <span className="relative flex size-2" title={`API ${status}`}>
      {status === "ok" && (
        <span
          className={`absolute inline-flex size-full animate-ping rounded-full ${color} opacity-40`}
        />
      )}
      <span className={`relative inline-flex size-2 rounded-full ${color}`} />
    </span>
  );
}

function TypeBadge({ type }: { type: string }) {
  const cls = type === "semantic" ? "badge-semantic" : "badge-episodic";
  return (
    <span
      className={`${cls} inline-flex shrink-0 items-center rounded-md px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider`}
    >
      {type}
    </span>
  );
}

function ScoreBar({
  label,
  value,
  accent = false,
}: {
  label: string;
  value: number;
  accent?: boolean;
}) {
  const pct = Math.min(100, Math.max(0, value * 100));
  return (
    <div className="flex items-center gap-3">
      <span className="w-[72px] text-[10px] uppercase tracking-wider text-[var(--text-4)]">
        {label}
      </span>
      <div className="h-[5px] flex-1 overflow-hidden rounded-full bg-white/[0.06]">
        <div
          className={`h-full rounded-full transition-all duration-500 ${accent ? "score-bar-accent" : "score-bar-default"}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="mono-numeric w-11 text-right text-[11px] text-[var(--text-3)]">
        {value.toFixed(3)}
      </span>
    </div>
  );
}

function FormLabel({ children }: { children: React.ReactNode }) {
  return <span className="mb-1.5 block text-[12px] text-[var(--text-3)]">{children}</span>;
}

function FieldInput(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={`field-input ${props.className ?? ""}`} />;
}

function FieldTextarea(props: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea {...props} className={`field-textarea ${props.className ?? ""}`} />;
}

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-lg px-3 py-1.5 text-[11px] font-medium uppercase tracking-wider transition ${
        active
          ? "bg-white/[0.08] text-[var(--text-1)]"
          : "text-[var(--text-4)] hover:bg-white/[0.03] hover:text-[var(--text-2)]"
      }`}
    >
      {children}
    </button>
  );
}

function SubmitButton({
  children,
  disabled,
  small = false,
}: {
  children: React.ReactNode;
  disabled?: boolean;
  small?: boolean;
}) {
  return (
    <button
      type="submit"
      disabled={disabled}
      className={`inline-flex items-center justify-center gap-2 rounded-xl bg-[var(--text-1)] font-medium text-[var(--bg)] transition hover:bg-white disabled:cursor-not-allowed disabled:opacity-50 ${
        small ? "h-9 px-3.5 text-[12px]" : "h-10 px-4 text-sm"
      }`}
    >
      {children}
    </button>
  );
}

function ImportanceSlider({
  value,
  onChange,
}: {
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <FormLabel>Importance</FormLabel>
      <div className="flex items-center gap-3">
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="importance-slider flex-1"
        />
        <span className="mono-numeric w-9 text-right text-[11px] text-[var(--text-3)]">
          {value.toFixed(2)}
        </span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  File drop zone                                                     */
/* ------------------------------------------------------------------ */

function FileDropZone({
  file,
  onFile,
}: {
  file: File | null;
  onFile: (f: File | null) => void;
}) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function onDragOver(e: DragEvent) {
    e.preventDefault();
  }
  function onDragEnter(e: DragEvent) {
    e.preventDefault();
    setDragging(true);
  }
  function onDragLeave(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
  }
  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) onFile(f);
  }

  return (
    <div
      onDragOver={onDragOver}
      onDragEnter={onDragEnter}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
      className={`drop-zone ${dragging ? "drop-zone-active" : ""} ${file ? "drop-zone-filled" : ""}`}
    >
      <input
        ref={inputRef}
        type="file"
        className="hidden"
        onChange={(e) => onFile(e.target.files?.[0] ?? null)}
      />
      {file ? (
        <div className="space-y-1 text-center">
          <p className="max-w-full truncate text-[13px] text-[var(--text-1)]">{file.name}</p>
          <p className="text-[11px] text-[var(--text-4)]">
            {(file.size / 1024 / 1024).toFixed(2)} MB
            {file.size > 20 * 1024 * 1024 && (
              <span className="ml-1 text-red-400">exceeds 20 MB limit</span>
            )}
          </p>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onFile(null);
            }}
            className="text-[11px] text-[var(--text-3)] underline underline-offset-2 hover:text-[var(--text-1)]"
          >
            Remove
          </button>
        </div>
      ) : (
        <div className="space-y-1.5 text-center">
          <FileUp className="mx-auto size-5 text-[var(--text-4)]" />
          <p className="text-[12px] text-[var(--text-3)]">Drop file or click to browse</p>
          <p className="text-[10px] text-[var(--text-4)]">Image, audio, video, or PDF · max 20 MB</p>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Result cards                                                       */
/* ------------------------------------------------------------------ */

function ResultCard({ result }: { result: RankedQueryResult }) {
  const { record } = result;
  return (
    <div className="glass-border space-y-3 rounded-2xl bg-[var(--surface)] p-5">
      <div className="flex items-start justify-between gap-3">
        <p className="flex-1 text-sm leading-relaxed text-[var(--text-1)]">{record.content}</p>
        <TypeBadge type={record.memory_type} />
      </div>
      {record.summary && (
        <p className="text-[12px] italic leading-5 text-[var(--text-3)]">{record.summary}</p>
      )}
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[10px] text-[var(--text-4)]">
        <span>{record.modality}</span>
        {record.session_id && (
          <>
            <span>·</span>
            <span>{record.session_id}</span>
          </>
        )}
        {record.media_ref && (
          <>
            <span>·</span>
            <span>{shortMediaRef(record.media_ref)}</span>
          </>
        )}
        <span>·</span>
        <span>{formatDate(record.created_at)}</span>
        <span>·</span>
        <span>{record.access_count}× accessed</span>
      </div>
      <div className="space-y-1.5 pt-1">
        <ScoreBar label="relevance" value={result.raw_similarity} />
        <ScoreBar label="recency" value={result.recency_score} />
        <ScoreBar label="importance" value={result.importance_score} />
        <div className="mt-1.5 border-t border-[var(--border)] pt-1.5">
          <ScoreBar label="final" value={result.final_score} accent />
        </div>
      </div>
    </div>
  );
}

function MemoryCard({ record }: { record: MemoryRecord }) {
  return (
    <div className="glass-border rounded-xl bg-[var(--surface)] p-4">
      <div className="flex items-start justify-between gap-3">
        <p className="flex-1 text-sm leading-relaxed text-[var(--text-1)]">{record.content}</p>
        <TypeBadge type={record.memory_type} />
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-[10px] text-[var(--text-4)]">
        <span>{record.modality}</span>
        {record.session_id && (
          <>
            <span>·</span>
            <span>{record.session_id}</span>
          </>
        )}
        {record.turn_number != null && (
          <>
            <span>·</span>
            <span>turn {record.turn_number}</span>
          </>
        )}
        <span>·</span>
        <span>{formatDate(record.created_at)}</span>
        <span>·</span>
        <span>{record.access_count}× accessed</span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Event drawer                                                       */
/* ------------------------------------------------------------------ */

function EventDrawer({
  events,
  open,
  onToggle,
}: {
  events: PlaygroundEvent[];
  open: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="shrink-0 border-t border-[var(--border)] bg-[var(--bg-alt)]">
      <button
        type="button"
        onClick={onToggle}
        className="flex h-10 w-full items-center justify-between px-5 text-[11px] uppercase tracking-wider text-[var(--text-4)] transition hover:text-[var(--text-3)]"
      >
        <span className="flex items-center gap-2">
          <Signal className="size-3" />
          Event stream
          <span className="mono-numeric rounded bg-white/[0.06] px-1.5 py-0.5 text-[10px]">
            {events.length}
          </span>
        </span>
        <ChevronDown
          className={`size-3.5 transition-transform ${open ? "" : "rotate-180"}`}
        />
      </button>
      {open && (
        <div className="max-h-[280px] overflow-y-auto px-5 pb-4">
          {events.length === 0 ? (
            <p className="text-[12px] text-[var(--text-4)]">
              Events will appear here as you interact with the API.
            </p>
          ) : (
            <div className="space-y-1.5">
              {events.map((ev, i) => (
                <div key={`${ev.timestamp}-${i}`} className="flex gap-3 text-[11px]">
                  <span className="mono-numeric shrink-0 text-[var(--text-4)]">
                    {formatDate(ev.timestamp)}
                  </span>
                  <span
                    className={`shrink-0 font-medium ${
                      ev.event_type === "memory.stored"
                        ? "text-emerald-400/70"
                        : ev.event_type === "memory.retrieved"
                          ? "text-amber-400/70"
                          : ev.event_type === "memory.ranked"
                            ? "text-sky-400/70"
                            : "text-[var(--text-3)]"
                    }`}
                  >
                    {ev.event_type}
                  </span>
                  <span className="mono-numeric truncate text-[var(--text-4)]">
                    {JSON.stringify(ev.data)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Main component                                                     */
/* ------------------------------------------------------------------ */

export function PlaygroundApp() {
  const apiBaseUrl =
    process.env.NEXT_PUBLIC_MEMORY_API_BASE_URL ?? "http://localhost:8000";

  /* ---- health ---- */
  const [apiHealth, setApiHealth] = useState<"checking" | "ok" | "error">("checking");

  /* ---- overview ---- */
  const [overview, setOverview] = useState<Overview | null>(null);
  const [events, setEvents] = useState<PlaygroundEvent[]>([]);
  const [eventsOpen, setEventsOpen] = useState(false);

  /* ---- UI ---- */
  const [notice, setNotice] = useState<Notice>(null);
  const [busyLabel, setBusyLabel] = useState<string | null>(null);
  const [sidebarTab, setSidebarTab] = useState<SidebarTab>("store");
  const [storeMode, setStoreMode] = useState<StoreMode>("semantic");
  const [exploreMode, setExploreMode] = useState<ExploreMode>("recent");

  /* ---- semantic form ---- */
  const [semContent, setSemContent] = useState("");
  const [semImportance, setSemImportance] = useState(0.5);
  const [semCategory, setSemCategory] = useState("general");
  const [semConfidence, setSemConfidence] = useState(1.0);

  /* ---- episodic (shared) ---- */
  const [epiSession, setEpiSession] = useState("session-playground");
  const [epiImportance, setEpiImportance] = useState(0.5);
  const [epiSummary, setEpiSummary] = useState("");

  /* ---- episodic text ---- */
  const [epiText, setEpiText] = useState("");
  const [epiTurn, setEpiTurn] = useState("");

  /* ---- episodic file ---- */
  const [epiFile, setEpiFile] = useState<File | null>(null);
  const [epiFileContent, setEpiFileContent] = useState("");

  /* ---- query ---- */
  const [queryMode, setQueryMode] = useState<QueryMode>("text");
  const [queryText, setQueryText] = useState("");
  const [queryFile, setQueryFile] = useState<File | null>(null);
  const [topK, setTopK] = useState(6);
  const [queryTypeFilter, setQueryTypeFilter] = useState<"all" | "semantic" | "episodic">("all");
  const [queryResults, setQueryResults] = useState<RankedQueryResult[]>([]);
  const [querySubmitted, setQuerySubmitted] = useState(false);

  /* ---- explore ---- */
  const [recentCount, setRecentCount] = useState("5");
  const [sessionFilter, setSessionFilter] = useState("session-playground");
  const [timeStart, setTimeStart] = useState(initialTimeInput(-120));
  const [timeEnd, setTimeEnd] = useState(initialTimeInput(0));
  const [exploreRecords, setExploreRecords] = useState<MemoryRecord[]>([]);
  const [exploreLabel, setExploreLabel] = useState("");

  /* ================================================================ */
  /*  Effects                                                          */
  /* ================================================================ */

  useEffect(() => {
    checkHealth()
      .then((ok) => setApiHealth(ok ? "ok" : "error"))
      .catch(() => setApiHealth("error"));
  }, [apiBaseUrl]);

  useEffect(() => {
    if (!notice) return;
    const t = setTimeout(() => setNotice(null), 4000);
    return () => clearTimeout(t);
  }, [notice]);

  const loadInitial = useEffectEvent(async () => {
    await refreshData();
  });

  useEffect(() => {
    void loadInitial();
  }, []);

  /* ================================================================ */
  /*  Helpers                                                          */
  /* ================================================================ */

  async function refreshData() {
    try {
      const [ov, ev] = await Promise.all([getOverview(), getEvents(40)]);
      setOverview(ov);
      setEvents(ev.events);
    } catch {
      /* silent */
    }
  }

  async function withBusy<T>(label: string, work: () => Promise<T>): Promise<T | null> {
    setBusyLabel(label);
    setNotice(null);
    try {
      const result = await work();
      await refreshData();
      return result;
    } catch (error) {
      setNotice({
        tone: "error",
        text: error instanceof Error ? error.message : "Request failed.",
      });
      return null;
    } finally {
      setBusyLabel(null);
    }
  }

  const isBusy = busyLabel !== null;

  /* ================================================================ */
  /*  Store handlers                                                   */
  /* ================================================================ */

  async function handleStoreSemantic(e: FormEvent) {
    e.preventDefault();
    if (!semContent.trim()) return;
    const result = await withBusy("Storing fact", () =>
      createSemanticMemory({
        content: semContent.trim(),
        importance: semImportance,
        category: semCategory,
        confidence: semConfidence,
      }),
    );
    if (result) {
      setSemContent("");
      setNotice({ tone: "ok", text: `Stored semantic memory ${result.record.id.slice(0, 8)}` });
    }
  }

  async function handleStoreTextEpisode(e: FormEvent) {
    e.preventDefault();
    if (!epiText.trim() || !epiSession.trim()) return;
    const result = await withBusy("Storing episode", () =>
      createTextEpisode({
        session_id: epiSession.trim(),
        text: epiText.trim(),
        summary: epiSummary.trim() || undefined,
        turn_number: epiTurn ? parseInt(epiTurn, 10) : undefined,
        importance: epiImportance,
      }),
    );
    if (result) {
      setEpiText("");
      setEpiSummary("");
      setEpiTurn("");
      setNotice({ tone: "ok", text: `Stored episode ${result.record.id.slice(0, 8)}` });
    }
  }

  async function handleStoreFileEpisode(e: FormEvent) {
    e.preventDefault();
    if (!epiFile || !epiSession.trim()) return;
    const result = await withBusy("Uploading episode", () =>
      createFileEpisode({
        session_id: epiSession.trim(),
        file: epiFile,
        content: epiFileContent.trim() || undefined,
        summary: epiSummary.trim() || undefined,
        importance: epiImportance,
      }),
    );
    if (result) {
      setEpiFile(null);
      setEpiFileContent("");
      setEpiSummary("");
      setNotice({ tone: "ok", text: `Stored file episode ${result.record.id.slice(0, 8)}` });
    }
  }

  async function storeExampleFact(content: string, category: string) {
    const result = await withBusy("Storing example", () =>
      createSemanticMemory({ content, category }),
    );
    if (result) setNotice({ tone: "ok", text: "Stored example fact" });
  }

  async function storeExampleEpisode(content: string, session: string) {
    const result = await withBusy("Storing example", () =>
      createTextEpisode({ session_id: session, text: content }),
    );
    if (result) setNotice({ tone: "ok", text: "Stored example episode" });
  }

  /* ================================================================ */
  /*  Query handlers                                                   */
  /* ================================================================ */

  async function runQuery(text: string) {
    if (!text.trim()) return;
    const result = await withBusy("Searching", () =>
      queryMemories({
        query: text.trim(),
        top_k: topK,
        memory_types: queryTypeFilter === "all" ? undefined : [queryTypeFilter],
      }),
    );
    if (result) {
      setQueryResults(result.results);
      setQuerySubmitted(true);
      setNotice({ tone: "ok", text: `${result.results.length} results` });
    }
  }

  async function runMediaQuery(file: File, modality: "image" | "audio") {
    const queryFn = modality === "image" ? queryByImage : queryByAudio;
    const result = await withBusy(`Searching by ${modality}`, () =>
      queryFn({
        file,
        top_k: topK,
        memory_types: queryTypeFilter === "all" ? undefined : [queryTypeFilter],
      }),
    );
    if (result) {
      setQueryResults(result.results);
      setQuerySubmitted(true);
      setNotice({ tone: "ok", text: `${result.results.length} results via ${modality}` });
    }
  }

  function handleQuery(e?: FormEvent) {
    e?.preventDefault();
    if (queryMode === "text") {
      void runQuery(queryText);
    } else if (queryFile) {
      void runMediaQuery(queryFile, queryMode);
    }
  }

  /* ================================================================ */
  /*  Explore handlers                                                 */
  /* ================================================================ */

  async function handleRecent(e: FormEvent) {
    e.preventDefault();
    const n = Math.max(1, parseInt(recentCount, 10) || 5);
    const result = await withBusy("Loading recent", () => getRecentEpisodes(n));
    if (result) {
      setExploreRecords(result.records);
      setExploreLabel(`${result.records.length} recent episodes`);
    }
  }

  async function handleSession(e: FormEvent) {
    e.preventDefault();
    if (!sessionFilter.trim()) return;
    const result = await withBusy("Loading session", () =>
      getSessionEpisodes(sessionFilter.trim()),
    );
    if (result) {
      setExploreRecords(result.records);
      setExploreLabel(`${result.records.length} episodes in ${sessionFilter.trim()}`);
    }
  }

  async function handleTimeRange(e: FormEvent) {
    e.preventDefault();
    if (!timeStart || !timeEnd) return;
    const result = await withBusy("Loading range", () =>
      getTimeRangeEpisodes(new Date(timeStart).toISOString(), new Date(timeEnd).toISOString()),
    );
    if (result) {
      setExploreRecords(result.records);
      setExploreLabel(`${result.records.length} episodes in range`);
    }
  }

  /* ================================================================ */
  /*  Render                                                           */
  /* ================================================================ */

  return (
    <div className="playground-layout">
      {/* ---- Toast ---- */}
      {notice && (
        <div
          className={`animate-toast fixed right-4 top-4 z-50 max-w-sm rounded-xl px-4 py-2.5 text-[13px] shadow-2xl backdrop-blur-md ${
            notice.tone === "error"
              ? "border border-red-500/20 bg-red-500/10 text-red-300"
              : "border border-emerald-500/20 bg-emerald-500/10 text-emerald-300"
          }`}
        >
          {notice.text}
        </div>
      )}

      {/* ---- Header ---- */}
      <header className="flex h-[52px] shrink-0 items-center justify-between border-b border-[var(--border)] bg-[var(--bg-alt)] px-5">
        <div className="flex items-center gap-3">
          <HealthDot status={apiHealth} />
          <span className="font-[family-name:var(--font-display)] text-lg text-[var(--text-1)]">
            Memory Playground
          </span>
        </div>
        <div className="flex items-center gap-5">
          <div className="mono-numeric hidden items-center gap-4 text-[11px] text-[var(--text-4)] sm:flex">
            <span>
              sem{" "}
              <span className="text-[var(--text-2)]">{overview?.semantic_count ?? "—"}</span>
            </span>
            <span>
              epi{" "}
              <span className="text-[var(--text-2)]">{overview?.episodic_count ?? "—"}</span>
            </span>
            <span>
              sess{" "}
              <span className="text-[var(--text-2)]">
                {overview?.recent_sessions.length ?? "—"}
              </span>
            </span>
            <button
              type="button"
              onClick={() => void refreshData()}
              className="text-[var(--text-4)] transition hover:text-[var(--text-2)]"
              title="Refresh stats"
            >
              <RefreshCcw className="size-3" />
            </button>
          </div>
          <div className="flex items-center gap-3">
            <a
              href={`${apiBaseUrl.replace(/\/$/, "")}/docs`}
              target="_blank"
              rel="noreferrer"
              className="text-[11px] text-[var(--text-4)] transition hover:text-[var(--text-2)]"
            >
              API docs
            </a>
            <a
              href="https://github.com/agentclash/agentic-memory"
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1 text-[11px] text-[var(--text-4)] transition hover:text-[var(--text-2)]"
            >
              <Layers3 className="size-3" />
              Source
            </a>
          </div>
        </div>
      </header>

      {/* ---- Body ---- */}
      <div className="playground-body">
        {/* ============================================================ */}
        {/*  Sidebar                                                      */}
        {/* ============================================================ */}
        <aside className="playground-sidebar">
          {/* sidebar tabs */}
          <div className="flex items-center gap-1 border-b border-[var(--border)] px-4 py-2">
            <TabButton active={sidebarTab === "store"} onClick={() => setSidebarTab("store")}>
              Store
            </TabButton>
            <TabButton active={sidebarTab === "explore"} onClick={() => setSidebarTab("explore")}>
              Explore
            </TabButton>
          </div>

          {/* ---- Store tab ---- */}
          {sidebarTab === "store" && (
            <>
              <div className="flex items-center gap-1 border-b border-[var(--border)] px-4 py-2">
                <TabButton
                  active={storeMode === "semantic"}
                  onClick={() => setStoreMode("semantic")}
                >
                  Semantic
                </TabButton>
                <TabButton active={storeMode === "text"} onClick={() => setStoreMode("text")}>
                  Text ep.
                </TabButton>
                <TabButton active={storeMode === "file"} onClick={() => setStoreMode("file")}>
                  File ep.
                </TabButton>
              </div>

              {/* Semantic form */}
              {storeMode === "semantic" && (
                <form className="space-y-4 p-5" onSubmit={handleStoreSemantic}>
                  <div>
                    <FormLabel>Fact</FormLabel>
                    <FieldTextarea
                      placeholder="The capital of France is Paris."
                      value={semContent}
                      onChange={(e) => setSemContent(e.target.value)}
                      rows={3}
                    />
                  </div>
                  <ImportanceSlider value={semImportance} onChange={setSemImportance} />
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <FormLabel>Category</FormLabel>
                      <select
                        value={semCategory}
                        onChange={(e) => setSemCategory(e.target.value)}
                        className="field-input"
                      >
                        {CATEGORIES.map((c) => (
                          <option key={c} value={c}>
                            {c}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <FormLabel>Confidence</FormLabel>
                      <FieldInput
                        type="number"
                        min="0"
                        max="1"
                        step="0.1"
                        value={semConfidence}
                        onChange={(e) => setSemConfidence(parseFloat(e.target.value) || 1)}
                      />
                    </div>
                  </div>
                  <SubmitButton disabled={isBusy}>
                    <Database className="size-3.5" />
                    Store fact
                  </SubmitButton>
                </form>
              )}

              {/* Text episode form */}
              {storeMode === "text" && (
                <form className="space-y-4 p-5" onSubmit={handleStoreTextEpisode}>
                  <div>
                    <FormLabel>Session ID</FormLabel>
                    <FieldInput
                      value={epiSession}
                      onChange={(e) => setEpiSession(e.target.value)}
                    />
                  </div>
                  <div>
                    <FormLabel>Episode text</FormLabel>
                    <FieldTextarea
                      placeholder="We debugged retrieval after a failed memory lookup."
                      value={epiText}
                      onChange={(e) => setEpiText(e.target.value)}
                      rows={3}
                    />
                  </div>
                  <div>
                    <FormLabel>Turn #</FormLabel>
                    <FieldInput
                      type="number"
                      min="0"
                      placeholder="—"
                      value={epiTurn}
                      onChange={(e) => setEpiTurn(e.target.value)}
                    />
                  </div>
                  <ImportanceSlider value={epiImportance} onChange={setEpiImportance} />
                  <div>
                    <FormLabel>Summary</FormLabel>
                    <FieldInput
                      placeholder="Optional retrieval context"
                      value={epiSummary}
                      onChange={(e) => setEpiSummary(e.target.value)}
                    />
                  </div>
                  <SubmitButton disabled={isBusy}>
                    <History className="size-3.5" />
                    Store episode
                  </SubmitButton>
                </form>
              )}

              {/* File episode form */}
              {storeMode === "file" && (
                <form className="space-y-4 p-5" onSubmit={handleStoreFileEpisode}>
                  <div>
                    <FormLabel>Session ID</FormLabel>
                    <FieldInput
                      value={epiSession}
                      onChange={(e) => setEpiSession(e.target.value)}
                    />
                  </div>
                  <div>
                    <FormLabel>File</FormLabel>
                    <FileDropZone file={epiFile} onFile={setEpiFile} />
                  </div>
                  <div>
                    <FormLabel>Content label</FormLabel>
                    <FieldInput
                      placeholder="Human-readable description"
                      value={epiFileContent}
                      onChange={(e) => setEpiFileContent(e.target.value)}
                    />
                  </div>
                  <ImportanceSlider value={epiImportance} onChange={setEpiImportance} />
                  <div>
                    <FormLabel>Summary</FormLabel>
                    <FieldInput
                      placeholder="Optional retrieval context"
                      value={epiSummary}
                      onChange={(e) => setEpiSummary(e.target.value)}
                    />
                  </div>
                  <SubmitButton disabled={isBusy}>
                    <FileUp className="size-3.5" />
                    Store file episode
                  </SubmitButton>
                </form>
              )}
            </>
          )}

          {/* ---- Explore tab ---- */}
          {sidebarTab === "explore" && (
            <>
              <div className="flex items-center gap-1 border-b border-[var(--border)] px-4 py-2">
                <TabButton
                  active={exploreMode === "recent"}
                  onClick={() => setExploreMode("recent")}
                >
                  Recent
                </TabButton>
                <TabButton
                  active={exploreMode === "session"}
                  onClick={() => setExploreMode("session")}
                >
                  Session
                </TabButton>
                <TabButton
                  active={exploreMode === "time"}
                  onClick={() => setExploreMode("time")}
                >
                  Time range
                </TabButton>
              </div>

              {exploreMode === "recent" && (
                <form className="space-y-4 p-5" onSubmit={handleRecent}>
                  <div>
                    <FormLabel>Count</FormLabel>
                    <FieldInput
                      type="number"
                      min="1"
                      max="50"
                      value={recentCount}
                      onChange={(e) => setRecentCount(e.target.value)}
                    />
                  </div>
                  <SubmitButton disabled={isBusy} small>
                    <History className="size-3" />
                    Load recent
                  </SubmitButton>
                </form>
              )}

              {exploreMode === "session" && (
                <form className="space-y-4 p-5" onSubmit={handleSession}>
                  <div>
                    <FormLabel>Session ID</FormLabel>
                    <FieldInput
                      value={sessionFilter}
                      onChange={(e) => setSessionFilter(e.target.value)}
                    />
                  </div>
                  <SubmitButton disabled={isBusy} small>
                    <Clock3 className="size-3" />
                    Load session
                  </SubmitButton>
                </form>
              )}

              {exploreMode === "time" && (
                <form className="space-y-4 p-5" onSubmit={handleTimeRange}>
                  <div>
                    <FormLabel>Start</FormLabel>
                    <FieldInput
                      type="datetime-local"
                      value={timeStart}
                      onChange={(e) => setTimeStart(e.target.value)}
                    />
                  </div>
                  <div>
                    <FormLabel>End</FormLabel>
                    <FieldInput
                      type="datetime-local"
                      value={timeEnd}
                      onChange={(e) => setTimeEnd(e.target.value)}
                    />
                  </div>
                  <SubmitButton disabled={isBusy} small>
                    <Clock3 className="size-3" />
                    Load range
                  </SubmitButton>
                </form>
              )}
            </>
          )}
        </aside>

        {/* ============================================================ */}
        {/*  Main content                                                 */}
        {/* ============================================================ */}
        <main className="playground-main">
          {/* ---- Query bar (sticky) ---- */}
          <div className="query-bar sticky top-0 z-10 border-b border-[var(--border)] px-6 py-4">
            <form className="space-y-3" onSubmit={handleQuery}>
              {/* Mode toggle */}
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => { setQueryMode("text"); setQueryFile(null); }}
                  className={`flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[11px] font-medium uppercase tracking-wider transition ${
                    queryMode === "text"
                      ? "bg-white/[0.08] text-[var(--text-1)]"
                      : "text-[var(--text-4)] hover:bg-white/[0.03] hover:text-[var(--text-2)]"
                  }`}
                >
                  <Type className="size-3" />
                  Text
                </button>
                <button
                  type="button"
                  onClick={() => { setQueryMode("image"); setQueryText(""); }}
                  className={`flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[11px] font-medium uppercase tracking-wider transition ${
                    queryMode === "image"
                      ? "bg-white/[0.08] text-[var(--text-1)]"
                      : "text-[var(--text-4)] hover:bg-white/[0.03] hover:text-[var(--text-2)]"
                  }`}
                >
                  <Image className="size-3" />
                  Image
                </button>
                <button
                  type="button"
                  onClick={() => { setQueryMode("audio"); setQueryText(""); }}
                  className={`flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[11px] font-medium uppercase tracking-wider transition ${
                    queryMode === "audio"
                      ? "bg-white/[0.08] text-[var(--text-1)]"
                      : "text-[var(--text-4)] hover:bg-white/[0.03] hover:text-[var(--text-2)]"
                  }`}
                >
                  <Music className="size-3" />
                  Audio
                </button>
              </div>

              {/* Text input */}
              {queryMode === "text" && (
                <div className="flex items-center gap-3">
                  <div className="relative flex-1">
                    <Search className="absolute left-3.5 top-1/2 size-4 -translate-y-1/2 text-[var(--text-4)]" />
                    <input
                      className="field-input pl-10"
                      placeholder="Query your memories..."
                      value={queryText}
                      onChange={(e) => setQueryText(e.target.value)}
                    />
                  </div>
                  <SubmitButton disabled={isBusy}>Search</SubmitButton>
                </div>
              )}

              {/* Media file input */}
              {queryMode !== "text" && (
                <div className="flex items-center gap-3">
                  <div className="flex-1">
                    <FileDropZone file={queryFile} onFile={setQueryFile} />
                  </div>
                  <SubmitButton disabled={isBusy || !queryFile}>
                    <Search className="size-3.5" />
                    Search
                  </SubmitButton>
                </div>
              )}

              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] uppercase tracking-wider text-[var(--text-4)]">
                    top_k
                  </span>
                  <select
                    value={topK}
                    onChange={(e) => setTopK(parseInt(e.target.value, 10))}
                    className="field-select"
                  >
                    {[3, 5, 6, 10, 15, 20].map((n) => (
                      <option key={n} value={n}>
                        {n}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] uppercase tracking-wider text-[var(--text-4)]">
                    types
                  </span>
                  <select
                    value={queryTypeFilter}
                    onChange={(e) =>
                      setQueryTypeFilter(e.target.value as "all" | "semantic" | "episodic")
                    }
                    className="field-select"
                  >
                    <option value="all">all</option>
                    <option value="semantic">semantic only</option>
                    <option value="episodic">episodic only</option>
                  </select>
                </div>
                {busyLabel && (
                  <span className="ml-auto animate-pulse text-[11px] text-[var(--text-4)]">
                    {busyLabel}...
                  </span>
                )}
              </div>
            </form>
          </div>

          {/* ---- Results area ---- */}
          <div className="space-y-4 p-6">
            {/* Query results */}
            {queryResults.length > 0 && (
              <div className="space-y-3">
                <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--text-4)]">
                  Query results · {queryResults.length} hits
                </p>
                {queryResults.map((r) => (
                  <ResultCard key={r.record.id} result={r} />
                ))}
              </div>
            )}

            {/* No results after query */}
            {queryResults.length === 0 && querySubmitted && (
              <div className="py-8 text-center">
                <p className="text-sm text-[var(--text-3)]">No memories matched your query.</p>
                <p className="mt-1 text-[12px] text-[var(--text-4)]">
                  Try a different query or store more memories.
                </p>
              </div>
            )}

            {/* Explore results */}
            {exploreRecords.length > 0 && (
              <div className="space-y-3">
                <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--text-4)]">
                  {exploreLabel}
                </p>
                {exploreRecords.map((r) => (
                  <MemoryCard key={r.id} record={r} />
                ))}
              </div>
            )}

            {/* ---- Onboarding empty state ---- */}
            {!querySubmitted && exploreRecords.length === 0 && (
              <div className="flex flex-col items-center justify-center py-16 text-center">
                <div className="mb-10 space-y-3">
                  <h2 className="font-[family-name:var(--font-display)] text-2xl text-[var(--text-1)]">
                    Query your memory store
                  </h2>
                  <p className="max-w-md text-sm leading-6 text-[var(--text-3)]">
                    Store some memories using the sidebar, then search here to see how the
                    retrieval system ranks them across semantic and episodic stores.
                  </p>
                </div>

                <div className="w-full max-w-lg space-y-8">
                  {/* Quick-add semantic facts */}
                  <div className="space-y-2">
                    <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--text-4)]">
                      Quick-add semantic facts
                    </p>
                    {EXAMPLE_FACTS.map((ex) => (
                      <button
                        key={ex.content}
                        type="button"
                        onClick={() => void storeExampleFact(ex.content, ex.category)}
                        disabled={isBusy}
                        className="glass-border flex w-full items-center gap-3 rounded-xl px-4 py-3 text-left transition hover:bg-white/[0.03] disabled:opacity-50"
                      >
                        <span className="badge-semantic shrink-0 rounded px-1.5 py-0.5 text-[9px] font-bold uppercase">
                          sem
                        </span>
                        <span className="text-[13px] text-[var(--text-2)]">{ex.content}</span>
                        <Zap className="ml-auto size-3.5 shrink-0 text-[var(--text-4)]" />
                      </button>
                    ))}
                  </div>

                  {/* Quick-add episodic memories */}
                  <div className="space-y-2">
                    <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--text-4)]">
                      Quick-add episodic memories
                    </p>
                    {EXAMPLE_EPISODES.map((ex) => (
                      <button
                        key={ex.content}
                        type="button"
                        onClick={() => void storeExampleEpisode(ex.content, ex.session)}
                        disabled={isBusy}
                        className="glass-border flex w-full items-center gap-3 rounded-xl px-4 py-3 text-left transition hover:bg-white/[0.03] disabled:opacity-50"
                      >
                        <span className="badge-episodic shrink-0 rounded px-1.5 py-0.5 text-[9px] font-bold uppercase">
                          epi
                        </span>
                        <span className="text-[13px] text-[var(--text-2)]">{ex.content}</span>
                        <Zap className="ml-auto size-3.5 shrink-0 text-[var(--text-4)]" />
                      </button>
                    ))}
                  </div>

                  {/* Suggested queries */}
                  <div className="space-y-3 pt-2">
                    <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--text-4)]">
                      Then try a query
                    </p>
                    <div className="flex flex-wrap justify-center gap-2">
                      {SUGGESTED_QUERIES.map((q) => (
                        <button
                          key={q}
                          type="button"
                          onClick={() => {
                            setQueryText(q);
                            void runQuery(q);
                          }}
                          className="glass-border rounded-lg px-3 py-1.5 text-[12px] text-[var(--text-3)] transition hover:bg-white/[0.04] hover:text-[var(--text-1)]"
                        >
                          {q}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </main>
      </div>

      {/* ---- Event drawer ---- */}
      <EventDrawer
        events={events}
        open={eventsOpen}
        onToggle={() => setEventsOpen(!eventsOpen)}
      />
    </div>
  );
}
