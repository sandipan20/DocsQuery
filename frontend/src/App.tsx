import { useEffect, useRef, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  AlertCircle,
  Check,
  Copy,
  Cpu,
  FileText,
  LoaderCircle,
  Moon,
  Search,
  Send,
  ShieldCheck,
  Sparkles,
  Sun,
  Terminal,
  Trash2,
  Upload,
  X,
} from "lucide-react"

import {
  askDocsQuery,
  deleteDocument,
  ensureSession,
  getDocuments,
  uploadDocuments,
} from "./lib/api"
import type { DocumentInfo } from "./types/api"
import "./App.css"

const SAMPLE_PROMPTS = [
  "What are the core concepts and findings covered in this document?",
  "Give me a structured summary with exact page citations.",
  "What key procedures, rules, or commands are documented?",
]

function App() {
  const queryClient = useQueryClient()
  const fileInput = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const [question, setQuestion] = useState("")
  const [scope, setScope] = useState<"all" | "selected">("all")
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  const [docFilter, setDocFilter] = useState("")
  const [copied, setCopied] = useState(false)

  const [theme, setTheme] = useState<"light" | "dark">(() => {
    try {
      const stored = localStorage.getItem("docsquery_theme")
      return (stored === "dark" || stored === "light") ? stored : "light"
    } catch {
      return "light"
    }
  })

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme)
    try {
      localStorage.setItem("docsquery_theme", theme)
    } catch {
      // Ignore in restricted environments
    }
  }, [theme])

  const toggleTheme = () => {
    setTheme((prev) => (prev === "light" ? "dark" : "light"))
  }

  const sessionQuery = useQuery({
    queryKey: ["session"],
    queryFn: ensureSession,
    retry: false,
  })

  const documentsQuery = useQuery({
    queryKey: ["documents"],
    queryFn: getDocuments,
    enabled: sessionQuery.isSuccess,
  })

  const refreshDocuments = async () => {
    await queryClient.invalidateQueries({ queryKey: ["documents"] })
  }

  const uploadMutation = useMutation({
    mutationFn: uploadDocuments,
    onSuccess: refreshDocuments,
  })

  const deleteMutation = useMutation({
    mutationFn: deleteDocument,
    onSuccess: async (_, documentId) => {
      setSelectedIds((ids) => ids.filter((id) => id !== documentId))
      await refreshDocuments()
    },
  })

  const queryMutation = useMutation({ mutationFn: askDocsQuery })
  const documents = documentsQuery.data ?? []

  const filteredDocuments = documents.filter((doc) =>
    doc.filename.toLowerCase().includes(docFilter.toLowerCase().trim())
  )

  const activeDocuments = scope === "all"
    ? documents
    : documents.filter((document) => selectedIds.includes(document.document_id))

  const canAsk = question.trim().length > 0
    && activeDocuments.length > 0
    && !queryMutation.isPending

  function toggleDocument(documentId: string) {
    setSelectedIds((ids) => ids.includes(documentId)
      ? ids.filter((id) => id !== documentId)
      : [...ids, documentId])
  }

  function handleSelectAll() {
    if (selectedIds.length === documents.length) {
      setSelectedIds([])
    } else {
      setSelectedIds(documents.map((d) => d.document_id))
    }
  }

  function handleFilesSelected(event: React.ChangeEvent<HTMLInputElement>) {
    const files = event.target.files
    if (files?.length) uploadMutation.mutate(Array.from(files))
    event.target.value = ""
  }

  function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!canAsk) return
    queryMutation.mutate({
      query: question.trim(),
      top_k: 5,
      ...(scope === "selected" ? { document_ids: selectedIds } : {}),
    })
  }

  const handleCopyAnswer = async () => {
    if (!queryMutation.data?.answer) return
    try {
      await navigator.clipboard.writeText(queryMutation.data.answer)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // Fallback
    }
  }

  const handleUsePrompt = (promptText: string) => {
    setQuestion(promptText)
    textareaRef.current?.focus()
  }

  const pageError = sessionQuery.error ?? documentsQuery.error

  return (
    <main className={`workspace-shell ${theme === "light" ? "theme-light" : ""}`}>
      <aside className="library-panel">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true">
            <Cpu size={20} />
          </div>
          <div>
            <p className="brand-name">DOCSQUERY // OS</p>
            <p className="brand-caption">NEURAL EVIDENCE INTERFACE</p>
          </div>
        </div>

        <div className="library-heading">
          <div>
            <p className="eyebrow">// DATA VAULT</p>
            <h1>DOCUMENTS</h1>
          </div>
          <span className="document-count">{documents.length}</span>
        </div>

        <input
          ref={fileInput}
          className="visually-hidden"
          type="file"
          accept="application/pdf,.pdf"
          multiple
          onChange={handleFilesSelected}
          aria-label="Choose PDF files to upload"
        />
        <button
          className="upload-button"
          type="button"
          disabled={!sessionQuery.isSuccess || uploadMutation.isPending}
          onClick={() => fileInput.current?.click()}
        >
          {uploadMutation.isPending
            ? <LoaderCircle className="spin" size={17} />
            : <Upload size={17} />}
          <span>{uploadMutation.isPending ? "INJECTING SOURCE..." : "ADD PDF EVIDENCE"}</span>
        </button>

        {documents.length > 2 && (
          <div className="vault-filter-wrap">
            <Search size={13} className="filter-icon" />
            <input
              type="text"
              className="vault-filter-input"
              placeholder="Filter mounted documents..."
              value={docFilter}
              onChange={(e) => setDocFilter(e.target.value)}
              aria-label="Filter documents"
            />
            {docFilter && (
              <button
                type="button"
                className="filter-clear-btn"
                onClick={() => setDocFilter("")}
                aria-label="Clear filter"
              >
                <X size={12} />
              </button>
            )}
          </div>
        )}

        <div className="document-list" aria-label="Current-session documents">
          {documentsQuery.isLoading && (
            <div className="list-state"><LoaderCircle className="spin" size={17} /> MOUNTING STORAGE...</div>
          )}
          {!documentsQuery.isLoading && documents.length === 0 && (
            <div className="empty-library">
              <div className="empty-icon"><FileText size={21} /></div>
              <p>NO DOCUMENTS MOUNTED</p>
              <span>Add PDF sources to initialize vector + BM25 neural search indices.</span>
            </div>
          )}
          {filteredDocuments.map((document) => (
            <DocumentRow
              key={document.document_id}
              document={document}
              checked={selectedIds.includes(document.document_id)}
              onToggle={() => toggleDocument(document.document_id)}
              onDelete={() => deleteMutation.mutate(document.document_id)}
              deleting={deleteMutation.isPending && deleteMutation.variables === document.document_id}
              disabled={!sessionQuery.isSuccess}
            />
          ))}
        </div>

        <div className="library-footer">
          <span className="status-dot" />
          <span>
            {sessionQuery.isSuccess ? (
              <>
                <ShieldCheck size={14} style={{ display: "inline", marginRight: 5, verticalAlign: "middle" }} />
                SESSION ACTIVE // ENCRYPTED
              </>
            ) : (
              "INITIALIZING NEURAL SESSION..."
            )}
          </span>
        </div>
      </aside>

      <section className="chat-panel">
        <header className="topbar">
          <div>
            <p className="eyebrow">// NEURAL GROUNDING ENGINE</p>
            <h2>QUERY INTELLIGENCE HUD</h2>
          </div>
          <div className="topbar-actions">
            <div className="scope-summary">
              <span className="scope-indicator" />
              <span>{scope === "all" ? `SCOPE: ALL (${documents.length})` : `SCOPE: FILTERED (${activeDocuments.length})`}</span>
            </div>
            <button
              type="button"
              className="theme-toggle"
              onClick={toggleTheme}
              title={`Switch to ${theme === "light" ? "Dark Cyber" : "Light Holo"} theme`}
              aria-label="Toggle visual theme"
            >
              {theme === "light" ? (
                <>
                  <Moon size={13} />
                  <span>DARK CYBER</span>
                </>
              ) : (
                <>
                  <Sun size={13} />
                  <span>HOLO LIGHT</span>
                </>
              )}
            </button>
          </div>
        </header>

        <div className="conversation-area">
          {pageError && <ErrorNotice message={pageError.message} />}
          {uploadMutation.isError && <ErrorNotice message={uploadMutation.error.message} />}
          {deleteMutation.isError && <ErrorNotice message={deleteMutation.error.message} />}

          {!queryMutation.data && !queryMutation.isPending && !pageError && (
            <div className="welcome-state">
              <span className="welcome-index">
                <Terminal size={14} style={{ display: "inline", marginRight: 6, verticalAlign: "middle" }} />
                SYSTEM CORE ONLINE // ZERO HALLUCINATION DIRECTIVE
              </span>
              <h3>Knowledge, with its sources attached.</h3>
              <p>Every response is strictly grounded in retrieved vector passages and BM25 keywords. Citations are verified and linked to page coordinates in real time.</p>

              {documents.length > 0 && (
                <div className="prompt-suggestions">
                  <span className="suggestions-label">// QUICK QUERY DIRECTIVES</span>
                  <div className="suggestion-chips">
                    {SAMPLE_PROMPTS.map((promptText) => (
                      <button
                        key={promptText}
                        type="button"
                        className="prompt-chip"
                        onClick={() => handleUsePrompt(promptText)}
                      >
                        <Sparkles size={13} />
                        <span>{promptText}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {queryMutation.isPending && (
            <div className="answer-loading" role="status">
              <LoaderCircle className="spin" size={20} />
              <div><strong>SCANNING NEURAL VECTORS</strong><span>Retrieving candidate passages & cross-encoder reranking</span></div>
            </div>
          )}
          {queryMutation.isError && <ErrorNotice message={queryMutation.error.message} />}
          {queryMutation.data && (
            <article className="answer-result">
              <div className="answer-meta">
                <div className="meta-left">
                  <span>SYNTHESIZED INTEL</span>
                  <span className="latency-badge">EXEC LATENCY: {queryMutation.data.metrics.total_latency_ms.toFixed(0)} MS</span>
                </div>
                <button
                  type="button"
                  className="copy-intel-btn"
                  onClick={handleCopyAnswer}
                  title="Copy answer text to clipboard"
                >
                  {copied ? (
                    <>
                      <Check size={13} style={{ color: "var(--green)" }} />
                      <span>COPIED TO CLIPBOARD</span>
                    </>
                  ) : (
                    <>
                      <Copy size={13} />
                      <span>COPY INTEL</span>
                    </>
                  )}
                </button>
              </div>
              <p className="answer-text">{queryMutation.data.answer}</p>
              {queryMutation.data.citations.length > 0 && (
                <div className="citation-section">
                  <p className="eyebrow">VERIFIED SOURCE CITATIONS</p>
                  <div className="citation-list">
                    {queryMutation.data.citations.map((citation) => (
                      <div className="citation-row" key={citation.citation_id}>
                        <span className="citation-token">[{citation.citation_id}]</span>
                        <FileText size={16} />
                        <strong>{citation.source}</strong>
                        <span>Page {citation.page_number}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </article>
          )}
        </div>

        <div className="composer-wrap">
          <div className="source-picker">
            <span className="eyebrow">VECTOR TARGETS</span>
            <div className="scope-toggle" role="group" aria-label="Document search scope">
              <button type="button" className={scope === "all" ? "selected" : ""} onClick={() => setScope("all")}>
                ALL SOURCES ({documents.length})
              </button>
              <button
                type="button"
                className={scope === "selected" ? "selected" : ""}
                onClick={() => setScope("selected")}
                disabled={documents.length === 0}
              >
                SELECTED ({selectedIds.length})
              </button>
            </div>
            {scope === "selected" && (
              <button
                type="button"
                className="theme-toggle"
                style={{ minHeight: "26px", fontSize: "9px", padding: "0 8px" }}
                onClick={handleSelectAll}
              >
                {selectedIds.length === documents.length ? "CLEAR ALL" : "SELECT ALL"}
              </button>
            )}
            {scope === "selected" && selectedIds.length === 0 && (
              <span className="scope-hint">Select one or more files in your vault.</span>
            )}
          </div>
          <form className="question-form" onSubmit={handleSubmit}>
            <textarea
              ref={textareaRef}
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault()
                  event.currentTarget.form?.requestSubmit()
                }
              }}
              placeholder="Enter query vector (e.g. 'What is calculus?', 'Summarize Section 2')..."
              aria-label="Question"
              maxLength={1000}
              disabled={!sessionQuery.isSuccess}
            />
            <div className="composer-bottom">
              <span>[ENTER] EXECUTE QUERY · [SHIFT + ENTER] NEW LINE</span>
              <button type="submit" aria-label="Send question" title="Send question" disabled={!canAsk}>
                {queryMutation.isPending ? <LoaderCircle className="spin" size={17} /> : <Send size={17} />}
              </button>
            </div>
          </form>
          {activeDocuments.length === 0 && <p className="no-sources">// Inject a PDF document before executing queries.</p>}
        </div>
      </section>
    </main>
  )
}

function DocumentRow({
  document,
  checked,
  onToggle,
  onDelete,
  deleting,
  disabled,
}: {
  document: DocumentInfo
  checked: boolean
  onToggle: () => void
  onDelete: () => void
  deleting: boolean
  disabled: boolean
}) {
  return (
    <div className={`document-row${checked ? " is-checked" : ""}`}>
      <label className="document-select">
        <input type="checkbox" checked={checked} onChange={onToggle} disabled={disabled} />
        <span className="checkmark" aria-hidden="true" />
        <span className="file-icon"><FileText size={17} /></span>
        <span className="document-label">
          <strong title={document.filename}>{document.filename}</strong>
          <small>{document.page_count} pages · {document.chunk_count} passages</small>
        </span>
      </label>
      <button
        className="icon-button delete-button"
        type="button"
        aria-label={`Delete ${document.filename}`}
        title={`Delete ${document.filename}`}
        onClick={onDelete}
        disabled={deleting}
      >
        {deleting ? <LoaderCircle className="spin" size={16} /> : <Trash2 size={16} />}
      </button>
    </div>
  )
}

function ErrorNotice({ message }: { message: string }) {
  return <div className="error-notice"><AlertCircle size={17} /><span>{message}</span></div>
}

export default App