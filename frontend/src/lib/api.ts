import type {
  DocumentInfo,
  QueryRequest,
  QueryResponse,
} from "../types/api"

const API_BASE_URL = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "")

function apiUrl(path: string): string {
  return `${API_BASE_URL}${path}`
}

function getStoredSessionId(): string | null {
  try {
    return sessionStorage.getItem("docsquery_session_id")
  } catch {
    return null
  }
}

function storeSessionId(id: string | null | undefined): void {
  if (id && typeof id === "string") {
    try {
      sessionStorage.setItem("docsquery_session_id", id)
    } catch {
      // Ignore storage errors in restricted contexts
    }
  }
}

function getHeaders(extraHeaders: Record<string, string> = {}): Record<string, string> {
  const headers: Record<string, string> = { ...extraHeaders }
  const sessionId = getStoredSessionId()
  if (sessionId) {
    headers["X-Session-ID"] = sessionId
  }
  return headers
}

function captureSessionFromResponse(response: Response): void {
  const headerSessionId = response.headers.get("x-session-id")
  if (headerSessionId) {
    storeSessionId(headerSessionId)
  }
}

async function throwResponseError(response: Response): Promise<never> {
  let message = "The request could not be completed."
  try {
    const body = await response.json() as { detail?: unknown }
    if (typeof body.detail === "string") message = body.detail
  } catch {
    // Use the generic message for non-JSON error responses.
  }
  throw new Error(message)
}

export async function ensureSession(): Promise<{ session_ready: boolean; session_id?: string }> {
  const response = await fetch(apiUrl("/api/v1/session"), {
    credentials: "include",
    headers: getHeaders(),
  })
  if (!response.ok) await throwResponseError(response)
  captureSessionFromResponse(response)
  const data = await response.json() as { session_ready: boolean; session_id?: string }
  if (data.session_id) {
    storeSessionId(data.session_id)
  }
  return data
}

export async function getDocuments(): Promise<DocumentInfo[]> {
  const response = await fetch(apiUrl("/api/v1/documents"), {
    credentials: "include",
    headers: getHeaders(),
  })
  if (!response.ok) await throwResponseError(response)
  captureSessionFromResponse(response)
  return response.json() as Promise<DocumentInfo[]>
}

export async function uploadDocuments(files: File[]): Promise<DocumentInfo[]> {
  const body = new FormData()
  files.forEach((file) => body.append("files", file))
  const response = await fetch(apiUrl("/api/v1/documents"), {
    method: "POST",
    credentials: "include",
    headers: getHeaders(),
    body,
  })
  if (!response.ok) await throwResponseError(response)
  captureSessionFromResponse(response)
  return response.json() as Promise<DocumentInfo[]>
}

export async function deleteDocument(documentId: string): Promise<void> {
  const response = await fetch(apiUrl(`/api/v1/documents/${encodeURIComponent(documentId)}`), {
    method: "DELETE",
    credentials: "include",
    headers: getHeaders(),
  })
  if (!response.ok) await throwResponseError(response)
  captureSessionFromResponse(response)
}

export async function askDocsQuery(request: QueryRequest): Promise<QueryResponse> {
  const response = await fetch(apiUrl("/api/v1/query"), {
    method: "POST",
    credentials: "include",
    headers: getHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(request),
  })
  if (!response.ok) await throwResponseError(response)
  captureSessionFromResponse(response)
  return response.json() as Promise<QueryResponse>
}