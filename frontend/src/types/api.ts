/**
 * Request sent to the DocsQuery query endpoint.
 */
export interface QueryRequest {
  query: string
  top_k?: number
  document_ids?: string[]
}

export interface DocumentInfo {
  document_id: string
  filename: string
  page_count: number
  chunk_count: number
}

/**
 * One citation returned by DocsQuery.
 */
export interface Citation {
  citation_id: string
  source: string
  page_number: number
  chunk_id: string
}

/**
 * Latency information returned by the backend.
 */
export interface QueryMetrics {
  retrieval_latency_ms: number
  generation_latency_ms: number
  total_latency_ms: number
}

/**
 * Response returned by POST /api/v1/query.
 */
export interface QueryResponse {
  query: string
  answer: string
  citations: Citation[]
  metrics: QueryMetrics
}