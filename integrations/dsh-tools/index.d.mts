export declare const FASTRAG_TOOLS: Readonly<{
  searchDocuments: 'search_documents'
  queryListings: 'query_listings'
  getListings: 'get_listings'
}>
export type FastragToolName = (typeof FASTRAG_TOOLS)[keyof typeof FASTRAG_TOOLS]
export type FastragProfile = 'customer' | 'handoff_copilot' | 'internal_search'

export declare const DEFAULT_BASE_URL: string
export declare const DEFAULT_API_KEY_ENV: string

/** Plugin config. Plain JSON, so it can go in a dsh profile patch. Never put the key itself here. */
export interface FastragToolsConfig {
  /** fastRag API root. Default http://127.0.0.1:8000 */
  baseUrl?: string
  /** Name of the environment variable that holds the tenant API key. Default FASTRAG_API_KEY */
  apiKeyEnv?: string
  /** Retrieval profile. Default "customer" (public content only). */
  profile?: FastragProfile
  /** Tools to register. Default: all three. */
  tools?: FastragToolName[]
  /** Collection slugs per kind. Default: all collections of that kind. */
  collections?: { documents?: string[]; listings?: string[] }
  /** Passages per search (1-20). Default 5. */
  topK?: number
  /** HTTP timeout per call. Default 15000. */
  timeoutMs?: number
  /** Passage text is cut to this length for the model. Default 1500. */
  maxPassageChars?: number
  /** Replace the model-facing description of a tool, e.g. to match the agent's language. */
  descriptions?: Partial<Record<FastragToolName, string>>
}

export interface ResolvedFastragConfig {
  readonly baseUrl: string
  readonly apiKey: string
  readonly profile: FastragProfile
  readonly tools: readonly FastragToolName[]
  readonly documentCollections: string[] | undefined
  readonly listingCollections: string[] | undefined
  readonly topK: number
  readonly timeoutMs: number
  readonly maxPassageChars: number
  readonly descriptions: Readonly<Record<FastragToolName, string>>
}

export declare class FastragError extends Error {
  readonly status: number | undefined
  constructor(message: string, options?: { status?: number; cause?: unknown })
}

export interface FastragDeps {
  fetch?: typeof fetch
  env?: Record<string, string | undefined>
}

export declare function parseFastragConfig(raw: unknown, env?: Record<string, string | undefined>): ResolvedFastragConfig

export interface FastragClient {
  search(body: Record<string, unknown>, signal?: AbortSignal): Promise<unknown>
  queryListings(body: Record<string, unknown>, signal?: AbortSignal): Promise<unknown>
  lookupListings(body: Record<string, unknown>, signal?: AbortSignal): Promise<unknown>
}
export declare function createFastragClient(
  config: { baseUrl: string; apiKey: string; timeoutMs: number },
  deps?: { fetch?: typeof fetch },
): FastragClient

/**
 * Builds the tool definitions with the host's `defineTool`
 * (`import { defineTool } from '@deepseek-ai/dsh-tools'`).
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export declare function createFastragTools<T>(defineTool: (options: any) => T, config: FastragToolsConfig | unknown, deps?: FastragDeps): T[]

export interface SearchPassage {
  id: string
  title: string
  heading: string | null
  text: string
  document_id: string | null
  collection: string | null
  uri: string | null
  page_start: number | null
  page_end: number | null
  updated: string | null
  age_days: number | null
  stale: boolean
  score: number | null
}
export interface SearchResult {
  trace_id: string | null
  nothing_relevant: boolean
  reason: string | null
  notes: string[]
  passages: SearchPassage[]
}
export interface ListingsResult {
  trace_id: string | null
  total: number
  as_of: string | null
  applied_filters: Record<string, unknown>
  items: Record<string, unknown>[]
}
export declare function toSearchResult(response: unknown, maxChars: number): SearchResult
export declare function renderSearch(result: SearchResult): string
export declare function toListingsResult(response: unknown): ListingsResult
export declare function renderListings(result: ListingsResult): string
export declare function formatVnd(vnd: number): string
