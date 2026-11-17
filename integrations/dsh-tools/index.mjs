/**
 * fastRag retrieval tools for DeepSeek Harness (dsh) and any host built on
 * `@deepseek-ai/dsh-tools`.
 *
 * The module has no dependencies. The host passes in its own `defineTool`, so
 * it can be bundled into an existing tools plugin (esbuild, tsdown, ...) or
 * loaded through `./plugin.mjs`. Tools call the fastRag HTTP API with a tenant
 * API key; the key's access-group ceiling applies as usual.
 *
 * Tool names match fastRag's MCP server and `GET /v1/tools`.
 */

export const FASTRAG_TOOLS = Object.freeze({
  searchDocuments: 'search_documents',
  queryListings: 'query_listings',
  getListings: 'get_listings',
})

export const DEFAULT_BASE_URL = 'http://127.0.0.1:8000'
export const DEFAULT_API_KEY_ENV = 'FASTRAG_API_KEY'

const TOOL_NAMES = Object.values(FASTRAG_TOOLS)
const PROFILES = ['customer', 'handoff_copilot', 'internal_search']
const SORTS = ['price_asc', 'price_desc', 'area_asc', 'area_desc', 'price_per_m2_asc', 'newest']
const MAX_LISTINGS = 20
const MAX_IDS = 20

const DEFAULT_DESCRIPTIONS = {
  [FASTRAG_TOOLS.searchDocuments]:
    "Search the company's documents (policies, FAQs, project information, procedures). Pass one standalone " +
    'question in the language of the documents (for Vietnamese documents, write it in Vietnamese). Returns ' +
    'numbered passages with sources. Answer only from the passages; if the result says nothing relevant was ' +
    'found, do not answer from it.',
  [FASTRAG_TOOLS.queryListings]:
    'Find listings (properties, products) with filters. Prices, areas and districts may be written as people ' +
    'say them, e.g. "3,5 tỷ", "850tr", "70m2", "Q7". Each result has an id and an as-of date; quote the date ' +
    'when stating price or availability.',
  [FASTRAG_TOOLS.getListings]:
    'Refresh listings by the ids returned earlier by query_listings (current price, status and as-of date).',
}

/** A fastRag failure the model should see as the tool's error message. */
export class FastragError extends Error {
  /**
   * @param {string} message
   * @param {{ status?: number, cause?: unknown }} [options]
   */
  constructor(message, options = {}) {
    super(message, options.cause === undefined ? undefined : { cause: options.cause })
    this.name = 'FastragError'
    this.status = options.status
  }
}

/**
 * Validates plugin config and resolves the API key from the environment.
 * Throws on anything wrong so configuration errors surface when the host starts.
 *
 * @param {unknown} raw
 * @param {Record<string, string | undefined>} [env]
 */
export function parseFastragConfig(raw, env = process.env) {
  const value = /** @type {Record<string, unknown>} */ (raw ?? {})
  if (typeof value !== 'object' || Array.isArray(value)) throw configError('config must be an object')

  const baseUrl = optionalString(value.baseUrl, 'baseUrl') ?? DEFAULT_BASE_URL
  let parsedUrl
  try {
    parsedUrl = new URL(baseUrl)
  } catch {
    throw configError(`baseUrl is not a URL: ${JSON.stringify(baseUrl)}`)
  }
  if (parsedUrl.protocol !== 'http:' && parsedUrl.protocol !== 'https:') {
    throw configError('baseUrl must be http or https')
  }

  const apiKeyEnv = optionalString(value.apiKeyEnv, 'apiKeyEnv') ?? DEFAULT_API_KEY_ENV
  const apiKey = env[apiKeyEnv]?.trim()
  if (!apiKey) throw configError(`environment variable ${apiKeyEnv} must hold a fastRag API key`)

  const profile = optionalString(value.profile, 'profile') ?? 'customer'
  if (!PROFILES.includes(profile)) throw configError(`profile must be one of ${PROFILES.join(', ')}`)

  const tools = optionalStringList(value.tools, 'tools') ?? TOOL_NAMES
  for (const name of tools) {
    if (!TOOL_NAMES.includes(name)) throw configError(`unknown tool ${JSON.stringify(name)}; known: ${TOOL_NAMES.join(', ')}`)
  }
  if (tools.length === 0) throw configError('tools must name at least one tool')

  const collections = /** @type {Record<string, unknown>} */ (value.collections ?? {})
  if (typeof collections !== 'object' || Array.isArray(collections)) {
    throw configError('collections must be an object like { "documents": ["docs"], "listings": ["listings"] }')
  }

  const descriptions = /** @type {Record<string, unknown>} */ (value.descriptions ?? {})
  if (typeof descriptions !== 'object' || Array.isArray(descriptions)) throw configError('descriptions must be an object')
  for (const [name, text] of Object.entries(descriptions)) {
    if (!TOOL_NAMES.includes(name)) throw configError(`descriptions: unknown tool ${JSON.stringify(name)}`)
    if (typeof text !== 'string' || !text.trim()) throw configError(`descriptions.${name} must be a non-empty string`)
  }

  return Object.freeze({
    baseUrl: baseUrl.replace(/\/+$/, ''),
    apiKey,
    profile,
    tools: Object.freeze([...new Set(tools)]),
    documentCollections: optionalStringList(collections.documents, 'collections.documents'),
    listingCollections: optionalStringList(collections.listings, 'collections.listings'),
    topK: optionalInt(value.topK, 'topK', 1, 20) ?? 5,
    timeoutMs: optionalInt(value.timeoutMs, 'timeoutMs', 500, 120_000) ?? 15_000,
    maxPassageChars: optionalInt(value.maxPassageChars, 'maxPassageChars', 200, 20_000) ?? 1_500,
    descriptions: Object.freeze({ ...DEFAULT_DESCRIPTIONS, ...descriptions }),
  })
}

/**
 * A minimal fastRag API client.
 *
 * @param {{ baseUrl: string, apiKey: string, timeoutMs: number }} config
 * @param {{ fetch?: typeof fetch }} [deps]
 */
export function createFastragClient(config, deps = {}) {
  const fetchImpl = deps.fetch ?? globalThis.fetch
  const { baseUrl, apiKey, timeoutMs } = config

  /** @param {string} path @param {unknown} body @param {AbortSignal | undefined} signal */
  async function post(path, body, signal) {
    const timeout = AbortSignal.timeout(timeoutMs)
    let response
    try {
      response = await fetchImpl(`${baseUrl}${path}`, {
        method: 'POST',
        headers: { authorization: `Bearer ${apiKey}`, 'content-type': 'application/json', accept: 'application/json' },
        body: JSON.stringify(body),
        signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
      })
    } catch (error) {
      if (signal?.aborted) throw error
      if (timeout.aborted) throw new FastragError(`fastRag did not answer within ${timeoutMs} ms.`, { cause: error })
      throw new FastragError(`fastRag is unreachable at ${baseUrl}.`, { cause: error })
    }
    if (response.ok) return response.json()

    const detail = await errorDetail(response)
    if (response.status === 401 || response.status === 403) {
      throw new FastragError(`fastRag refused the API key (HTTP ${response.status}): ${detail}`, { status: response.status })
    }
    if (response.status === 422) {
      throw new FastragError(`fastRag could not use these arguments: ${detail}`, { status: 422 })
    }
    throw new FastragError(`fastRag failed (HTTP ${response.status}): ${detail}`, { status: response.status })
  }

  return {
    /** @param {Record<string, unknown>} body @param {AbortSignal} [signal] */
    search: (body, signal) => post('/v1/search', body, signal),
    /** @param {Record<string, unknown>} body @param {AbortSignal} [signal] */
    queryListings: (body, signal) => post('/v1/listings/query', body, signal),
    /** @param {Record<string, unknown>} body @param {AbortSignal} [signal] */
    lookupListings: (body, signal) => post('/v1/listings/lookup', body, signal),
  }
}

/**
 * Builds the fastRag tool definitions.
 *
 * @template T
 * @param {(options: any) => T} defineTool the host's `defineTool` from `@deepseek-ai/dsh-tools`
 * @param {unknown} rawConfig plugin config, see README
 * @param {{ fetch?: typeof fetch, env?: Record<string, string | undefined> }} [deps]
 * @returns {T[]}
 */
export function createFastragTools(defineTool, rawConfig, deps = {}) {
  if (typeof defineTool !== 'function') throw new TypeError('createFastragTools: pass defineTool from @deepseek-ai/dsh-tools')
  const config = parseFastragConfig(rawConfig, deps.env)
  const client = createFastragClient(config, deps)
  const enabled = new Set(config.tools)
  /** @type {T[]} */
  const tools = []
  const common = { timeoutMs: config.timeoutMs + 1_000, isConcurrencySafe: () => true }
  const json = { type: 'json' }

  if (enabled.has(FASTRAG_TOOLS.searchDocuments)) {
    tools.push(
      defineTool({
        ...common,
        name: FASTRAG_TOOLS.searchDocuments,
        description: config.descriptions[FASTRAG_TOOLS.searchDocuments],
        parameters: {
          query: { type: 'string', required: true, description: 'One standalone question, e.g. "Phí quản lý chung cư là bao nhiêu?"' },
        },
        output: {
          schema: json,
          render: (/** @type {unknown} */ _args, /** @type {any} */ value) => text(renderSearch(value)),
          presentationMeta: (/** @type {unknown} */ _args, /** @type {any} */ value) => searchMeta(value),
        },
        /** @param {{ query: string }} args @param {{ signal?: AbortSignal }} exec */
        async execute(args, exec) {
          const query = args.query.trim()
          if (!query) throw new FastragError('query must not be empty.')
          const response = await client.search(
            { query, profile: config.profile, top_k: config.topK, collections: config.documentCollections ?? null },
            exec?.signal,
          )
          return toSearchResult(response, config.maxPassageChars)
        },
      }),
    )
  }

  if (enabled.has(FASTRAG_TOOLS.queryListings)) {
    tools.push(
      defineTool({
        ...common,
        name: FASTRAG_TOOLS.queryListings,
        description: config.descriptions[FASTRAG_TOOLS.queryListings],
        parameters: {
          price_min: { type: 'string', description: 'Minimum price, e.g. "2 tỷ", "850tr".' },
          price_max: { type: 'string', description: 'Maximum price, e.g. "3,5 tỷ".' },
          area_min: { type: 'string', description: 'Minimum area, e.g. "60m2".' },
          area_max: { type: 'string', description: 'Maximum area.' },
          bedrooms_min: { type: 'integer', description: 'Minimum number of bedrooms.' },
          bedrooms_max: { type: 'integer', description: 'Maximum number of bedrooms.' },
          district: { type: 'string', description: 'District, e.g. "Q7", "Quận 7", "Thủ Đức".' },
          project: { type: 'string', description: 'Project name.' },
          property_type: { type: 'string', description: 'Property type, e.g. "căn hộ", "nhà phố".' },
          status: { type: 'string', description: 'Status, e.g. "còn hàng" (available).' },
          direction: { type: 'string', description: 'Facing direction, e.g. "Đông Nam".' },
          keywords: { type: 'string', description: 'Other words to match.' },
          sort: { type: 'string', enum: SORTS, description: 'Sort order.' },
          limit: { type: 'integer', description: `Number of results (1-${MAX_LISTINGS}, default 5).` },
        },
        output: {
          schema: json,
          render: (/** @type {unknown} */ _args, /** @type {any} */ value) => text(renderListings(value)),
          presentationMeta: (/** @type {unknown} */ _args, /** @type {any} */ value) => listingsMeta(value),
        },
        /** @param {Record<string, any>} args @param {{ signal?: AbortSignal }} exec */
        async execute(args, exec) {
          const { sort, limit, ...rest } = args
          /** @type {Record<string, unknown>} */
          const filters = {}
          for (const [key, v] of Object.entries(rest)) {
            if (v === undefined || v === null || (typeof v === 'string' && !v.trim())) continue
            filters[key] = typeof v === 'string' ? v.trim() : v
          }
          const response = await client.queryListings(
            {
              filters,
              sort: sort ?? null,
              limit: clamp(limit ?? 5, 1, MAX_LISTINGS),
              profile: config.profile,
              collections: config.listingCollections ?? null,
            },
            exec?.signal,
          )
          return toListingsResult(response)
        },
      }),
    )
  }

  if (enabled.has(FASTRAG_TOOLS.getListings)) {
    tools.push(
      defineTool({
        ...common,
        name: FASTRAG_TOOLS.getListings,
        description: config.descriptions[FASTRAG_TOOLS.getListings],
        parameters: {
          ids: { type: 'array', items: { type: 'string' }, required: true, description: 'Listing ids from query_listings.' },
        },
        output: {
          schema: json,
          render: (/** @type {unknown} */ _args, /** @type {any} */ value) => text(renderListings(value)),
          presentationMeta: (/** @type {unknown} */ _args, /** @type {any} */ value) => listingsMeta(value),
        },
        /** @param {{ ids: string[] }} args @param {{ signal?: AbortSignal }} exec */
        async execute(args, exec) {
          const ids = [...new Set(args.ids.map((id) => id.trim()).filter(Boolean))]
          if (ids.length === 0) throw new FastragError('ids must contain at least one listing id from query_listings.')
          if (ids.length > MAX_IDS) throw new FastragError(`At most ${MAX_IDS} ids per call.`)
          const response = await client.lookupListings(
            { ids, profile: config.profile, collections: config.listingCollections ?? null },
            exec?.signal,
          )
          return toListingsResult(response)
        },
      }),
    )
  }

  return tools
}

// ---------------------------------------------------------------------------
// Canonical values (plain JSON) and their model-facing rendering.

/** @param {any} response @param {number} maxChars */
export function toSearchResult(response, maxChars) {
  const signals = response?.signals ?? {}
  const passages = Array.isArray(response?.passages) ? response.passages : []
  const numberById = new Map(passages.map((/** @type {any} */ p, /** @type {number} */ i) => [String(p.id), i + 1]))
  const passageNumber = (/** @type {unknown} */ chunkId) => numberById.get(`chunk:${chunkId}`) ?? numberById.get(String(chunkId))

  /** @type {string[]} */
  const notes = []
  for (const conflict of Array.isArray(signals.conflicts) ? signals.conflicts : []) {
    const numbers = (conflict.passages ?? []).map(passageNumber).filter(Boolean)
    if (numbers.length < 2) continue
    const newer = passageNumber(conflict.newer_passage)
    notes.push(
      `Passages ${numbers.join(' and ')} may disagree (${conflict.kind ?? 'values differ'})` +
        (newer ? `; [${newer}] is newer.` : '.'),
    )
  }
  const stale = passages.filter((/** @type {any} */ p) => p.stale).length
  if (stale > 0) notes.push(`${stale} passage(s) may be out of date; mention the date or check with staff.`)
  if (Array.isArray(signals.degraded) && signals.degraded.length > 0) {
    notes.push(`Search ran in degraded mode (${signals.degraded.join(', ')}); ranking may be less accurate.`)
  }

  return {
    trace_id: response?.trace_id ?? null,
    nothing_relevant: signals.nothing_relevant !== false,
    reason: signals.reason ?? null,
    notes,
    passages: passages.map((/** @type {any} */ p) => ({
      id: String(p.id),
      title: p.source?.title ?? '',
      heading: p.heading ?? null,
      text: truncate(String(p.text ?? ''), maxChars),
      document_id: p.source?.document_id ?? null,
      collection: p.source?.collection ?? null,
      uri: p.source?.uri ?? null,
      page_start: p.source?.page_start ?? null,
      page_end: p.source?.page_end ?? null,
      updated: p.source?.modified_at ?? p.source?.ingested_at ?? null,
      age_days: p.age_days ?? null,
      stale: p.stale === true,
      score: p.score ?? null,
    })),
  }
}

/** @param {ReturnType<typeof toSearchResult>} result */
export function renderSearch(result) {
  if (result.nothing_relevant || result.passages.length === 0) {
    return (
      'Nothing relevant was found in the company documents' +
      (result.reason ? ` (${result.reason})` : '') +
      '. Do not answer this from the documents: say you do not have that information, or hand off to a person.'
    )
  }
  const blocks = result.passages.map((p, i) => {
    const where = [p.title, p.heading].filter(Boolean).join(' > ') || 'Untitled'
    const pages = p.page_start ? (p.page_end && p.page_end !== p.page_start ? `, pp. ${p.page_start}-${p.page_end}` : `, p. ${p.page_start}`) : ''
    const age = p.age_days !== null && p.age_days !== undefined ? `, updated ${p.age_days} day(s) ago` : ''
    return `[${i + 1}] ${where}${pages}${age}${p.stale ? ' (may be out of date)' : ''}\n${p.text}`
  })
  const notes = result.notes.length > 0 ? `\n\nNotes:\n${result.notes.map((n) => `- ${n}`).join('\n')}` : ''
  return `Passages from the company documents. Answer only from them.\n\n${blocks.join('\n\n')}${notes}`
}

/** @param {ReturnType<typeof toSearchResult>} result */
function searchMeta(result) {
  return {
    trace_id: result.trace_id,
    nothing_relevant: result.nothing_relevant,
    sources: result.passages.map((p) => ({ id: p.id, title: p.title, uri: p.uri, document_id: p.document_id })),
  }
}

/** @param {any} response */
export function toListingsResult(response) {
  return {
    trace_id: response?.trace_id ?? null,
    total: Number(response?.total ?? 0),
    as_of: response?.as_of ?? null,
    applied_filters: response?.applied_filters ?? {},
    items: Array.isArray(response?.items) ? response.items : [],
  }
}

const LISTING_SKIP = new Set(['id', 'collection_id', 'as_of', 'attributes'])
const LISTING_ORDER = ['listing_code', 'title', 'project', 'property_type', 'status', 'price_vnd', 'price_per_m2_vnd', 'area_m2', 'bedrooms', 'bathrooms']

/** @param {ReturnType<typeof toListingsResult>} result */
export function renderListings(result) {
  const applied = Object.entries(result.applied_filters)
    .map(([k, v]) => `${k}=${Array.isArray(v) ? v.join('|') : typeof v === 'object' ? JSON.stringify(v) : v}`)
    .join(', ')
  const filters = applied ? ` Filters applied: ${applied}.` : ''
  if (result.items.length === 0) return `No listing matches.${filters}`

  const lines = result.items.map((/** @type {Record<string, any>} */ item) => {
    const keys = [...LISTING_ORDER.filter((k) => k in item), ...Object.keys(item).filter((k) => !LISTING_SKIP.has(k) && !LISTING_ORDER.includes(k))]
    const parts = keys.map((k) => formatListingField(k, item[k])).filter(Boolean)
    for (const [k, v] of Object.entries(item.attributes ?? {})) if (v !== null && v !== '') parts.push(`${k}: ${v}`)
    return `- [${item.id}] ${parts.join(' · ')}`
  })
  const asOf = result.as_of ? ` Data as of ${result.as_of}; quote this date with prices or availability.` : ''
  return `${result.items.length} of ${result.total} matching listing(s).${asOf}${filters}\n${lines.join('\n')}`
}

/** @param {string} key @param {unknown} value */
function formatListingField(key, value) {
  if (value === null || value === undefined || value === '') return ''
  if (typeof value === 'number') {
    if (key === 'price_vnd') return `price: ${formatVnd(value)}`
    if (key === 'price_per_m2_vnd') return `price/m²: ${formatVnd(value)}`
    if (key === 'area_m2') return `area: ${value} m²`
  }
  return `${key}: ${Array.isArray(value) ? value.join(', ') : value}`
}

/** @param {number} vnd */
export function formatVnd(vnd) {
  const trim = (/** @type {number} */ n) => String(Number(n.toFixed(2)))
  if (Math.abs(vnd) >= 1e9) return `${trim(vnd / 1e9)} billion VND`
  if (Math.abs(vnd) >= 1e6) return `${trim(vnd / 1e6)} million VND`
  return `${Math.round(vnd).toLocaleString('en-US')} VND`
}

/** @param {ReturnType<typeof toListingsResult>} result */
function listingsMeta(result) {
  return { trace_id: result.trace_id, total: result.total, as_of: result.as_of, ids: result.items.map((i) => String(i.id)) }
}

// ---------------------------------------------------------------------------
// Helpers

/** @param {string} value */
function text(value) {
  return [{ type: 'text', text: value }]
}

/** @param {string} value @param {number} max */
function truncate(value, max) {
  return value.length <= max ? value : `${value.slice(0, max).trimEnd()} ...`
}

/** @param {number} n @param {number} lo @param {number} hi */
function clamp(n, lo, hi) {
  return Math.min(Math.max(Math.trunc(n), lo), hi)
}

/** @param {Response} response */
async function errorDetail(response) {
  let body
  try {
    body = await response.json()
  } catch {
    return response.statusText || 'no detail'
  }
  const detail = body?.detail ?? body
  if (Array.isArray(detail?.rejected)) {
    return detail.rejected.map((/** @type {any} */ r) => `${r.filter}: ${r.reason}`).join('; ')
  }
  if (Array.isArray(detail)) {
    return detail.map((/** @type {any} */ d) => `${(d.loc ?? []).filter((/** @type {unknown} */ x) => x !== 'body').join('.')}: ${d.msg}`).join('; ')
  }
  if (typeof detail === 'string') return detail
  return truncate(JSON.stringify(detail), 500)
}

/** @param {string} message */
function configError(message) {
  return new Error(`fastrag-tools: ${message}`)
}

/** @param {unknown} value @param {string} name */
function optionalString(value, name) {
  if (value === undefined || value === null) return undefined
  if (typeof value !== 'string' || !value.trim()) throw configError(`${name} must be a non-empty string`)
  return value.trim()
}

/** @param {unknown} value @param {string} name @returns {string[] | undefined} */
function optionalStringList(value, name) {
  if (value === undefined || value === null) return undefined
  if (!Array.isArray(value) || value.some((v) => typeof v !== 'string' || !v.trim())) {
    throw configError(`${name} must be a list of non-empty strings`)
  }
  return value.map((v) => v.trim())
}

/** @param {unknown} value @param {string} name @param {number} lo @param {number} hi */
function optionalInt(value, name, lo, hi) {
  if (value === undefined || value === null) return undefined
  if (typeof value !== 'number' || !Number.isInteger(value) || value < lo || value > hi) {
    throw configError(`${name} must be an integer from ${lo} to ${hi}`)
  }
  return value
}
