// Run: node --test test/*.test.mjs
// Set DSH_TOOLS_MODULE to the path of @deepseek-ai/dsh-tools (lib/index.js) to
// also check the definitions against the real defineTool.
import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { after, before, describe, it } from 'node:test'
import { pathToFileURL } from 'node:url'
import { createFastragTools, FASTRAG_TOOLS, formatVnd, parseFastragConfig } from '../index.mjs'

const ENV = { FASTRAG_API_KEY: 'fr_test' }

/** Identity stand-in for defineTool: returns the options, so tests can call execute/render directly. */
const fakeDefineTool = (options) => options

/** @type {{ path: string, auth: string | undefined, body: any }[]} */
const calls = []
/** @type {(path: string, body: any) => { status: number, body: unknown, delayMs?: number }} */
let respond = () => ({ status: 500, body: {} })
let baseUrl = ''
let server

before(async () => {
  server = createServer((req, res) => {
    let raw = ''
    req.on('data', (chunk) => (raw += chunk))
    req.on('end', () => {
      const body = raw ? JSON.parse(raw) : undefined
      calls.push({ path: req.url ?? '', auth: req.headers.authorization, body })
      const out = respond(req.url ?? '', body)
      setTimeout(() => {
        res.writeHead(out.status, { 'content-type': 'application/json' })
        res.end(JSON.stringify(out.body))
      }, out.delayMs ?? 0)
    })
  })
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
  baseUrl = `http://127.0.0.1:${server.address().port}`
})

after(() => server.close())

function tools(config = {}, defineTool = fakeDefineTool) {
  const list = createFastragTools(defineTool, { baseUrl, ...config }, { env: ENV })
  return Object.fromEntries(list.map((t) => [t.name, t]))
}

function lastCall() {
  return calls.at(-1)
}

const SEARCH_RESPONSE = {
  schema_version: '1.0',
  trace_id: 'tr-1',
  profile: 'customer',
  groups: ['public'],
  passages: [
    {
      id: 'chunk:11',
      text: 'Phí quản lý là 15.000 đồng/m2/tháng.',
      context: null,
      heading: 'Phí dịch vụ',
      score: 0.91,
      scores: {},
      source: { document_id: 'd1', collection: 'docs', title: 'FAQ cư dân', uri: 'https://example.test/faq', external_id: 'faq.pdf', version: 1, page_start: 3, page_end: 3, modified_at: '2026-10-01T00:00:00+00:00', ingested_at: null },
      flags: [],
      age_days: 47,
      stale: false,
    },
    {
      id: 'chunk:12',
      text: 'Phí quản lý là 12.000 đồng/m2/tháng.',
      context: null,
      heading: null,
      score: 0.7,
      scores: {},
      source: { document_id: 'd2', collection: 'docs', title: 'Bảng phí 2024', uri: null, external_id: 'fees.docx', version: 1, page_start: null, page_end: null, modified_at: null, ingested_at: '2024-01-01T00:00:00+00:00' },
      flags: [],
      age_days: 1051,
      stale: true,
    },
  ],
  signals: {
    nothing_relevant: false,
    reason: null,
    conflicts: [{ kind: 'money', passages: [11, 12], newer_passage: 11 }],
    degraded: [],
  },
  timings_ms: {},
}

describe('config', () => {
  it('fills defaults and reads the key from the environment', () => {
    const config = parseFastragConfig({}, ENV)
    assert.equal(config.baseUrl, 'http://127.0.0.1:8000')
    assert.equal(config.apiKey, 'fr_test')
    assert.equal(config.profile, 'customer')
    assert.deepEqual([...config.tools], Object.values(FASTRAG_TOOLS))
    assert.equal(config.topK, 5)
  })

  it('reads the key from a custom variable and trims the base URL', () => {
    const config = parseFastragConfig({ baseUrl: 'http://rag:8000/', apiKeyEnv: 'RAG_KEY' }, { RAG_KEY: ' fr_x ' })
    assert.equal(config.baseUrl, 'http://rag:8000')
    assert.equal(config.apiKey, 'fr_x')
  })

  it('fails loudly on a missing key or bad values', () => {
    assert.throws(() => parseFastragConfig({}, {}), /FASTRAG_API_KEY/)
    assert.throws(() => parseFastragConfig({ tools: ['search_catalog'] }, ENV), /unknown tool/)
    assert.throws(() => parseFastragConfig({ tools: [] }, ENV), /at least one/)
    assert.throws(() => parseFastragConfig({ profile: 'admin' }, ENV), /profile/)
    assert.throws(() => parseFastragConfig({ baseUrl: 'file:///etc' }, ENV), /http/)
    assert.throws(() => parseFastragConfig({ topK: 0 }, ENV), /topK/)
    assert.throws(() => parseFastragConfig({ descriptions: { nope: 'x' } }, ENV), /unknown tool/)
  })

  it('registers only the selected tools, with description overrides', () => {
    const t = tools({ tools: ['search_documents'], descriptions: { search_documents: 'Tra cứu tài liệu.' } })
    assert.deepEqual(Object.keys(t), ['search_documents'])
    assert.equal(t.search_documents.description, 'Tra cứu tài liệu.')
  })
})

describe('search_documents', () => {
  it('sends the configured profile, top_k and collections with the API key', async () => {
    respond = () => ({ status: 200, body: SEARCH_RESPONSE })
    const t = tools({ profile: 'handoff_copilot', topK: 3, collections: { documents: ['docs'] } })
    await t.search_documents.execute({ query: '  Phí quản lý bao nhiêu? ' }, {})
    assert.equal(lastCall().path, '/v1/search')
    assert.equal(lastCall().auth, 'Bearer fr_test')
    assert.deepEqual(lastCall().body, { query: 'Phí quản lý bao nhiêu?', profile: 'handoff_copilot', top_k: 3, collections: ['docs'] })
  })

  it('renders numbered passages with sources, conflicts and staleness', async () => {
    respond = () => ({ status: 200, body: SEARCH_RESPONSE })
    const t = tools()
    const value = await t.search_documents.execute({ query: 'phí quản lý' }, {})
    const [block] = t.search_documents.output.render({}, value)
    assert.match(block.text, /\[1\] FAQ cư dân > Phí dịch vụ, p\. 3, updated 47 day\(s\) ago\n/)
    assert.match(block.text, /\[2\] Bảng phí 2024, updated 1051 day\(s\) ago \(may be out of date\)/)
    assert.match(block.text, /Passages 1 and 2 may disagree \(money\); \[1\] is newer\./)
    assert.match(block.text, /1 passage\(s\) may be out of date/)
    assert.deepEqual(t.search_documents.output.presentationMeta({}, value), {
      trace_id: 'tr-1',
      nothing_relevant: false,
      sources: [
        { id: 'chunk:11', title: 'FAQ cư dân', uri: 'https://example.test/faq', document_id: 'd1' },
        { id: 'chunk:12', title: 'Bảng phí 2024', uri: null, document_id: 'd2' },
      ],
    })
    assert.doesNotThrow(() => JSON.stringify(value))
  })

  it('tells the model not to answer when nothing is relevant', async () => {
    respond = () => ({
      status: 200,
      body: { ...SEARCH_RESPONSE, signals: { nothing_relevant: true, reason: 'top_score_below_threshold' } },
    })
    const t = tools()
    const value = await t.search_documents.execute({ query: 'giá vàng hôm nay' }, {})
    const [block] = t.search_documents.output.render({}, value)
    assert.match(block.text, /^Nothing relevant was found .*\(top_score_below_threshold\).*Do not answer/)
    assert.doesNotMatch(block.text, /15\.000/)
  })

  it('cuts long passages', async () => {
    const long = { ...SEARCH_RESPONSE.passages[0], text: 'a'.repeat(5000) }
    respond = () => ({ status: 200, body: { ...SEARCH_RESPONSE, passages: [long], signals: { nothing_relevant: false } } })
    const value = await tools({ maxPassageChars: 300 }).search_documents.execute({ query: 'x' }, {})
    assert.equal(value.passages[0].text.length, 304)
  })
})

describe('listings', () => {
  const LISTINGS = {
    trace_id: 'tr-2',
    profile: 'customer',
    groups: ['public'],
    total: 7,
    as_of: '2026-11-17T08:00:00+00:00',
    applied_filters: { district: ['q7'], price_max: 3500000000 },
    items: [
      { id: 'l-1', collection_id: 'c', as_of: '2026-11-17T08:00:00+00:00', listing_code: 'A-12', project: 'Sunrise', price_vnd: 3200000000, area_m2: 72.5, bedrooms: 2, district_code: 'q7', attributes: { view: 'river' } },
    ],
  }

  it('sends only filled filters and clamps the limit', async () => {
    respond = () => ({ status: 200, body: LISTINGS })
    const t = tools({ collections: { listings: ['listings'] } })
    await t.query_listings.execute({ district: ' Q7 ', price_max: '3,5 tỷ', project: '  ', bedrooms_min: 2, sort: 'price_asc', limit: 99 }, {})
    assert.equal(lastCall().path, '/v1/listings/query')
    assert.deepEqual(lastCall().body, {
      filters: { district: 'Q7', price_max: '3,5 tỷ', bedrooms_min: 2 },
      sort: 'price_asc',
      limit: 20,
      profile: 'customer',
      collections: ['listings'],
    })
  })

  it('renders listings with readable prices and the as-of date', async () => {
    respond = () => ({ status: 200, body: LISTINGS })
    const t = tools()
    const value = await t.query_listings.execute({}, {})
    const [block] = t.query_listings.output.render({}, value)
    assert.match(block.text, /^1 of 7 matching listing\(s\)\. Data as of 2026-11-17T08:00:00\+00:00/)
    assert.match(block.text, /Filters applied: district=q7, price_max=3500000000\./)
    assert.match(block.text, /- \[l-1\] listing_code: A-12 · project: Sunrise · price: 3\.2 billion VND · area: 72\.5 m² · bedrooms: 2 · district_code: q7 · view: river/)
    assert.deepEqual(t.query_listings.output.presentationMeta({}, value), { trace_id: 'tr-2', total: 7, as_of: LISTINGS.as_of, ids: ['l-1'] })
  })

  it('passes rejected filters back to the model', async () => {
    respond = () => ({
      status: 422,
      body: { detail: { error: 'invalid_filters', rejected: [{ filter: 'district', reason: 'unknown district "Q99"' }] } },
    })
    await assert.rejects(tools().query_listings.execute({ district: 'Q99' }, {}), /could not use these arguments: district: unknown district "Q99"/)
  })

  it('looks up unique ids', async () => {
    respond = () => ({ status: 200, body: LISTINGS })
    await tools().get_listings.execute({ ids: ['l-1', ' l-1', 'l-2'] }, {})
    assert.equal(lastCall().path, '/v1/listings/lookup')
    assert.deepEqual(lastCall().body, { ids: ['l-1', 'l-2'], profile: 'customer', collections: null })
    await assert.rejects(tools().get_listings.execute({ ids: [' '] }, {}), /at least one listing id/)
  })

  it('formats VND amounts', () => {
    assert.equal(formatVnd(3_500_000_000), '3.5 billion VND')
    assert.equal(formatVnd(850_000_000), '850 million VND')
    assert.equal(formatVnd(45_000), '45,000 VND')
  })
})

describe('errors', () => {
  it('explains a refused key', async () => {
    respond = () => ({ status: 401, body: { detail: 'invalid token' } })
    await assert.rejects(tools().search_documents.execute({ query: 'x' }, {}), /refused the API key \(HTTP 401\): invalid token/)
  })

  it('explains validation errors from the API', async () => {
    respond = () => ({ status: 422, body: { detail: [{ loc: ['body', 'query'], msg: 'String should have at most 2000 characters' }] } })
    await assert.rejects(tools().search_documents.execute({ query: 'x' }, {}), /query: String should have at most 2000 characters/)
  })

  it('times out', async () => {
    respond = () => ({ status: 200, body: SEARCH_RESPONSE, delayMs: 1500 })
    await assert.rejects(tools({ timeoutMs: 500 }).search_documents.execute({ query: 'x' }, {}), /did not answer within 500 ms/)
  })

  it('reports an unreachable server', async () => {
    const t = createFastragTools(fakeDefineTool, { baseUrl: 'http://127.0.0.1:1' }, { env: ENV })
    await assert.rejects(t[0].execute({ query: 'x' }, {}), /unreachable at http:\/\/127\.0\.0\.1:1/)
  })

  it('passes the caller cancellation through', async () => {
    respond = () => ({ status: 200, body: SEARCH_RESPONSE, delayMs: 1000 })
    const controller = new AbortController()
    const run = tools().search_documents.execute({ query: 'x' }, { signal: controller.signal })
    controller.abort()
    await assert.rejects(run, (error) => error.name === 'AbortError')
  })
})

describe('with the real defineTool', { skip: !process.env.DSH_TOOLS_MODULE && 'set DSH_TOOLS_MODULE' }, () => {
  it('compiles every tool and validates arguments', async () => {
    const { defineTool } = await import(pathToFileURL(process.env.DSH_TOOLS_MODULE).href)
    respond = () => ({ status: 200, body: SEARCH_RESPONSE })
    const t = tools({}, defineTool)
    assert.deepEqual(Object.keys(t).sort(), Object.values(FASTRAG_TOOLS).sort())
    assert.equal(t.query_listings.parameters.properties.sort.enum.length, 6)
    assert.deepEqual(t.search_documents.parameters.required, ['query'])
    assert.equal(t.search_documents.isConcurrencySafe({ query: 'x' }), true)

    const value = await t.search_documents.execute({ query: 'phí quản lý' }, { signal: new AbortController().signal })
    assert.match(t.search_documents.output.render({ query: 'x' }, value)[0].text, /\[1\] FAQ cư dân/)
    await assert.rejects(t.search_documents.execute({}, {}), /invalid arguments/)
    await assert.rejects(t.query_listings.execute({ sort: 'cheapest' }, {}), /invalid arguments/)
  })
})
