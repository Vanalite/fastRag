# @fastrag/dsh-tools

fastRag retrieval as native tools for [DeepSeek Harness](https://github.com/deepseek-ai) (`dsh`) and any other host built on `@deepseek-ai/dsh-tools`. The agent gets three tools:

| Tool | Calls | What the model sees |
|---|---|---|
| `search_documents` | `POST /v1/search` | Numbered passages with title, heading, page and age. It also gets notes on conflicting or stale passages, or an explicit "nothing relevant, do not answer". |
| `query_listings` | `POST /v1/listings/query` | One line per listing with readable prices (`3.2 billion VND`), the filters fastRag applied and the `as_of` date. |
| `get_listings` | `POST /v1/listings/lookup` | The same lines, refreshed by id. |

These are the same tool names as fastRag's MCP server and `GET /v1/tools`.

- **No dependencies.** The host passes in its own `defineTool`, so this package bundles cleanly into an existing tools plugin.
- **No secrets in config.** The config names the environment variable that holds the API key, and the key never goes into a profile patch.
- **Errors stay in the turn.** fastRag being down, a refused key, or a filter it cannot read (`district: unknown district "Q99"`) all become ordinary tool errors that the model can act on. The turn does not end.
- **Read-only.** Every tool is marked concurrency-safe and honours the harness's cancellation signal.

## Option A: inside an existing tools plugin (any-agent)

Use this when your host already bundles a Cordis tools plugin, as any-agent's `@any/harness-tools` does with esbuild and `--external:@deepseek-ai/*`.

```jsonc
// package.json of the tools plugin
"dependencies": { "@fastrag/dsh-tools": "link:../../../fastRag/integrations/dsh-tools" }
```

```ts
// plugin.ts: in apply(), next to your own tools
import { defineTool } from '@deepseek-ai/dsh-tools'
import { createFastragTools } from '@fastrag/dsh-tools'

if (config.fastrag) {
  for (const tool of createFastragTools(defineTool, config.fastrag)) ctx.tools.register(tool)
}
```

The host then puts a `fastrag` object (see **Config**) into the plugin config of its profile patch and passes `FASTRAG_API_KEY` into the `dsh` child's environment.

## Option B: as its own plugin in a dsh profile

Install the package where `@deepseek-ai/dsh-tools` resolves (it is a peer dependency), then add a row to your profile patch:

```yaml
- insert:
    - id: fastrag-tools
      name: /abs/path/to/node_modules/@fastrag/dsh-tools/plugin.mjs
      config:
        baseUrl: http://127.0.0.1:8000
        tools: [search_documents]
```

## Config

All fields are optional, and the config is plain JSON.

| Field | Default | Meaning |
|---|---|---|
| `baseUrl` | `http://127.0.0.1:8000` | fastRag API root. |
| `apiKeyEnv` | `FASTRAG_API_KEY` | Name of the environment variable holding a tenant API key (`fastrag bootstrap` prints them). |
| `profile` | `customer` | `customer`, `handoff_copilot` or `internal_search`. The API key's ceiling still applies. |
| `tools` | all three | Which tools to register. |
| `collections` | all | `{ "documents": ["docs"], "listings": ["listings"] }` |
| `topK` | `5` | Passages per search (1–20). |
| `timeoutMs` | `15000` | Per-call HTTP timeout. |
| `maxPassageChars` | `1500` | Passage text is cut to this length before the model sees it. |
| `descriptions` | English | Replace a tool's model-facing description, for example to match the agent's prompt language. |

Bad config throws when the plugin loads (`fastrag-tools: environment variable FASTRAG_API_KEY must hold a fastRag API key`), so a misconfigured agent fails at startup, not mid-conversation.

Each tool's presentation metadata carries fastRag's `trace_id`, plus source ids for searches and listing ids for listings. The host can show citations, or send thumbs-up/down to `POST /v1/feedback`.

## Test

```bash
node --test test/*.test.mjs
# also check against the real defineTool:
DSH_TOOLS_MODULE=/path/to/node_modules/@deepseek-ai/dsh-tools/lib/index.js node --test test/*.test.mjs
```
