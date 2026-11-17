/**
 * Cordis plugin for DeepSeek Harness profiles. It registers the fastRag tools
 * named in its config. `@deepseek-ai/dsh-tools` must be resolvable from where
 * this file is installed (it is a peer dependency).
 *
 * A host that already bundles its own tools plugin should import
 * `createFastragTools` from the package root instead and pass its `defineTool`.
 *
 * Like every dsh SDK plugin, it never writes to stdout.
 */
import { defineTool } from '@deepseek-ai/dsh-tools'
import { createFastragTools } from './index.mjs'

export const name = 'fastrag-tools'
export const inject = ['tools']

/**
 * @param {{ tools: { register(tool: unknown): unknown } }} ctx
 * @param {unknown} config
 */
export function apply(ctx, config) {
  for (const tool of createFastragTools(defineTool, config)) ctx.tools.register(tool)
}
