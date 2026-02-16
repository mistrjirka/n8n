import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { ChatPromptTemplate } from '@langchain/core/prompts';
import { RunnableSequence } from '@langchain/core/runnables';
import { type AgentRunnableSequence, createToolCallingAgent } from '@langchain/classic/agents';
import type { BaseChatMemory } from '@langchain/classic/memory';
import type { DynamicStructuredTool, Tool } from '@langchain/classic/tools';
import { zodToJsonSchema } from 'zod-to-json-schema';

import type { N8nOutputParser } from '@utils/output_parsers/N8nOutputParser';

import {
	fixEmptyContentMessage,
	getAgentStepsParser,
	getResponseFormatConfig,
	makeSchemaStrict,
	type ResponseFormatResult,
} from '../../common';
import type { AgentOptions } from '../types';

/**
 * Creates an agent sequence with the given configuration.
 *
 * We pre-bind tools ourselves with { strict: true } because createToolCallingAgent()
 * calls bindTools() without strict, and if the model is wrapped with withConfig()
 * (for response_format), it skips bindTools() entirely.
 *
 * Our approach:
 * 1. Call model.bindTools(tools, { strict: true }) on the raw model
 * 2. Apply withConfig({ response_format }) on the result (for jsonSchema mode)
 * 3. Pass the fully-configured model to createToolCallingAgent — since it's now
 *    a RunnableBinding (not a BaseChatModel), createToolCallingAgent uses it as-is
 *    without calling bindTools() again
 */
export function createAgentSequence(
	model: BaseChatModel,
	tools: Array<DynamicStructuredTool | Tool>,
	prompt: ChatPromptTemplate,
	options: AgentOptions,
	outputParser?: N8nOutputParser,
	memory?: BaseChatMemory,
	fallbackModel?: BaseChatModel | null,
) {
	const allTools = getAllTools(model, tools);

	const { bound: primaryBound, enumMapping } = bindModelWithTools(
		model,
		allTools,
		options,
		outputParser,
	);

	let fallbackBound: any;
	if (fallbackModel) {
		const fallbackTools = getAllTools(fallbackModel, tools);
		fallbackBound = bindModelWithTools(fallbackModel, fallbackTools, options, outputParser).bound;
	}

	// createToolCallingAgent checks _isBaseChatModel(llm). Since primaryBound
	// is a RunnableBinding (from bindTools + optional withConfig), the check
	// returns false, and it uses the model as-is — no second bindTools call.
	const agent = createToolCallingAgent({
		llm: primaryBound,
		tools: allTools,
		prompt,
		streamRunnable: false,
	});

	let fallbackAgent: AgentRunnableSequence | undefined;
	if (fallbackBound) {
		const fallbackTools = getAllTools(fallbackModel!, tools);
		fallbackAgent = createToolCallingAgent({
			llm: fallbackBound,
			tools: fallbackTools,
			prompt,
			streamRunnable: false,
		});
	}

	const runnableAgent = RunnableSequence.from([
		fallbackAgent ? agent.withFallbacks([fallbackAgent]) : agent,
		getAgentStepsParser(outputParser, memory, options.structuredOutputMethod, enumMapping),
		fixEmptyContentMessage,
	]) as AgentRunnableSequence;

	runnableAgent.singleAction = true;
	runnableAgent.streamRunnable = false;

	return runnableAgent;
}

/**
 * Binds tools to the model with strict mode, then optionally layers on
 * response_format for JSON Schema mode.
 *
 * Order matters: bindTools first (on raw BaseChatModel), withConfig second.
 * bindTools returns a RunnableBinding. withConfig on that returns another
 * RunnableBinding wrapping the first. Both configs merge at invoke time.
 */
function bindModelWithTools(
	model: BaseChatModel,
	allTools: Array<DynamicStructuredTool | Tool>,
	options: AgentOptions,
	outputParser?: N8nOutputParser,
): { bound: any; enumMapping?: Map<string, string> } {
	const openAiTools = convertToolsToStrictOpenAIFormat(allTools);
	const configArgs: Record<string, unknown> = { tools: openAiTools };

	if (outputParser && options.structuredOutputMethod !== 'jsonSchema') {
		configArgs.tool_choice = {
			type: 'function',
			function: { name: 'format_final_json_response' },
		};
	}

	let bound = model.withConfig(configArgs as any);
	let enumMapping: Map<string, string> | undefined;

	// For JSON Schema mode, layer on response_format via withConfig
	if (outputParser && options.structuredOutputMethod === 'jsonSchema') {
		const result: ResponseFormatResult = getResponseFormatConfig(outputParser);
		bound = bound.withConfig({ response_format: result.config } as any);
		enumMapping = result.enumMapping;
	}

	// Final payload-level sanitization for strict tool schemas.
	// Some provider/model combinations still emit tool parameter schemas with
	// "$schema" and without complete "required" lists, which causes 400s.
	sanitizeBoundToolsForStrictMode(bound);

	return { bound, enumMapping };
}

function convertToolsToStrictOpenAIFormat(
	tools: Array<DynamicStructuredTool | Tool>,
): Array<{ type: 'function'; function: Record<string, unknown> }> {
	return tools.map((tool: any) => {
		let parameters: Record<string, unknown>;
		if (tool.schema) {
			parameters = zodToJsonSchema(tool.schema) as Record<string, unknown>;
		} else {
			parameters = { type: 'object', properties: {} };
		}

		parameters = makeSchemaStrict(parameters);

		return {
			type: 'function' as const,
			function: {
				name: tool.name,
				description: tool.description || '',
				parameters,
				strict: true,
			},
		};
	});
}

function sanitizeBoundToolsForStrictMode(boundModel: any): void {
	const visited = new Set<any>();
	let sanitizedCount = 0;

	const walk = (node: any) => {
		if (!node || typeof node !== 'object' || visited.has(node)) return;
		visited.add(node);

		const toolArrays = [node?.config?.tools, node?.kwargs?.tools].filter(Array.isArray);
		for (const tools of toolArrays) {
			for (const tool of tools) {
				const fn = tool?.function;
				if (!fn || typeof fn !== 'object') continue;

				fn.strict = true;
				if (fn.parameters && typeof fn.parameters === 'object') {
					sanitizeJsonSchemaStrict(fn.parameters);
					sanitizedCount++;
				}
			}
		}

		walk(node.bound);
	};

	walk(boundModel);
	console.log('[SANITIZE] strict tool schemas normalized:', sanitizedCount);
}

function sanitizeJsonSchemaStrict(schema: any): void {
	if (!schema || typeof schema !== 'object') return;

	delete schema.$schema;

	if (schema.type === 'object' || schema.properties) {
		schema.additionalProperties = false;
		if (schema.properties && typeof schema.properties === 'object') {
			const keys = Object.keys(schema.properties);
			schema.required = keys;
			for (const key of keys) {
				sanitizeJsonSchemaStrict(schema.properties[key]);
			}
		}
	}

	if (schema.items) sanitizeJsonSchemaStrict(schema.items);
	if (Array.isArray(schema.anyOf)) schema.anyOf.forEach(sanitizeJsonSchemaStrict);
	if (Array.isArray(schema.oneOf)) schema.oneOf.forEach(sanitizeJsonSchemaStrict);
	if (Array.isArray(schema.allOf)) schema.allOf.forEach(sanitizeJsonSchemaStrict);
}

/**
 * Uses provided tools and tried to get tools from model metadata
 * Some chat model nodes can define built-in tools in their metadata
 */
function getAllTools(model: BaseChatModel, tools: Array<DynamicStructuredTool | Tool>) {
	const modelTools = (model.metadata?.tools as Tool[]) ?? [];
	const allTools = [...tools, ...modelTools];
	return allTools;
}
