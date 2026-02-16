import type { AgentRunnableSequence } from '@langchain/classic/agents';
import type { BaseChatMemory } from '@langchain/classic/memory';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type {
	EngineRequest,
	EngineResponse,
	IExecuteFunctions,
	ISupplyDataFunctions,
} from 'n8n-workflow';

import {
	buildResponseMetadata,
	createEngineRequests,
	loadMemory,
	processEventStream,
	saveToMemory,
	type RequestResponseMetadata,
} from '@utils/agent-execution';
import { getTracingConfig } from '@utils/tracing';

import { SYSTEM_MESSAGE } from '../../prompt';
import type { AgentResult } from '../types';
import type { ItemContext } from './prepareItemContext';

type RunAgentResult = AgentResult | EngineRequest<RequestResponseMetadata>;
/**
 * Runs the agent for a single item, choosing between streaming or non-streaming execution.
 * Handles both regular execution and execution after tool calls.
 *
 * @param ctx - The execution context
 * @param executor - The agent runnable sequence
 * @param itemContext - Context for the current item
 * @param model - The chat model for token counting
 * @param memory - Optional memory for conversation context
 * @param response - Optional engine response with previous tool calls
 * @returns AgentResult or engine request with tool calls
 */
export async function runAgent(
	ctx: IExecuteFunctions | ISupplyDataFunctions,
	executor: AgentRunnableSequence,
	itemContext: ItemContext,
	model: BaseChatModel,
	memory: BaseChatMemory | undefined,
	response?: EngineResponse<RequestResponseMetadata>,
): Promise<RunAgentResult> {
	const { itemIndex, input, steps, tools, options } = itemContext;

	const invokeParams = {
		// steps are passed to the ToolCallingAgent in the runnable sequence to keep track of tool calls
		steps,
		input,
		system_message: options.systemMessage ?? SYSTEM_MESSAGE,
		formatting_instructions:
			'IMPORTANT: For your response to user, you MUST use the `format_final_json_response` tool with your complete answer formatted according to the required schema. Do not attempt to format the JSON manually - always use this tool. Your response will be rejected if it is not properly formatted through this tool. Only use this tool once you are ready to provide your final answer.',
	};
	const executeOptions = { signal: ctx.getExecutionCancelSignal() };

	// Check if streaming is actually available
	const isStreamingAvailable = 'isStreaming' in ctx ? ctx.isStreaming?.() : undefined;

	const debugCallback = {
		handleLLMStart: (
			_llm: any,
			_prompts: string[],
			_runId: string,
			_parentRunId?: string,
			extraParams?: any,
		) => {
			console.log('=== DEBUG: handleLLMStart ===');
			console.log('[LLM] Serialized LLM info:', JSON.stringify(_llm, null, 2));
			console.log('[LLM] Number of prompt messages:', _prompts?.length);
			const options = (extraParams?.invocation_params || extraParams) as any;
			if (options) {
				normalizeInvocationToolSchemas(options);
				console.log('[LLM] All invocation_params keys:', Object.keys(options));
				if (options.response_format) {
					console.log('[LLM] response_format:', JSON.stringify(options.response_format, null, 2));
				} else {
					console.log('[LLM] response_format: NOT PRESENT in invocation_params');
				}
				if (options.tools) {
					console.log('[LLM] tools count:', options.tools.length);
					console.log(
						'[LLM] tool names:',
						options.tools.map((t: any) => t?.function?.name ?? t?.name ?? 'unknown'),
					);
					// Log full tool schemas for debugging
					console.log('[LLM] tools (full):');
					console.log(JSON.stringify(options.tools, null, 2));
				}
				if (options.tool_choice) {
					console.log('[LLM] tool_choice:', JSON.stringify(options.tool_choice, null, 2));
				}
				if (options.model) {
					console.log('[LLM] model:', options.model);
				}
				// Log any other kwargs that might affect the request
				const importantKeys = ['strict', 'stream', 'temperature', 'max_tokens', 'model_kwargs'];
				for (const key of importantKeys) {
					if (options[key] !== undefined) {
						console.log(`[LLM] ${key}:`, JSON.stringify(options[key]));
					}
				}
			} else {
				console.log('[LLM] extraParams is empty/undefined');
			}
			console.log('=== DEBUG: handleLLMStart END ===');
		},
		handleLLMError: (err: any) => {
			console.log('=== DEBUG: handleLLMError ===');
			console.error('[ERR] Error constructor:', err?.constructor?.name);
			console.error('[ERR] Error message:', err?.message);
			console.error('[ERR] Error name:', err?.name);
			// Full error with all properties (including non-enumerable)
			try {
				const allProps = Object.getOwnPropertyNames(err);
				console.error('[ERR] All error property names:', allProps);
				for (const prop of allProps) {
					if (prop === 'stack') continue; // skip stack trace for brevity
					try {
						const val = err[prop];
						if (typeof val === 'object' && val !== null) {
							console.error(`[ERR] err.${prop}:`, JSON.stringify(val, null, 2));
						} else {
							console.error(`[ERR] err.${prop}:`, val);
						}
					} catch {
						console.error(`[ERR] err.${prop}: [could not serialize]`);
					}
				}
			} catch {
				console.error('[ERR] Could not enumerate error properties');
			}
			// Walk the cause chain to find the original HTTP error
			let current = err;
			let depth = 0;
			while (current && depth < 10) {
				if (current.cause && current.cause !== current) {
					current = current.cause;
					depth++;
					console.error(
						`[ERR] cause chain depth ${depth}:`,
						current?.constructor?.name,
						current?.message,
					);
					// OpenAI SDK errors store the body in .error
					if (current.error) {
						try {
							console.error(
								`[ERR] cause.error (raw API body):`,
								JSON.stringify(current.error, null, 2),
							);
						} catch {
							console.error(`[ERR] cause.error:`, String(current.error));
						}
					}
					if (current.status) {
						console.error(`[ERR] cause.status:`, current.status);
					}
					if (current.response) {
						console.error(`[ERR] cause.response.status:`, current.response?.status);
						if (current.response.body) {
							console.error(`[ERR] cause.response.body:`, String(current.response.body));
						}
					}
					// Some errors have the raw body in .body
					if (current.body) {
						try {
							console.error(`[ERR] cause.body:`, JSON.stringify(current.body, null, 2));
						} catch {
							console.error(`[ERR] cause.body:`, String(current.body));
						}
					}
				} else {
					break;
				}
			}
			// Direct access for common shapes
			if (err.response) {
				console.error('[ERR] err.response.status:', err.response.status);
				console.error('[ERR] err.response.statusText:', err.response.statusText);
				if (err.response.data) {
					console.error('[ERR] err.response.data:', JSON.stringify(err.response.data, null, 2));
				}
			}
			if (err.status) {
				console.error('[ERR] err.status:', err.status);
			}
			if (err.code) {
				console.error('[ERR] err.code:', err.code);
			}
			if (err.error) {
				console.error('[ERR] err.error:', JSON.stringify(err.error, null, 2));
			}
			// OpenAI SDK specific
			if (err.headers) {
				console.error('[ERR] err.headers:', JSON.stringify(err.headers, null, 2));
			}
			console.log('=== DEBUG: handleLLMError END ===');
		},
	};

	function normalizeInvocationToolSchemas(options: any): void {
		if (!Array.isArray(options?.tools)) return;

		for (const tool of options.tools) {
			const fn = tool?.function;
			if (!fn || typeof fn !== 'object') continue;

			fn.strict = true;

			const parameters = fn.parameters;
			if (!parameters || typeof parameters !== 'object') continue;

			normalizeToolParametersSchema(parameters);
		}
	}

	function normalizeToolParametersSchema(schema: any): void {
		if (!schema || typeof schema !== 'object') return;

		delete schema.$schema;

		if (schema.type === 'object' || schema.properties) {
			schema.additionalProperties = false;
			if (schema.properties && typeof schema.properties === 'object') {
				const keys = Object.keys(schema.properties);
				schema.required = keys;
				for (const key of keys) {
					normalizeToolParametersSchema(schema.properties[key]);
				}
			}
		}

		if (schema.items) normalizeToolParametersSchema(schema.items);
		if (Array.isArray(schema.anyOf)) schema.anyOf.forEach(normalizeToolParametersSchema);
		if (Array.isArray(schema.oneOf)) schema.oneOf.forEach(normalizeToolParametersSchema);
		if (Array.isArray(schema.allOf)) schema.allOf.forEach(normalizeToolParametersSchema);
	}

	if (
		'isStreaming' in ctx &&
		options.enableStreaming &&
		isStreamingAvailable &&
		ctx.getNode().typeVersion >= 2.1
	) {
		const chatHistory = await loadMemory(memory, model, options.maxTokensFromMemory);
		const tracingConfig = getTracingConfig(ctx);
		const existingCallbacks = (tracingConfig.callbacks as any) || [];
		const callbacksArray = Array.isArray(existingCallbacks)
			? existingCallbacks
			: [existingCallbacks];

		const eventStream = executor
			.withConfig({
				...tracingConfig,
				callbacks: [debugCallback, ...callbacksArray] as any,
			})
			.streamEvents(
				{
					...invokeParams,
					chat_history: chatHistory,
				},
				{
					version: 'v2',
					...executeOptions,
				},
			);

		const result = await processEventStream(ctx, eventStream, itemIndex);

		// If result contains tool calls, build the request object like the normal flow
		if (result.toolCalls && result.toolCalls.length > 0) {
			const actions = createEngineRequests(result.toolCalls, itemIndex, tools);

			return {
				actions,
				metadata: buildResponseMetadata(response, itemIndex),
			};
		}
		// Save conversation to memory including any tool call context
		if (memory && input && result?.output) {
			const previousCount = response?.metadata?.previousRequests?.length;
			await saveToMemory(input, result.output, memory, steps, previousCount);
		}

		if (options.returnIntermediateSteps && steps.length > 0) {
			result.intermediateSteps = steps;
		}

		return result;
	} else {
		// Handle regular execution
		const chatHistory = await loadMemory(memory, model, options.maxTokensFromMemory);

		const tracingConfig = getTracingConfig(ctx);
		const existingCallbacks = (tracingConfig.callbacks as any) || [];
		const callbacksArray = Array.isArray(existingCallbacks)
			? existingCallbacks
			: [existingCallbacks];

		const modelResponse = await executor
			.withConfig({
				...tracingConfig,
				callbacks: [debugCallback, ...callbacksArray] as any,
			})
			.invoke({
				...invokeParams,
				chat_history: chatHistory,
			});

		if ('returnValues' in modelResponse) {
			// Save conversation to memory including any tool call context
			if (memory && input && modelResponse.returnValues.output) {
				const previousCount = response?.metadata?.previousRequests?.length;
				await saveToMemory(input, modelResponse.returnValues.output, memory, steps, previousCount);
			}
			// Include intermediate steps if requested
			const result = { ...modelResponse.returnValues };
			if (options.returnIntermediateSteps && steps.length > 0) {
				result.intermediateSteps = steps;
			}
			return result;
		}

		// If response contains tool calls, we need to return this in the right format
		const actions = createEngineRequests(modelResponse, itemIndex, tools);

		return {
			actions,
			metadata: buildResponseMetadata(response, itemIndex),
		};
	}
}
