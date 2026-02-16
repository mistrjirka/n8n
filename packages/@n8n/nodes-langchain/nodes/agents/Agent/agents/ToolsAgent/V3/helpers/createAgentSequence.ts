import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { ChatPromptTemplate } from '@langchain/core/prompts';
import { RunnableSequence } from '@langchain/core/runnables';
import { type AgentRunnableSequence, createToolCallingAgent } from '@langchain/classic/agents';
import type { BaseChatMemory } from '@langchain/classic/memory';
import type { DynamicStructuredTool, Tool } from '@langchain/classic/tools';

import type { N8nOutputParser } from '@utils/output_parsers/N8nOutputParser';

import { fixEmptyContentMessage, getAgentStepsParser, getResponseFormatConfig } from '../../common';
import type { AgentOptions } from '../types';

/**
 * Creates an agent sequence with the given configuration.
 * The sequence includes the agent, output parser, and fallback logic.
 *
 * @param model - The primary chat model
 * @param tools - Array of tools available to the agent
 * @param prompt - The prompt template
 * @param _options - Additional options (maxIterations, returnIntermediateSteps)
 * @param outputParser - Optional output parser for structured responses
 * @param memory - Optional memory for conversation context
 * @param fallbackModel - Optional fallback model if primary fails
 * @returns AgentRunnableSequence ready for execution
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
	// Pre-bind tools with strict mode to enforce API-level schema validation on tool calls.
	// By passing the pre-bound model (a RunnableBinding, not a BaseChatModel),
	// createToolCallingAgent will skip its internal bindTools call and use ours.
	// bindTools is validated to exist in getChatModel(); strict is a provider-specific option.
	const bindOptions: Record<string, unknown> = { strict: true };
	let primaryModel: any = model;
	let secondaryModel: any = fallbackModel;

	if (outputParser) {
		if (options.structuredOutputMethod === 'jsonSchema') {
			// When using JSON Schema, we constrain the model's text output directly.
			// This pairs well with OpenRouter's structured output support.
			const responseFormat = getResponseFormatConfig(outputParser);
			primaryModel = model.withConfig({ response_format: responseFormat } as any);
			if (secondaryModel) {
				secondaryModel = secondaryModel.withConfig({ response_format: responseFormat } as any);
			}
		} else {
			// When using Tool Calling (default), force the model to always call a tool.
			// This prevents the model from "exiting" with plain text — it must call
			// format_final_json_response to finish, ensuring structured output.
			// We use the specific tool name to be explicit.
			bindOptions.tool_choice = {
				type: 'function',
				function: { name: 'format_final_json_response' },
			};
		}
	}

	console.log('--- Debug: createAgentSequence ---');
	console.log('Structured Output Method:', options.structuredOutputMethod);
	console.log('Bind Options:', JSON.stringify(bindOptions, null, 2));
	console.log('All Tools Count:', allTools.length);
	console.log('----------------------------------');

	const modelWithStrictTools = primaryModel.bindTools!(allTools, bindOptions as any);
	const agent = createToolCallingAgent({
		llm: modelWithStrictTools,
		tools: allTools,
		prompt,
		streamRunnable: false,
	});

	let fallbackAgent: AgentRunnableSequence | undefined;
	if (secondaryModel) {
		const fallbackTools = getAllTools(secondaryModel, tools);
		const fallbackWithStrictTools = secondaryModel.bindTools!(fallbackTools, bindOptions as any);
		fallbackAgent = createToolCallingAgent({
			llm: fallbackWithStrictTools,
			tools: fallbackTools,
			prompt,
			streamRunnable: false,
		});
	}
	const runnableAgent = RunnableSequence.from([
		fallbackAgent ? agent.withFallbacks([fallbackAgent]) : agent,
		getAgentStepsParser(outputParser, memory),
		fixEmptyContentMessage,
	]) as AgentRunnableSequence;

	runnableAgent.singleAction = true;
	runnableAgent.streamRunnable = false;

	return runnableAgent;
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
