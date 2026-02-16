import type { INodeProperties } from 'n8n-workflow';

import { SYSTEM_MESSAGE } from './prompt';

export const commonOptions: INodeProperties[] = [
	{
		displayName: 'System Message',
		name: 'systemMessage',
		type: 'string',
		default: SYSTEM_MESSAGE,
		description: 'The message that will be sent to the agent before the conversation starts',
		builderHint: {
			message:
				"Must include: agent's purpose, exact names of connected tools, and response instructions",
		},
		typeOptions: {
			rows: 6,
		},
	},
	{
		displayName: 'Max Iterations',
		name: 'maxIterations',
		type: 'number',
		default: 10,
		description: 'The maximum number of iterations the agent will run before stopping',
	},
	{
		displayName: 'Return Intermediate Steps',
		name: 'returnIntermediateSteps',
		type: 'boolean',
		default: false,
		description: 'Whether or not the output should include intermediate steps the agent took',
	},
	{
		displayName: 'Automatically Passthrough Binary Images',
		name: 'passthroughBinaryImages',
		type: 'boolean',
		default: true,
		description:
			'Whether or not binary images should be automatically passed through to the agent as image type messages',
	},
	{
		displayName: 'Structured Output Method',
		name: 'structuredOutputMethod',
		type: 'options',
		default: 'toolCalling',
		description:
			"How to enforce structured output when an output parser is connected. Tool Calling uses a dedicated tool that the model must call. JSON Schema constrains the model's text output directly — useful for free models that support structured output but not tool calling.",
		displayOptions: {
			show: {
				'/hasOutputParser': [true],
			},
		},
		options: [
			{
				name: 'Tool Calling (Default)',
				value: 'toolCalling',
				description: 'Uses format_final_json_response tool with tool_choice: required',
			},
			{
				name: 'JSON Schema (Free Models)',
				value: 'jsonSchema',
				description:
					'Uses response_format: json_schema — works with models that support structured output but not tool calling',
			},
		],
	},
];
