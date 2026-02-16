import type { BaseChatMemory } from '@langchain/community/memory/chat_memory';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { HumanMessage } from '@langchain/core/messages';
import type { BaseMessagePromptTemplateLike } from '@langchain/core/prompts';
import { FakeLLM, FakeStreamingChatModel } from '@langchain/core/utils/testing';
import { Buffer } from 'buffer';
import { mock } from 'jest-mock-extended';
import type { AgentAction, AgentFinish } from '@langchain/classic/agents';
import type { ToolsAgentAction } from '@langchain/classic/dist/agents/tool_calling/output_parser';
import type { Tool } from '@langchain/classic/tools';
import type { IExecuteFunctions, INode } from 'n8n-workflow';
import { NodeOperationError, BINARY_ENCODING, NodeConnectionTypes } from 'n8n-workflow';
import type { ZodType } from 'zod';
import { z } from 'zod';

import type { N8nOutputParser } from '@utils/output_parsers/N8nOutputParser';

import {
	getOutputParserSchema,
	extractBinaryMessages,
	fixEmptyContentMessage,
	handleParsedStepOutput,
	getChatModel,
	getOptionalMemory,
	prepareMessages,
	preparePrompt,
	getTools,
	getAgentStepsParser,
	handleAgentFinishOutput,
	makeSchemaStrict,
	restoreEnumValues,
} from '../../agents/ToolsAgent/common';

function getFakeOutputParser(returnSchema?: ZodType): N8nOutputParser {
	const fakeOutputParser = mock<N8nOutputParser>();
	(fakeOutputParser.getSchema as jest.Mock).mockReturnValue(returnSchema);
	return fakeOutputParser;
}

function createMockOutputParser(parseReturnValue?: Record<string, unknown>): N8nOutputParser {
	const mockParser = mock<N8nOutputParser>();
	(mockParser.parse as jest.Mock).mockResolvedValue(parseReturnValue);

	return mockParser;
}

const mockHelpers = mock<IExecuteFunctions['helpers']>();
const mockContext = mock<IExecuteFunctions>({ helpers: mockHelpers });

beforeEach(() => jest.resetAllMocks());

describe('getOutputParserSchema', () => {
	it('should return a default schema if getSchema returns undefined', () => {
		const schema = getOutputParserSchema(getFakeOutputParser(undefined));
		// The default schema requires a "text" field.
		expect(() => schema.parse({})).toThrow();
		expect(schema.parse({ text: 'hello' })).toEqual({ text: 'hello' });
	});

	it('should return the custom schema if provided', () => {
		const customSchema = z.object({ custom: z.number() });

		const schema = getOutputParserSchema(getFakeOutputParser(customSchema));
		expect(() => schema.parse({ custom: 'not a number' })).toThrow();
		expect(schema.parse({ custom: 123 })).toEqual({ custom: 123 });
	});
});

describe('extractBinaryMessages', () => {
	it('should extract a binary message from the input data when no id is provided', async () => {
		const fakeItem = {
			json: {},
			binary: {
				img1: {
					mimeType: 'image/png',
					// simulate that data already includes 'base64'
					data: 'data:image/png;base64,sampledata',
				},
			},
		};
		mockContext.getInputData.mockReturnValue([fakeItem]);

		const humanMsg: HumanMessage = await extractBinaryMessages(mockContext, 0);
		// Expect the HumanMessage's content to be an array containing one binary message.
		expect(Array.isArray(humanMsg.content)).toBe(true);
		expect(humanMsg.content[0]).toEqual({
			type: 'image_url',
			image_url: { url: 'data:image/png;base64,sampledata' },
		});
	});

	it('should extract a binary message using binary stream if id is provided', async () => {
		const fakeItem = {
			json: {},
			binary: {
				img2: {
					mimeType: 'image/jpeg',
					id: '1234',
					data: 'nonsense',
				},
			},
		};

		mockHelpers.getBinaryStream.mockResolvedValue(mock());
		mockHelpers.binaryToBuffer.mockResolvedValue(Buffer.from('fakebufferdata'));
		mockContext.getInputData.mockReturnValue([fakeItem]);

		const humanMsg: HumanMessage = await extractBinaryMessages(mockContext, 0);
		// eslint-disable-next-line @typescript-eslint/unbound-method
		expect(mockHelpers.getBinaryStream).toHaveBeenCalledWith('1234');
		// eslint-disable-next-line @typescript-eslint/unbound-method
		expect(mockHelpers.binaryToBuffer).toHaveBeenCalled();
		const expectedUrl = `data:image/jpeg;base64,${Buffer.from('fakebufferdata').toString(
			BINARY_ENCODING,
		)}`;
		expect(humanMsg.content[0]).toEqual({
			type: 'image_url',
			image_url: { url: expectedUrl },
		});
	});

	it('should extract markdown and CSV text files', async () => {
		const mdContent = '# Test Markdown\n\nThis is a test.';
		const csvContent = 'name,age\nJohn,30';
		const fakeItem = {
			json: {},
			binary: {
				markdown: {
					mimeType: 'text/markdown',
					fileName: 'test.md',
					data: `data:text/markdown;base64,${Buffer.from(mdContent).toString('base64')}`,
				},
				csv: {
					mimeType: 'text/csv',
					fileName: 'data.csv',
					data: `data:text/csv;base64,${Buffer.from(csvContent).toString('base64')}`,
				},
			},
		};
		mockContext.getInputData.mockReturnValue([fakeItem]);

		const humanMsg: HumanMessage = await extractBinaryMessages(mockContext, 0);

		expect(Array.isArray(humanMsg.content)).toBe(true);
		expect(humanMsg.content).toHaveLength(2);
		expect(humanMsg.content).toEqual(
			expect.arrayContaining([
				{ type: 'text', text: `File: test.md\nContent:\n${mdContent}` },
				{ type: 'text', text: `File: data.csv\nContent:\n${csvContent}` },
			]),
		);
	});

	it('should extract both images and text files together', async () => {
		const textContent = 'Some text content';
		const fakeItem = {
			json: {},
			binary: {
				image: {
					mimeType: 'image/png',
					fileName: 'test.png',
					data: 'imageData123',
				},
				text: {
					mimeType: 'text/plain',
					fileName: 'test.txt',
					data: `data:text/plain;base64,${Buffer.from(textContent).toString('base64')}`,
				},
			},
		};
		mockContext.getInputData.mockReturnValue([fakeItem]);

		const humanMsg: HumanMessage = await extractBinaryMessages(mockContext, 0);

		expect(Array.isArray(humanMsg.content)).toBe(true);
		expect(humanMsg.content).toHaveLength(2);
		expect(humanMsg.content).toEqual(
			expect.arrayContaining([
				{
					type: 'image_url',
					image_url: { url: 'data:image/png;base64,imageData123' },
				},
				{ type: 'text', text: `File: test.txt\nContent:\n${textContent}` },
			]),
		);
	});

	it('should decode base64-encoded text files without prefix', async () => {
		const textContent = 'Hello world!';
		const fakeItem = {
			json: {},
			binary: {
				text: {
					mimeType: 'text/plain',
					fileName: 'test.txt',
					// Default n8n binary format: base64 without data URL prefix
					data: Buffer.from(textContent).toString('base64'),
				},
			},
		};
		mockContext.getInputData.mockReturnValue([fakeItem]);

		const humanMsg: HumanMessage = await extractBinaryMessages(mockContext, 0);

		expect(Array.isArray(humanMsg.content)).toBe(true);
		expect(humanMsg.content).toHaveLength(1);
		expect(humanMsg.content[0]).toEqual({
			type: 'text',
			text: `File: test.txt\nContent:\n${textContent}`,
		});
	});
});

describe('fixEmptyContentMessage', () => {
	it('should replace empty string inputs with empty objects', () => {
		// Cast to any to bypass type issues with AgentFinish/AgentAction.
		const fakeSteps: ToolsAgentAction[] = [
			{
				messageLog: [
					{
						content: [{ input: '' }, { input: { already: 'object' } }],
					},
				],
			},
		] as unknown as ToolsAgentAction[];
		const fixed = fixEmptyContentMessage(fakeSteps) as ToolsAgentAction[];
		const messageContent = fixed?.[0]?.messageLog?.[0].content;

		// Type assertion needed since we're extending MessageContentComplex
		expect((messageContent?.[0] as unknown as { input: unknown })?.input).toEqual({});
		expect((messageContent?.[1] as unknown as { input: unknown })?.input).toEqual({
			already: 'object',
		});
	});
});

describe('handleParsedStepOutput', () => {
	it('should stringify the output if memory is provided', () => {
		const output = { key: 'value' };
		const fakeMemory = mock<BaseChatMemory>();
		const result = handleParsedStepOutput(output, fakeMemory);
		expect(result.returnValues).toEqual({ output: JSON.stringify(output) });
		expect(result.log).toEqual('Final response formatted');
	});

	it('should not stringify the output if memory is not provided', () => {
		const output = { key: 'value' };
		const result = handleParsedStepOutput(output);
		expect(result.returnValues).toEqual(output);
	});
});

describe('getChatModel', () => {
	it('should return the model if it is a valid chat model', async () => {
		// Cast fakeChatModel as any
		const fakeChatModel = mock<BaseChatModel>();
		fakeChatModel.bindTools = jest.fn();
		fakeChatModel.lc_namespace = ['chat_models'];
		mockContext.getInputConnectionData.mockResolvedValue(fakeChatModel);

		const model = await getChatModel(mockContext);
		expect(model).toEqual(fakeChatModel);
	});

	it('should throw if the model is not a valid chat model', async () => {
		const fakeInvalidModel = mock<BaseChatModel>(); // missing bindTools & lc_namespace
		fakeInvalidModel.lc_namespace = [];
		mockContext.getInputConnectionData.mockResolvedValue(fakeInvalidModel);
		mockContext.getNode.mockReturnValue(mock());
		await expect(getChatModel(mockContext)).rejects.toThrow(NodeOperationError);
	});

	it('should return the first model when multiple models are connected and no index specified', async () => {
		const fakeChatModel1 = new FakeStreamingChatModel({});
		const fakeChatModel2 = new FakeStreamingChatModel({});

		mockContext.getInputConnectionData.mockResolvedValue([fakeChatModel1, fakeChatModel2]);

		const model = await getChatModel(mockContext);
		expect(model).toEqual(fakeChatModel2); // Should return the last model (reversed array)
	});

	it('should return the model at specified index when multiple models are connected', async () => {
		const fakeChatModel1 = new FakeStreamingChatModel({});

		const fakeChatModel2 = new FakeStreamingChatModel({});

		mockContext.getInputConnectionData.mockResolvedValue([fakeChatModel1, fakeChatModel2]);

		const model = await getChatModel(mockContext, 0);
		expect(model).toEqual(fakeChatModel2); // Should return the first model after reversal (index 0)
	});

	it('should return the fallback model at index 1 when multiple models are connected', async () => {
		const fakeChatModel1 = new FakeStreamingChatModel({});
		const fakeChatModel2 = new FakeStreamingChatModel({});

		mockContext.getInputConnectionData.mockResolvedValue([fakeChatModel1, fakeChatModel2]);

		const model = await getChatModel(mockContext, 1);
		expect(model).toEqual(fakeChatModel1); // Should return the second model after reversal (index 1)
	});

	it('should return undefined when requested index is out of bounds', async () => {
		const fakeChatModel1 = mock<BaseChatModel>();
		fakeChatModel1.bindTools = jest.fn();
		fakeChatModel1.lc_namespace = ['chat_models'];

		mockContext.getInputConnectionData.mockResolvedValue([fakeChatModel1]);
		mockContext.getNode.mockReturnValue(mock());

		const result = await getChatModel(mockContext, 2);

		expect(result).toBeUndefined();
	});

	it('should throw error when single model does not support tools', async () => {
		const fakeInvalidModel = new FakeLLM({}); // doesn't support tool calls

		mockContext.getInputConnectionData.mockResolvedValue(fakeInvalidModel);
		mockContext.getNode.mockReturnValue(mock());

		await expect(getChatModel(mockContext)).rejects.toThrow(NodeOperationError);
		await expect(getChatModel(mockContext)).rejects.toThrow(
			'Tools Agent requires Chat Model which supports Tools calling',
		);
	});

	it('should throw error when model at specified index does not support tools', async () => {
		const fakeChatModel1 = new FakeStreamingChatModel({});
		const fakeInvalidModel = new FakeLLM({}); // doesn't support tool calls

		mockContext.getInputConnectionData.mockResolvedValue([fakeChatModel1, fakeInvalidModel]);
		mockContext.getNode.mockReturnValue(mock());

		await expect(getChatModel(mockContext, 0)).rejects.toThrow(NodeOperationError);
	});
});

describe('getOptionalMemory', () => {
	it('should return the memory if available', async () => {
		const fakeMemory = { some: 'memory' };
		mockContext.getInputConnectionData.mockResolvedValue(fakeMemory);

		const memory = await getOptionalMemory(mockContext);
		expect(memory).toEqual(fakeMemory);
	});
});

describe('getTools', () => {
	beforeEach(() => {
		const fakeTool = mock<Tool>();
		mockContext.getInputConnectionData
			.calledWith(NodeConnectionTypes.AiTool, 0)
			.mockResolvedValue([fakeTool]);
	});

	it('should retrieve tools without appending if outputParser is not provided', async () => {
		const tools = await getTools(mockContext);

		expect(tools.length).toEqual(1);
	});

	it('should retrieve tools and append the structured output parser tool if outputParser is provided', async () => {
		const fakeOutputParser = getFakeOutputParser(z.object({ text: z.string() }));
		const tools = await getTools(mockContext, fakeOutputParser);
		// Our fake getConnectedTools returns one tool; with outputParser, one extra is appended.
		expect(tools.length).toEqual(2);
		const dynamicTool = tools.find((t) => t.name === 'format_final_json_response');
		expect(dynamicTool).toBeDefined();
	});
});

describe('prepareMessages', () => {
	it('should include a binary message if binary data is present and passthroughBinaryImages is true', async () => {
		const fakeItem = {
			json: {},
			binary: {
				img1: {
					mimeType: 'image/png',
					data: 'data:image/png;base64,sampledata',
				},
			},
		};
		mockContext.getInputData.mockReturnValue([fakeItem]);
		const messages = await prepareMessages(mockContext, 0, {
			systemMessage: 'Test system',
			passthroughBinaryImages: true,
		});
		// Check if any message is an instance of HumanMessage
		const hasBinaryMessage = messages.some(
			(m) => typeof m === 'object' && m instanceof HumanMessage,
		);
		expect(hasBinaryMessage).toBe(true);
	});

	it('should not include a binary message if no binary data is present', async () => {
		const fakeItem = { json: {} }; // no binary key
		mockContext.getInputData.mockReturnValue([fakeItem]);
		const messages = await prepareMessages(mockContext, 0, {
			systemMessage: 'Test system',
			passthroughBinaryImages: true,
		});
		const hasHumanMessage = messages.some((m) => m instanceof HumanMessage);
		expect(hasHumanMessage).toBe(false);
	});

	it('should not include a binary message if no image data is present', async () => {
		const fakeItem = {
			json: {},
			binary: {
				img1: {
					mimeType: 'application/pdf',
					data: 'data:application/pdf;base64,sampledata',
				},
			},
		};
		mockContext.getInputData.mockReturnValue([fakeItem]);
		mockContext.logger = {
			debug: jest.fn(),
			info: jest.fn(),
			warn: jest.fn(),
			error: jest.fn(),
		};

		const messages = await prepareMessages(mockContext, 0, {
			systemMessage: 'Test system',
			passthroughBinaryImages: true,
		});
		const hasHumanMessage = messages.some((m) => m instanceof HumanMessage);
		expect(hasHumanMessage).toBe(false);
		expect(mockContext.logger.debug).toHaveBeenCalledTimes(1);
	});

	it('should not include system_message in prompt templates if not provided after version 1.9', async () => {
		const fakeItem = { json: {} };
		const mockNode = mock<INode>();
		mockNode.typeVersion = 1.9;
		mockContext.getInputData.mockReturnValue([fakeItem]);
		mockContext.getNode.mockReturnValue(mockNode);
		const messages = await prepareMessages(mockContext, 0, {});

		expect(messages.length).toBe(3);
		expect(messages).not.toContainEqual(['system', '{system_message}']);
	});

	it('should include system_message in prompt templates if provided after version 1.9', async () => {
		const fakeItem = { json: {} };
		const mockNode = mock<INode>();
		mockNode.typeVersion = 1.9;
		mockContext.getInputData.mockReturnValue([fakeItem]);
		mockContext.getNode.mockReturnValue(mockNode);

		const messages = await prepareMessages(mockContext, 0, { systemMessage: 'Hello' });

		expect(messages.length).toBe(4);
		expect(messages).toContainEqual(['system', '{system_message}']);
	});

	it('should include system_message in prompt templates if not provided before version 1.9', async () => {
		const fakeItem = { json: {} };
		const mockNode = mock<INode>();
		mockNode.typeVersion = 1.8;
		mockContext.getInputData.mockReturnValue([fakeItem]);
		mockContext.getNode.mockReturnValue(mockNode);

		const messages = await prepareMessages(mockContext, 0, {});

		expect(messages.length).toBe(4);
		expect(messages).toContainEqual(['system', '{system_message}']);
	});

	it('should include system_message with formatting_instructions in prompt templates if provided before version 1.9', async () => {
		const fakeItem = { json: {} };
		const mockNode = mock<INode>();
		mockNode.typeVersion = 1.8;
		mockContext.getInputData.mockReturnValue([fakeItem]);
		mockContext.getNode.mockReturnValue(mockNode);

		const messages = await prepareMessages(mockContext, 0, {
			systemMessage: 'Hello',
			outputParser: mock<N8nOutputParser>(),
		});

		expect(messages.length).toBe(4);
		expect(messages).toContainEqual(['system', '{system_message}\n\n{formatting_instructions}']);
	});

	it('should add formatting instructions when omitting system message after version 1.9', async () => {
		const fakeItem = { json: {} };
		const mockNode = mock<INode>();
		mockNode.typeVersion = 1.9;
		mockContext.getInputData.mockReturnValue([fakeItem]);
		mockContext.getNode.mockReturnValue(mockNode);

		const messages = await prepareMessages(mockContext, 0, {
			outputParser: mock<N8nOutputParser>(),
		});

		expect(messages.length).toBe(4);
		expect(messages).toContainEqual(['system', '{formatting_instructions}']);
	});
});

describe('preparePrompt', () => {
	it('should return a ChatPromptTemplate instance', () => {
		const sampleMessages: BaseMessagePromptTemplateLike[] = [
			['system', 'Test'],
			['human', 'Hello'],
		];
		const prompt = preparePrompt(sampleMessages);

		expect(prompt).toBeDefined();
	});
});

describe('getAgentStepsParser', () => {
	let mockMemory: BaseChatMemory;

	beforeEach(() => {
		mockMemory = mock<BaseChatMemory>();
	});

	describe('with format_final_json_response tool', () => {
		it('should parse output from format_final_json_response tool', async () => {
			const steps: AgentAction[] = [
				{
					tool: 'format_final_json_response',
					toolInput: { city: 'Berlin', temperature: 15 },
					log: '',
				},
			];

			const mockOutputParser = createMockOutputParser({
				city: 'Berlin',
				temperature: 15,
			});

			const parser = getAgentStepsParser(mockOutputParser, mockMemory);
			const result = await parser(steps);

			expect(mockOutputParser.parse).toHaveBeenCalledWith('{"city":"Berlin","temperature":15}');
			expect(result).toEqual({
				returnValues: { output: '{"city":"Berlin","temperature":15}' },
				log: 'Final response formatted',
			});
		});

		it('should stringify tool input if it is not an object', async () => {
			const steps: AgentAction[] = [
				{
					tool: 'format_final_json_response',
					toolInput: 'simple string',
					log: '',
				},
			];

			const mockOutputParser = createMockOutputParser({ text: 'simple string' });

			const parser = getAgentStepsParser(mockOutputParser, mockMemory);
			const result = await parser(steps);

			expect(mockOutputParser.parse).toHaveBeenCalledWith('simple string');
			expect(result).toEqual({
				returnValues: { output: '{"text":"simple string"}' },
				log: 'Final response formatted',
			});
		});
	});

	describe('manual parsing path', () => {
		it('should handle already wrapped output structure correctly', async () => {
			// Agent returns output that already has { output: {...} } structure
			const steps: AgentFinish = {
				returnValues: {
					output: '{"output":{"city":"Berlin","temperature":15}}',
				},
				log: '',
			};

			const mockOutputParser = createMockOutputParser({
				city: 'Berlin',
				temperature: 15,
			});

			const parser = getAgentStepsParser(mockOutputParser, mockMemory);
			const result = await parser(steps);

			// Should detect the existing wrapper and not double-wrap
			expect(mockOutputParser.parse).toHaveBeenCalledWith(
				'{"output":{"city":"Berlin","temperature":15}}',
			);
			expect(result).toEqual({
				returnValues: { output: '{"city":"Berlin","temperature":15}' },
				log: 'Final response formatted',
			});
		});

		it('should wrap output that is not already wrapped', async () => {
			// Agent returns plain data without { output: ... } wrapper
			const steps: AgentFinish = {
				returnValues: {
					output: '{"city":"Berlin","temperature":15}',
				},
				log: '',
			};

			const mockOutputParser = createMockOutputParser({
				city: 'Berlin',
				temperature: 15,
			});

			const parser = getAgentStepsParser(mockOutputParser, mockMemory);
			const result = await parser(steps);

			// Should wrap the data in { output: ... } for the parser
			expect(mockOutputParser.parse).toHaveBeenCalledWith(
				'{"output":{"city":"Berlin","temperature":15}}',
			);
			expect(result).toEqual({
				returnValues: { output: '{"city":"Berlin","temperature":15}' },
				log: 'Final response formatted',
			});
		});

		it('should handle output with additional properties correctly', async () => {
			// Output has more than just the "output" property
			const steps: AgentFinish = {
				returnValues: {
					output: '{"output":{"text":"Hello"},"metadata":{"source":"test"}}',
				},
				log: '',
			};

			const mockOutputParser = createMockOutputParser({
				text: 'Hello',
				metadata: { source: 'test' },
			});

			const parser = getAgentStepsParser(mockOutputParser, mockMemory);
			const result = await parser(steps);

			// Should wrap since it has multiple properties
			expect(mockOutputParser.parse).toHaveBeenCalledWith(
				'{"output":{"output":{"text":"Hello"},"metadata":{"source":"test"}}}',
			);
			expect(result).toEqual({
				returnValues: { output: '{"text":"Hello","metadata":{"source":"test"}}' },
				log: 'Final response formatted',
			});
		});

		it('should handle parse errors gracefully', async () => {
			const steps: AgentFinish = {
				returnValues: {
					output: 'invalid json',
				},
				log: '',
			};

			const mockOutputParser = createMockOutputParser({ text: 'invalid json' });

			const parser = getAgentStepsParser(mockOutputParser, mockMemory);
			const result = await parser(steps);

			// Should fallback to raw output when JSON parsing fails
			expect(mockOutputParser.parse).toHaveBeenCalledWith('invalid json');
			expect(result).toEqual({
				returnValues: { output: '{"text":"invalid json"}' },
				log: 'Final response formatted',
			});
		});

		it('should handle null output correctly', async () => {
			const steps: AgentFinish = {
				returnValues: {
					output: 'null',
				},
				log: '',
			};

			const mockOutputParser = createMockOutputParser({ result: null });

			const parser = getAgentStepsParser(mockOutputParser, mockMemory);
			const result = await parser(steps);

			// Should wrap null in { output: null }
			expect(mockOutputParser.parse).toHaveBeenCalledWith('{"output":null}');
			expect(result).toEqual({
				returnValues: { output: '{"result":null}' },
				log: 'Final response formatted',
			});
		});

		it('should handle undefined-like values correctly', async () => {
			const steps: AgentFinish = {
				returnValues: {
					output: 'undefined',
				},
				log: '',
			};

			const mockOutputParser = createMockOutputParser({ text: 'undefined' });

			const parser = getAgentStepsParser(mockOutputParser, mockMemory);
			const result = await parser(steps);

			// Should fallback to raw string since "undefined" is not valid JSON
			expect(mockOutputParser.parse).toHaveBeenCalledWith('undefined');
			expect(result).toEqual({
				returnValues: { output: '{"text":"undefined"}' },
				log: 'Final response formatted',
			});
		});

		it('should return output as-is without memory', async () => {
			const steps: AgentFinish = {
				returnValues: {
					output: '{"city":"Berlin","temperature":15}',
				},
				log: '',
			};

			const mockOutputParser = createMockOutputParser({
				city: 'Berlin',
				temperature: 15,
			});

			const parser = getAgentStepsParser(mockOutputParser, undefined);
			const result = await parser(steps);

			expect(result).toEqual({
				returnValues: { city: 'Berlin', temperature: 15 },
				log: 'Final response formatted',
			});
		});
	});

	describe('without output parser', () => {
		it('should pass through agent finish steps unchanged', async () => {
			const steps: AgentFinish = {
				returnValues: { output: 'Final answer' },
				log: '',
			};

			const parser = getAgentStepsParser(undefined, undefined);
			const result = await parser(steps);

			expect(result).toEqual({
				log: '',
				returnValues: { output: 'Final answer' },
			});
		});

		it('should handle array of agent actions', async () => {
			const steps: AgentAction[] = [
				{ tool: 'some_tool', toolInput: { query: 'test' }, log: '' },
				{ tool: 'another_tool', toolInput: { data: 'value' }, log: '' },
			];

			const parser = getAgentStepsParser(undefined, undefined);
			const result = await parser(steps);

			expect(result).toEqual(steps);
		});
	});

	describe('jsonSchema mode', () => {
		it('should trust provider output and skip local output parser', async () => {
			const steps: AgentFinish = {
				returnValues: {
					output: '{"city":"Berlin","temperature":15}',
				},
				log: '',
			};

			const mockOutputParser = createMockOutputParser({
				city: 'Paris',
				temperature: 24,
			});

			const parser = getAgentStepsParser(mockOutputParser, mockMemory, 'jsonSchema');
			const result = await parser(steps);

			expect(mockOutputParser.parse).not.toHaveBeenCalled();
			expect(result).toEqual({
				returnValues: { output: '{"city":"Berlin","temperature":15}' },
				log: 'Final response formatted',
			});
		});

		it('should wrap primitive trusted output in output key', async () => {
			const steps: AgentFinish = {
				returnValues: {
					output: '"Berlin"',
				},
				log: '',
			};

			const mockOutputParser = createMockOutputParser({ text: 'should not be used' });

			const parser = getAgentStepsParser(mockOutputParser, mockMemory, 'jsonSchema');
			const result = await parser(steps);

			expect(mockOutputParser.parse).not.toHaveBeenCalled();
			expect(result).toEqual({
				returnValues: { output: '{"output":"Berlin"}' },
				log: 'Final response formatted',
			});
		});
	});
});

describe('makeSchemaStrict enum sanitization', () => {
	it('should replace newlines/tabs with spaces in enum values', () => {
		const enumMapping = new Map<string, string>();
		const schema = {
			type: 'object',
			properties: {
				content: {
					type: 'string',
					enum: ['line1\nline2\ttab\rcarriage'],
				},
			},
		};
		const result = makeSchemaStrict(schema, enumMapping);
		expect(result.properties.content.enum).toEqual(['line1 line2 tab carriage']);
		expect(enumMapping.get('line1 line2 tab carriage')).toBe('line1\nline2\ttab\rcarriage');
	});

	it('should not create mapping entries for values without control chars', () => {
		const enumMapping = new Map<string, string>();
		const schema = {
			type: 'string',
			enum: ['clean-value', 'another'],
		};
		const result = makeSchemaStrict(schema, enumMapping);
		expect(result.enum).toEqual(['clean-value', 'another']);
		expect(enumMapping.size).toBe(0);
	});

	it('should handle emojis and special unicode without mangling', () => {
		const enumMapping = new Map<string, string>();
		const schema = {
			type: 'string',
			enum: ['🚨 Breaking\n🇺🇸 News'],
		};
		const result = makeSchemaStrict(schema, enumMapping);
		expect(result.enum).toEqual(['🚨 Breaking 🇺🇸 News']);
		expect(enumMapping.get('🚨 Breaking 🇺🇸 News')).toBe('🚨 Breaking\n🇺🇸 News');
	});

	it('should work without enumMapping parameter', () => {
		const schema = { type: 'string', enum: ['a\nb'] };
		const result = makeSchemaStrict(schema);
		expect(result.enum).toEqual(['a b']);
	});
});

describe('restoreEnumValues', () => {
	it('should restore sanitized string values', () => {
		const mapping = new Map([['line1 line2', 'line1\nline2']]);
		expect(restoreEnumValues('line1 line2', mapping)).toBe('line1\nline2');
	});

	it('should leave non-mapped values unchanged', () => {
		const mapping = new Map([['line1 line2', 'line1\nline2']]);
		expect(restoreEnumValues('other', mapping)).toBe('other');
	});

	it('should recurse into objects', () => {
		const mapping = new Map([['a b', 'a\nb']]);
		const input = { content: 'a b', category: 'news' };
		expect(restoreEnumValues(input, mapping)).toEqual({ content: 'a\nb', category: 'news' });
	});

	it('should recurse into arrays', () => {
		const mapping = new Map([['a b', 'a\nb']]);
		const input = [{ content: 'a b' }, { content: 'untouched' }];
		expect(restoreEnumValues(input, mapping)).toEqual([
			{ content: 'a\nb' },
			{ content: 'untouched' },
		]);
	});

	it('should handle nested structure matching real workflow output', () => {
		const mapping = new Map([
			[
				'🚨#Breaking:   🇺🇸🇮🇷 USS FORD ORDERED Stay informed.',
				'🚨#Breaking:  \n🇺🇸🇮🇷 USS FORD ORDERED\nStay informed.',
			],
		]);
		const input = {
			analyzed_news: [
				{
					content: '🚨#Breaking:   🇺🇸🇮🇷 USS FORD ORDERED Stay informed.',
					category: 'middle-east',
				},
			],
		};
		const result = restoreEnumValues(input, mapping) as any;
		expect(result.analyzed_news[0].content).toBe(
			'🚨#Breaking:  \n🇺🇸🇮🇷 USS FORD ORDERED\nStay informed.',
		);
		expect(result.analyzed_news[0].category).toBe('middle-east');
	});

	it('should pass through non-string/object/array values', () => {
		const mapping = new Map([['a', 'b']]);
		expect(restoreEnumValues(42, mapping)).toBe(42);
		expect(restoreEnumValues(null, mapping)).toBe(null);
		expect(restoreEnumValues(true, mapping)).toBe(true);
	});
});

describe('getAgentStepsParser jsonSchema mode with enum mapping', () => {
	let mockMemory: BaseChatMemory;

	beforeEach(() => {
		mockMemory = mock<BaseChatMemory>();
	});

	it('should restore sanitized enum values in jsonSchema mode', async () => {
		const enumMapping = new Map([['🚨 Breaking  Stay informed.', '🚨 Breaking\n\nStay informed.']]);
		const steps: AgentFinish = {
			returnValues: {
				output: '{"content":"🚨 Breaking  Stay informed.","cat":"news"}',
			},
			log: '',
		};
		const mockOutputParser = createMockOutputParser({});
		const parser = getAgentStepsParser(mockOutputParser, mockMemory, 'jsonSchema', enumMapping);
		const result = (await parser(steps)) as AgentFinish;
		expect(mockOutputParser.parse).not.toHaveBeenCalled();
		expect(result.returnValues.output).toContain('🚨 Breaking\\n\\nStay informed.');
	});
});

describe('handleAgentFinishOutput', () => {
	it('should merge multi-output text arrays into a single string', () => {
		const steps: AgentFinish = {
			returnValues: {
				output: [
					{ index: 0, type: 'text', text: 'First part' },
					{ index: 1, type: 'text', text: 'Second part' },
				],
			},
			log: '',
		};

		const result = handleAgentFinishOutput(steps);

		expect(result).toEqual({
			log: '',
			returnValues: {
				output: 'First part\nSecond part',
			},
		});
	});

	it('should not modify non-text multi-output arrays', () => {
		const steps: AgentFinish = {
			returnValues: {
				output: [
					{ index: 0, type: 'text', text: 'Text part' },
					{ index: 1, type: 'image', url: 'http://example.com/image.png' },
				],
			},
			log: '',
		};

		const result = handleAgentFinishOutput(steps);

		expect(result).toEqual(steps);
	});

	it('should not modify simple string output', () => {
		const steps: AgentFinish = {
			returnValues: {
				output: 'Simple string output',
			},
			log: '',
		};

		const result = handleAgentFinishOutput(steps);

		expect(result).toEqual(steps);
	});

	it('should handle agent action arrays unchanged', () => {
		const steps: AgentAction[] = [
			{
				tool: 'tool1',
				toolInput: {},
				log: '',
			},
			{
				tool: 'tool2',
				toolInput: {},
				log: '',
			},
		];

		const result = handleAgentFinishOutput(steps);

		expect(result).toEqual(steps);
	});

	it('should filter out thinking blocks and return only text blocks', () => {
		const steps: AgentFinish = {
			returnValues: {
				output: [
					{ index: 0, type: 'thinking', thinking: 'Internal reasoning...' },
					{ index: 1, type: 'text', text: 'User-facing output' },
				],
			},
			log: '',
		};

		const result = handleAgentFinishOutput(steps) as AgentFinish;

		expect(result.returnValues.output).toBe('User-facing output');
	});

	it('should return thinking content when no text blocks exist', () => {
		const steps: AgentFinish = {
			returnValues: {
				output: [
					{ index: 0, type: 'thinking', thinking: 'Only thinking content' },
					{ index: 1, type: 'thinking', thinking: 'More thinking' },
				],
			},
			log: '',
		};

		const result = handleAgentFinishOutput(steps) as AgentFinish;

		expect(result.returnValues.output).toBe('Only thinking content\nMore thinking');
	});

	it('should return empty string when no text or thinking blocks exist', () => {
		const steps: AgentFinish = {
			returnValues: {
				output: [{ index: 0, type: 'unknown' }],
			},
			log: '',
		};

		const result = handleAgentFinishOutput(steps) as AgentFinish;

		expect(result.returnValues.output).toBe('');
	});
});

describe('makeSchemaStrict', () => {
	it('should sanitize enum values containing newlines and record mapping', () => {
		const enumMapping = new Map<string, string>();
		const schema = {
			type: 'string',
			enum: [
				'🚨#Breaking:  \n🇺🇸🇮🇷 USS GERALD R. FORD ORDERED\nStay informed.',
				'Line one\nLine two\nLine three',
			],
		};

		const result = makeSchemaStrict(schema, enumMapping);

		// Newlines replaced with spaces for strict-mode compatibility
		expect(result.enum).toEqual([
			'🚨#Breaking:   🇺🇸🇮🇷 USS GERALD R. FORD ORDERED Stay informed.',
			'Line one Line two Line three',
		]);
		// Mapping records originals
		expect(enumMapping.size).toBe(2);
		expect(enumMapping.get('🚨#Breaking:   🇺🇸🇮🇷 USS GERALD R. FORD ORDERED Stay informed.')).toBe(
			'🚨#Breaking:  \n🇺🇸🇮🇷 USS GERALD R. FORD ORDERED\nStay informed.',
		);
	});

	it('should sanitize enum values containing tabs and carriage returns', () => {
		const enumMapping = new Map<string, string>();
		const schema = {
			type: 'string',
			enum: ['col1\tcol2\tcol3', 'line\r\nbreak'],
		};

		const result = makeSchemaStrict(schema, enumMapping);

		expect(result.enum).toEqual(['col1 col2 col3', 'line  break']);
		expect(enumMapping.size).toBe(2);
	});

	it('should preserve enum values with emojis and special unicode', () => {
		const schema = {
			type: 'string',
			enum: ['🇺🇸 USA', '🇷🇺 Russia', '🇮🇷 Iran'],
		};

		const result = makeSchemaStrict(schema);

		expect(result.enum).toEqual(['🇺🇸 USA', '🇷🇺 Russia', '🇮🇷 Iran']);
	});

	it('should convert const with newlines to sanitized enum', () => {
		const enumMapping = new Map<string, string>();
		const schema = {
			type: 'string',
			const: 'First line\nSecond line',
		};

		const result = makeSchemaStrict(schema, enumMapping);

		expect(result.const).toBeUndefined();
		expect(result.enum).toEqual(['First line Second line']);
		expect(enumMapping.get('First line Second line')).toBe('First line\nSecond line');
	});

	it('should sanitize enum values in nested object schemas', () => {
		const enumMapping = new Map<string, string>();
		const schema = {
			type: 'object',
			properties: {
				content: {
					type: 'string',
					enum: ['🚨#Breaking:\n🇺🇸 USS FORD ORDERED TO MIDDLE EAST'],
				},
				category: {
					type: 'string',
					enum: ['ukraine-russia', 'middle-east'],
				},
			},
		};

		const result = makeSchemaStrict(schema, enumMapping);

		expect(result.properties.content.enum).toEqual([
			'🚨#Breaking: 🇺🇸 USS FORD ORDERED TO MIDDLE EAST',
		]);
		// Clean values should not generate mapping entries
		expect(result.properties.category.enum).toEqual(['ukraine-russia', 'middle-east']);
		expect(enumMapping.size).toBe(1);
		expect(result.additionalProperties).toBe(false);
		expect(result.required).toEqual(['content', 'category']);
	});

	it('should sanitize enum values in array item schemas', () => {
		const enumMapping = new Map<string, string>();
		const schema = {
			type: 'array',
			items: {
				type: 'object',
				properties: {
					content: {
						type: 'string',
						enum: ['Post #1\nWith newline', 'Post #2\tWith tab'],
					},
				},
			},
		};

		const result = makeSchemaStrict(schema, enumMapping);

		expect(result.items.properties.content.enum).toEqual([
			'Post #1 With newline',
			'Post #2 With tab',
		]);
		expect(enumMapping.size).toBe(2);
		expect(result.items.additionalProperties).toBe(false);
	});

	it('should strip unsupported keywords while sanitizing enum', () => {
		const enumMapping = new Map<string, string>();
		const schema = {
			type: 'string',
			enum: ['value\nwith newline'],
			$schema: 'http://json-schema.org/draft-07/schema#',
			default: 'some default',
			minLength: 1,
			format: 'uri',
		};

		const result = makeSchemaStrict(schema, enumMapping);

		expect(result.enum).toEqual(['value with newline']);
		expect(enumMapping.get('value with newline')).toBe('value\nwith newline');
		expect(result.$schema).toBeUndefined();
		expect(result.default).toBeUndefined();
		expect(result.minLength).toBeUndefined();
		expect(result.format).toBeUndefined();
	});

	it('should produce valid JSON and sanitized enum for strict-mode acceptance', () => {
		const enumMapping = new Map<string, string>();
		const rawContent =
			'🚨#Breaking:  \n🇺🇸🇮🇷 USS GERALD R. FORD ORDERED FROM VENEZUELA TO MIDDLE EAST, JOINING LINCOLN IN PERSIAN GULF\nStay informed. Follow @rageintel';

		const schema = {
			type: 'object',
			properties: {
				analyzed_news: {
					type: 'array',
					items: {
						type: 'object',
						properties: {
							content: {
								type: 'string',
								description: 'Must be an exact match from the input list',
								enum: [rawContent],
							},
							category: {
								type: 'string',
								enum: ['ukraine-russia', 'middle-east', 'Europe', 'USA', 'other'],
							},
						},
						required: ['content', 'category'],
					},
				},
			},
		};

		const result = makeSchemaStrict(schema, enumMapping);

		// The enum value must not contain literal newlines (strict-mode forbids them)
		const sanitizedEnum = result.properties.analyzed_news.items.properties.content.enum[0];
		expect(sanitizedEnum).not.toContain('\n');
		expect(sanitizedEnum).not.toContain('\r');
		expect(sanitizedEnum).not.toContain('\t');

		// Original must be recoverable via mapping
		expect(enumMapping.get(sanitizedEnum)).toBe(rawContent);

		// Verify round-trip through JSON is clean
		const serialized = JSON.stringify(result);
		expect(() => JSON.parse(serialized)).not.toThrow();
	});
});
