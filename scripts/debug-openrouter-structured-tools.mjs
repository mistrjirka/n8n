#!/usr/bin/env node

const apiKey = process.env.OPENROUTER_API_KEY;
const model = process.env.OPENROUTER_MODEL ?? 'openai/gpt-5-mini';
const baseUrl = process.env.OPENROUTER_BASE_URL ?? 'https://openrouter.ai/api/v1/chat/completions';

if (!apiKey) {
	console.error('Missing OPENROUTER_API_KEY');
	console.error('Usage: OPENROUTER_API_KEY=sk-or-... node scripts/debug-openrouter-structured-tools.mjs');
	process.exit(1);
}

const sanitizeStrictLiteral = (value) =>
	value
		.replace(/\\/g, '\\\\')
		.replace(/\r/g, '\\r')
		.replace(/\n/g, '\\n')
		.replace(/\t/g, '\\t')
		.replace(/[\u0000-\u001F]/g, ' ');

const rawContentLiteral =
	'🚨#Breaking:  \n🇺🇸🇮🇷 USS GERALD R. FORD ORDERED FROM VENEZUELA TO MIDDLE EAST, JOINING LINCOLN IN PERSIAN GULF\nStay informed. Follow @rageintel';

const responseFormat = {
	type: 'json_schema',
	json_schema: {
		name: 'structured_response',
		strict: true,
		schema: {
			type: 'object',
			properties: {
				output: {
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
										enum: [
											rawContentLiteral,
										],
									},
									link: { type: 'string' },
									category: {
										type: 'string',
										enum: ['ukraine-russia', 'middle-east', 'Europe', 'USA', 'other'],
									},
									verification: {
										type: 'string',
										enum: [
											'probably-true',
											'probably-false',
											'madeup',
											'unverifiable',
										],
									},
									verification_reasoning: { type: 'string' },
									interest_reason: { type: 'string' },
								},
								required: [
									'content',
									'link',
									'category',
									'verification',
									'verification_reasoning',
									'interest_reason',
								],
								additionalProperties: false,
							},
						},
					},
					required: ['analyzed_news'],
					additionalProperties: false,
				},
			},
			required: ['output'],
			additionalProperties: false,
		},
	},
};

const responseFormatSanitized = JSON.parse(JSON.stringify(responseFormat));
responseFormatSanitized.json_schema.schema.properties.output.properties.analyzed_news.items.properties.content.enum = [
	sanitizeStrictLiteral(rawContentLiteral),
];

const makeTool = ({ includeStrict = true, includeRequired = true, includeSchema = false } = {}) => {
	const parameters = {
		type: 'object',
		properties: {
			input: { type: 'string' },
		},
		additionalProperties: false,
	};

	if (includeRequired) parameters.required = ['input'];
	if (includeSchema) parameters.$schema = 'http://json-schema.org/draft-07/schema#';

	const fn = {
		name: 'Think',
		description:
			'Use the tool to think about something. It will not obtain new information or change the database, but just append the thought to the log.',
		parameters,
	};

	if (includeStrict) fn.strict = true;

	return {
		type: 'function',
		function: fn,
	};
};

const baseMessages = [
	{
		role: 'user',
		content:
			'Analyze this news: 🚨#Breaking: USS GERALD R. FORD ORDERED FROM VENEZUELA TO MIDDLE EAST, JOINING LINCOLN IN PERSIAN GULF',
	},
];

const tests = [
	{
		name: 'A_current_failure_shape',
		expectOk: false,
		payload: {
			model,
			messages: baseMessages,
			response_format: responseFormat,
			tools: [makeTool({ includeStrict: true, includeRequired: false, includeSchema: true })],
			stream: false,
		},
	},
	{
		name: 'B_strict_required_no_schema',
		expectOk: false,
		payload: {
			model,
			messages: baseMessages,
			response_format: responseFormat,
			tools: [makeTool({ includeStrict: true, includeRequired: true, includeSchema: false })],
			stream: false,
		},
	},
	{
		name: 'C_strict_required_with_schema',
		expectOk: false,
		payload: {
			model,
			messages: baseMessages,
			response_format: responseFormat,
			tools: [makeTool({ includeStrict: true, includeRequired: true, includeSchema: true })],
			stream: false,
		},
	},
	{
		name: 'D_response_format_only',
		expectOk: false,
		payload: {
			model,
			messages: baseMessages,
			response_format: responseFormat,
			stream: false,
		},
	},
	{
		name: 'E_tools_only_strict_required',
		expectOk: true,
		payload: {
			model,
			messages: baseMessages,
			tools: [makeTool({ includeStrict: true, includeRequired: true, includeSchema: false })],
			stream: false,
		},
	},
	{
		name: 'F_expected_success_sanitized_response_format_plus_tools',
		expectOk: true,
		payload: {
			model,
			messages: baseMessages,
			response_format: responseFormatSanitized,
			tools: [makeTool({ includeStrict: true, includeRequired: true, includeSchema: false })],
			stream: false,
		},
	},
];

async function runTest(test) {
	const start = Date.now();
	const res = await fetch(baseUrl, {
		method: 'POST',
		headers: {
			Authorization: `Bearer ${apiKey}`,
			'Content-Type': 'application/json',
			'HTTP-Referer': 'http://localhost:5678',
			'X-Title': 'n8n Structured Tools Debug',
		},
		body: JSON.stringify(test.payload),
	});
	const text = await res.text();
	let parsed;
	try {
		parsed = JSON.parse(text);
	} catch {
		parsed = { raw: text };
	}
	return {
		name: test.name,
		status: res.status,
		ok: res.ok,
		ms: Date.now() - start,
		body: parsed,
	};
}

(async () => {
	console.log(`Model: ${model}`);
	console.log(`Endpoint: ${baseUrl}`);
	console.log(`Running ${tests.length} tests...\n`);

	let failedExpectations = 0;

	for (const test of tests) {
		console.log(`=== ${test.name} ===`);
		const result = await runTest(test);
		console.log(`Status: ${result.status} (${result.ok ? 'OK' : 'FAIL'}) in ${result.ms}ms`);
		const matched = result.ok === test.expectOk;
		if (!matched) {
			failedExpectations++;
			console.log(
				`EXPECTATION MISMATCH: expected ${test.expectOk ? 'OK' : 'FAIL'}, got ${
					result.ok ? 'OK' : 'FAIL'
				}`,
			);
		}
		console.log(JSON.stringify(result.body, null, 2));
		console.log();
	}

	if (failedExpectations > 0) {
		console.error(`FAILED: ${failedExpectations} expectation(s) did not match.`);
		process.exit(1);
	}

	console.log('PASS: all expectations matched.');
})();
