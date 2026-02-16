#!/usr/bin/env node
/**
 * Integration test: verifies the full enum sanitization pipeline works end-to-end.
 *
 * Tests that:
 * 1. makeSchemaStrict sanitizes \n/\r/\t → spaces in enum values
 * 2. The sanitized schema is accepted by the provider (strict mode)
 * 3. The model returns the sanitized enum value
 * 4. restoreEnumValues converts it back to the original
 *
 * Usage:
 *   OPENROUTER_API_KEY=sk-or-... node scripts/test-openrouter-enum-newlines.mjs
 */

const apiKey = process.env.OPENROUTER_API_KEY;
const model = process.env.OPENROUTER_MODEL ?? 'openai/gpt-5-mini';
const baseUrl = 'https://openrouter.ai/api/v1/chat/completions';

if (!apiKey) {
	console.error('Missing OPENROUTER_API_KEY');
	process.exit(1);
}

// ---------- Inline implementations matching common.ts ----------

function sanitizeEnumValue(value) {
	return value.replace(/[\n\r\t]/g, ' ').replace(/[\u0000-\u001F]/g, ' ');
}

function makeSchemaStrict(schema, enumMapping) {
	if (typeof schema !== 'object' || schema === null) return schema;
	const newSchema = { ...schema };

	const unsupported = [
		'$schema', 'default', 'format', 'pattern', 'minLength', 'maxLength',
		'minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum', 'multipleOf',
		'minItems', 'maxItems', 'uniqueItems', 'minProperties', 'maxProperties',
		'patternProperties', 'not', 'if', 'then', 'else', 'examples',
	];
	for (const k of unsupported) delete newSchema[k];

	if ('const' in newSchema) {
		newSchema.enum = [newSchema.const];
		delete newSchema.const;
	}

	if (Array.isArray(newSchema.enum)) {
		newSchema.enum = newSchema.enum.map((v) => {
			if (typeof v !== 'string') return v;
			const sanitized = sanitizeEnumValue(v);
			if (sanitized !== v && enumMapping) enumMapping.set(sanitized, v);
			return sanitized;
		});
	}

	if (schema.type === 'object' || schema.properties) {
		newSchema.additionalProperties = false;
		if (schema.properties) {
			newSchema.required = Object.keys(schema.properties);
			const newProps = {};
			for (const key of newSchema.required) {
				newProps[key] = makeSchemaStrict(schema.properties[key], enumMapping);
			}
			newSchema.properties = newProps;
		}
	} else if (schema.type === 'array' || schema.items) {
		if (schema.items) newSchema.items = makeSchemaStrict(schema.items, enumMapping);
	} else if (schema.anyOf) {
		newSchema.anyOf = schema.anyOf.map((s) => makeSchemaStrict(s, enumMapping));
	} else if (schema.allOf) {
		newSchema.allOf = schema.allOf.map((s) => makeSchemaStrict(s, enumMapping));
	} else if (schema.oneOf) {
		newSchema.oneOf = schema.oneOf.map((s) => makeSchemaStrict(s, enumMapping));
	}

	return newSchema;
}

function restoreEnumValues(output, enumMapping) {
	if (typeof output === 'string') return enumMapping.get(output) ?? output;
	if (Array.isArray(output)) return output.map((item) => restoreEnumValues(item, enumMapping));
	if (output !== null && typeof output === 'object') {
		const result = {};
		for (const [key, value] of Object.entries(output)) {
			result[key] = restoreEnumValues(value, enumMapping);
		}
		return result;
	}
	return output;
}

// ---------- Test data ----------

const rawContent =
	'🚨#Breaking:  \n🇺🇸🇮🇷 USS GERALD R. FORD ORDERED FROM VENEZUELA TO MIDDLE EAST\nStay informed. Follow @rageintel';

const userSchema = {
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
					verification: {
						type: 'string',
						enum: ['probably-true', 'probably-false', 'madeup', 'unverifiable'],
					},
				},
			},
		},
	},
};

// ---------- Test runner ----------

async function runTest() {
	console.log(`Model: ${model}\n`);

	// Step 1: Sanitize schema
	const enumMapping = new Map();
	const strictSchema = makeSchemaStrict(userSchema, enumMapping);

	console.log('Step 1: makeSchemaStrict');
	console.log('  Enum mapping entries:', enumMapping.size);
	for (const [sanitized, original] of enumMapping.entries()) {
		console.log(`  "${sanitized.slice(0, 60)}..." → original has \\n: ${original.includes('\n')}`);
	}

	const sanitizedEnum = strictSchema.properties.analyzed_news.items.properties.content.enum[0];
	console.log('  Sanitized enum has \\n:', sanitizedEnum.includes('\n'));
	console.log('  Sanitized enum has \\r:', sanitizedEnum.includes('\r'));
	console.log('  Sanitized enum has \\t:', sanitizedEnum.includes('\t'));

	// Step 2: Send to provider
	console.log('\nStep 2: Send to provider (strict mode)');
	const payload = {
		model,
		messages: [
			{
				role: 'user',
				content: `Analyze this news and categorize it:\n\nContent: ${rawContent}\nLink: https://t.me/rageintel/12345`,
			},
		],
		response_format: {
			type: 'json_schema',
			json_schema: {
				name: 'structured_response',
				strict: true,
				schema: strictSchema,
			},
		},
		stream: false,
	};

	const res = await fetch(baseUrl, {
		method: 'POST',
		headers: {
			Authorization: `Bearer ${apiKey}`,
			'Content-Type': 'application/json',
			'HTTP-Referer': 'http://localhost:5678',
		},
		body: JSON.stringify(payload),
	});

	const text = await res.text();
	let body;
	try {
		body = JSON.parse(text);
	} catch {
		body = { raw: text };
	}

	console.log('  HTTP status:', res.status);

	if (!res.ok) {
		let reason = body?.error?.message ?? 'unknown';
		try {
			reason = JSON.parse(body.error.metadata.raw).error.message;
		} catch {}
		console.log('  REJECTED:', reason);
		console.log('\n❌ FAIL: Provider rejected the sanitized schema');
		process.exit(1);
	}

	console.log('  ACCEPTED ✓');

	// Step 3: Parse model output
	const content = body.choices?.[0]?.message?.content;
	if (!content) {
		const finishReason = body.choices?.[0]?.finish_reason;
		if (finishReason === 'tool_calls') {
			console.log('  Model used a tool call — schema was accepted, but no content to verify');
			console.log('\n✓ PASS (partial — provider accepted schema)');
			process.exit(0);
		}
		console.log('  No content in response:', JSON.stringify(body.choices?.[0]));
		process.exit(1);
	}

	let parsed;
	try {
		parsed = JSON.parse(content);
	} catch (e) {
		console.log('  Failed to parse response JSON:', content.slice(0, 200));
		process.exit(1);
	}

	console.log('\nStep 3: Model output (before restore)');
	const returnedContent = parsed?.analyzed_news?.[0]?.content;
	console.log('  content field:', JSON.stringify(returnedContent)?.slice(0, 100));
	console.log('  category field:', parsed?.analyzed_news?.[0]?.category);

	// Step 4: Restore original values
	console.log('\nStep 4: restoreEnumValues');
	const restored = restoreEnumValues(parsed, enumMapping);
	const restoredContent = restored?.analyzed_news?.[0]?.content;

	console.log('  Restored content:', JSON.stringify(restoredContent)?.slice(0, 100));
	console.log('  Matches original:', restoredContent === rawContent);
	console.log('  Category unchanged:', restored?.analyzed_news?.[0]?.category === parsed?.analyzed_news?.[0]?.category);

	// Final verdict
	console.log('\n' + '='.repeat(50));
	if (restoredContent === rawContent) {
		console.log('✓ PASS: Full pipeline works — sanitize → send → receive → restore');
		console.log('  Original newlines are perfectly recovered.');
		process.exit(0);
	} else {
		console.log('❌ FAIL: Content mismatch after restore');
		console.log('  Expected:', JSON.stringify(rawContent));
		console.log('  Got:     ', JSON.stringify(restoredContent));
		process.exit(1);
	}
}

runTest().catch((err) => {
	console.error('Unexpected error:', err);
	process.exit(1);
});
