#!/usr/bin/env node
/**
 * Proof: OpenAI strict mode rejects \n in enum string literals.
 *
 * Sends two identical requests differing ONLY in whether the enum value
 * contains a literal newline character or a space in its place.
 *
 * Usage:
 *   OPENROUTER_API_KEY=sk-or-... node scripts/prove-enum-newlines.mjs
 */

const apiKey = process.env.OPENROUTER_API_KEY;
const model = process.env.OPENROUTER_MODEL ?? 'openai/gpt-5-mini';
const baseUrl = 'https://openrouter.ai/api/v1/chat/completions';

if (!apiKey) {
	console.error('Missing OPENROUTER_API_KEY');
	process.exit(1);
}

// The exact content from the user's Telegram feed — contains a real \n
const contentWithNewlines =
	'🚨#Breaking:  \n🇺🇸🇮🇷 USS GERALD R. FORD ORDERED FROM VENEZUELA TO MIDDLE EAST\nStay informed.';

// Same content but \n replaced with space (our sanitization)
const contentSanitized = contentWithNewlines.replace(/\n/g, ' ');

function buildSchema(enumValue) {
	return {
		type: 'json_schema',
		json_schema: {
			name: 'structured_response',
			strict: true,
			schema: {
				type: 'object',
				properties: {
					analyzed_news: {
						type: 'array',
						items: {
							type: 'object',
							properties: {
								content: {
									type: 'string',
									enum: [enumValue],
								},
								category: {
									type: 'string',
									enum: ['ukraine-russia', 'middle-east', 'other'],
								},
							},
							required: ['content', 'category'],
							additionalProperties: false,
						},
					},
				},
				required: ['analyzed_news'],
				additionalProperties: false,
			},
		},
	};
}

async function sendRequest(label, enumValue) {
	const body = {
		model,
		messages: [
			{
				role: 'user',
				content: `Categorize this: ${contentSanitized}`,
			},
		],
		response_format: buildSchema(enumValue),
		stream: false,
	};

	console.log(`\n--- ${label} ---`);
	console.log(
		'Enum value (JSON):',
		JSON.stringify(enumValue).slice(0, 100) + (enumValue.length > 80 ? '...' : ''),
	);
	console.log('Contains literal \\n:', enumValue.includes('\n'));

	const res = await fetch(baseUrl, {
		method: 'POST',
		headers: {
			Authorization: `Bearer ${apiKey}`,
			'Content-Type': 'application/json',
			'HTTP-Referer': 'http://localhost:5678',
		},
		body: JSON.stringify(body),
	});

	const text = await res.text();
	let parsed;
	try {
		parsed = JSON.parse(text);
	} catch {
		parsed = { raw: text };
	}

	console.log('HTTP status:', res.status);

	if (!res.ok) {
		const rawError = parsed?.error?.metadata?.raw;
		if (rawError) {
			try {
				const inner = JSON.parse(rawError);
				console.log('Provider error:', inner.error?.message);
			} catch {
				console.log('Provider error:', rawError);
			}
		} else {
			console.log('Error:', JSON.stringify(parsed?.error?.message ?? parsed));
		}
	} else {
		const content = parsed?.choices?.[0]?.message?.content;
		if (content) {
			console.log('Model output:', content.slice(0, 200));
		} else {
			console.log('Response:', JSON.stringify(parsed?.choices?.[0]).slice(0, 200));
		}
	}

	return res.ok;
}

(async () => {
	console.log(`Model: ${model}`);
	console.log('='.repeat(60));

	const result1 = await sendRequest('TEST A: Enum with REAL newlines (\\n)', contentWithNewlines);
	const result2 = await sendRequest(
		'TEST B: Enum with newlines replaced by spaces',
		contentSanitized,
	);

	console.log('\n' + '='.repeat(60));
	console.log('RESULTS:');
	console.log(`  A (with \\n):     ${result1 ? 'ACCEPTED ✓' : 'REJECTED ✗'}`);
	console.log(`  B (sanitized):   ${result2 ? 'ACCEPTED ✓' : 'REJECTED ✗'}`);

	if (!result1 && result2) {
		console.log(
			'\n✓ PROVEN: The literal \\n in enum values is the sole cause of rejection.',
		);
		console.log(
			'  Replacing \\n with spaces makes the identical schema accepted.',
		);
		process.exit(0);
	} else if (result1 && result2) {
		console.log('\nBoth accepted — provider may have changed behavior.');
		process.exit(0);
	} else {
		console.log('\nUnexpected result combination.');
		process.exit(1);
	}
})();
