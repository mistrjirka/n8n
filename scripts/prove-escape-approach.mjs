#!/usr/bin/env node
/**
 * Test three approaches for enum values with newlines in strict mode:
 * A) Raw \n  → expect REJECTED (provider error)
 * B) Replace \n with space → expect ACCEPTED (lossy)
 * C) Replace \n with \\n (literal backslash-n) → test if ACCEPTED (lossless)
 */

const apiKey = process.env.OPENROUTER_API_KEY;
const model = 'openai/gpt-5-mini';
const baseUrl = 'https://openrouter.ai/api/v1/chat/completions';

if (!apiKey) { console.error('Missing OPENROUTER_API_KEY'); process.exit(1); }

// Content with real newline characters
const raw = '🚨#Breaking:  \n🇺🇸🇮🇷 USS GERALD R. FORD ORDERED\nStay informed.';

// Approach B: spaces
const withSpaces = raw.replace(/[\n\r\t]/g, ' ');

// Approach C: escape to literal backslash sequences
const withEscapes = raw
	.replace(/\\/g, '\\\\')   // escape existing backslashes first
	.replace(/\n/g, '\\n')    // literal two-char sequence: \ n
	.replace(/\r/g, '\\r')
	.replace(/\t/g, '\\t');

console.log('Raw (JSON):     ', JSON.stringify(raw));
console.log('Spaces (JSON):  ', JSON.stringify(withSpaces));
console.log('Escaped (JSON): ', JSON.stringify(withEscapes));
console.log();

function buildPayload(enumValue) {
	return {
		model,
		messages: [{ role: 'user', content: `Categorize this news: ${withSpaces}` }],
		response_format: {
			type: 'json_schema',
			json_schema: {
				name: 'test',
				strict: true,
				schema: {
					type: 'object',
					properties: {
						content: { type: 'string', enum: [enumValue] },
						cat: { type: 'string', enum: ['news', 'other'] },
					},
					required: ['content', 'cat'],
					additionalProperties: false,
				},
			},
		},
		stream: false,
	};
}

async function test(label, enumValue) {
	console.log(`--- ${label} ---`);
	console.log('Enum value has literal \\n:', enumValue.includes('\n'));

	const res = await fetch(baseUrl, {
		method: 'POST',
		headers: {
			Authorization: `Bearer ${apiKey}`,
			'Content-Type': 'application/json',
		},
		body: JSON.stringify(buildPayload(enumValue)),
	});
	const json = await res.json();

	if (!res.ok) {
		let reason = json?.error?.message ?? 'unknown';
		try {
			reason = JSON.parse(json.error.metadata.raw).error.message;
		} catch {}
		console.log(`  REJECTED (${res.status}): ${reason}`);
		return { ok: false };
	}

	const content = json.choices[0].message.content;
	const parsed = JSON.parse(content);
	console.log('  ACCEPTED');
	console.log('  Model returned content:', JSON.stringify(parsed.content));

	// Try restoring
	const restored = parsed.content
		.replace(/\\n/g, '\n')
		.replace(/\\r/g, '\r')
		.replace(/\\t/g, '\t')
		.replace(/\\\\/g, '\\');
	const matchesOriginal = restored === raw;
	console.log('  Restored matches original:', matchesOriginal);
	if (!matchesOriginal) {
		console.log('  Restored:', JSON.stringify(restored));
		console.log('  Original:', JSON.stringify(raw));
	}
	return { ok: true, matchesOriginal, returned: parsed.content };
}

(async () => {
	const a = await test('A: Raw newlines', raw);
	console.log();
	const b = await test('B: Spaces (lossy)', withSpaces);
	console.log();
	const c = await test('C: Escaped \\\\n (lossless)', withEscapes);

	console.log('\n' + '='.repeat(50));
	console.log('SUMMARY:');
	console.log(`  A (raw \\n):      ${a.ok ? 'ACCEPTED' : 'REJECTED'}`);
	console.log(`  B (spaces):      ${b.ok ? 'ACCEPTED' : 'REJECTED'}${b.ok ? ' (lossy)' : ''}`);
	console.log(`  C (escaped \\\\n): ${c.ok ? 'ACCEPTED' : 'REJECTED'}${c.matchesOriginal ? ' (lossless roundtrip ✓)' : ''}`);
})();
