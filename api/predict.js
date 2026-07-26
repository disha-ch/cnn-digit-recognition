export const config = {
  runtime: 'nodejs',
};

function json(res, status, body) {
  res.statusCode = status;
  res.setHeader('Content-Type', 'application/json');
  res.end(JSON.stringify(body));
}

function extractBase64(dataUrl) {
  if (!dataUrl) return null;
  return dataUrl.includes(',') ? dataUrl.split(',')[1] : dataUrl;
}

function buildPrompt() {
  return [
    'You are classifying a handwritten digit from 0 to 9.',
    'Look carefully at the image and return ONLY valid JSON.',
    'Schema: {"predictedDigit": number, "probabilities": [10 numbers that sum to 1]}',
    'Do not include markdown, code fences, or extra text.',
  ].join(' ');
}

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    json(res, 405, { error: 'Method not allowed' });
    return;
  }

  const apiKey = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
  if (!apiKey) {
    json(res, 500, {
      error:
        'Missing GEMINI_API_KEY. Add it in Vercel Environment Variables to enable digit predictions.',
    });
    return;
  }

  try {
    const { image } = req.body || {};
    const base64 = extractBase64(image);

    if (!base64) {
      json(res, 400, { error: 'Missing image data' });
      return;
    }

    const response = await fetch(
      'https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-goog-api-key': apiKey,
        },
        body: JSON.stringify({
          contents: [
            {
              role: 'user',
              parts: [
                { text: buildPrompt() },
                { inline_data: { mime_type: 'image/png', data: base64 } },
              ],
            },
          ],
          generationConfig: {
            temperature: 0.1,
            responseMimeType: 'application/json',
          },
        }),
      },
    );

    const data = await response.json();
    if (!response.ok) {
      json(res, 500, { error: data?.error?.message || 'Gemini request failed' });
      return;
    }

    const text =
      data?.candidates?.[0]?.content?.parts
        ?.map((part) => part.text || '')
        .join('')
        .trim() || '{}';

    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = {};
    }

    const probabilities = Array.isArray(parsed.probabilities) ? parsed.probabilities.slice(0, 10) : [];
    const normalized =
      probabilities.length === 10
        ? probabilities.map((value) => Number(value) || 0)
        : new Array(10).fill(0.1);

    const predictedDigit =
      Number.isInteger(parsed.predictedDigit) && parsed.predictedDigit >= 0 && parsed.predictedDigit <= 9
        ? parsed.predictedDigit
        : normalized.indexOf(Math.max(...normalized));

    json(res, 200, {
      predictedDigit,
      probabilities: normalized,
      source: 'gemini',
    });
  } catch (error) {
    json(res, 500, { error: error.message || 'Unexpected server error' });
  }
}
