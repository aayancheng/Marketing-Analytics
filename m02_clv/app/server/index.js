/**
 * Express proxy server for m02_clv (Customer Lifetime Value).
 *
 * Forwards /api/* requests to the FastAPI backend on port 8002.
 *
 * Start:
 *   cd m02_clv/app/server && npm start
 */

import express from 'express';
import cors from 'cors';
import { readFile } from 'fs/promises';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const PORT = process.env.PORT || 3002;
const FASTAPI_BASE = process.env.FASTAPI_BASE || 'http://localhost:8002';

const __dirname = dirname(fileURLToPath(import.meta.url));
const DOCS_DIR = join(__dirname, '..', '..', 'docs');

const app = express();
app.use(cors({ origin: 'http://localhost:5174' }));
app.use(express.json());

// ---------------------------------------------------------------------------
// Generic proxy helper
// ---------------------------------------------------------------------------

async function proxy(req, res, targetPath) {
    const queryString = req.url.includes('?') ? '?' + req.url.split('?')[1] : '';
    const url = `${FASTAPI_BASE}${targetPath}${queryString}`;
    const options = {
        method: req.method,
        headers: { 'Content-Type': 'application/json' },
    };
    if (req.method === 'POST') {
        options.body = JSON.stringify(req.body);
    }
    try {
        const resp = await fetch(url, options);
        const data = await resp.json();
        res.status(resp.status).json(data);
    } catch (err) {
        res.status(502).json({ error: 'upstream_error', message: err.message });
    }
}

// ---------------------------------------------------------------------------
// API routes
// ---------------------------------------------------------------------------

app.get('/health', (req, res) => proxy(req, res, '/health'));
app.get('/api/customers', (req, res) => proxy(req, res, '/api/customers'));
app.get('/api/customer/:id', (req, res) => proxy(req, res, `/api/customer/${req.params.id}`));
app.post('/api/predict', (req, res) => proxy(req, res, '/api/predict'));
app.get('/api/portfolio', (req, res) => proxy(req, res, '/api/portfolio'));
app.get('/api/segments', (req, res) => proxy(req, res, '/api/segments'));

// ---------------------------------------------------------------------------
// Markdown docs endpoint
// ---------------------------------------------------------------------------

const allowed = new Set(['model_card', 'data_dictionary', 'validation_report', 'research_brief']);

app.get('/api/docs/:docname', async (req, res) => {
    const name = req.params.docname;
    if (!allowed.has(name)) {
        return res.status(404).json({ error: 'not_found' });
    }
    try {
        const md = await readFile(join(DOCS_DIR, `${name}.md`), 'utf-8');
        res.json({ name, content: md });
    } catch {
        res.status(404).json({ error: 'not_found' });
    }
});

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

app.listen(PORT, '127.0.0.1', () => {
    console.log(`Express proxy on http://127.0.0.1:${PORT}`);
});
