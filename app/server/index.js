import express from 'express';
import cors from 'cors';
import axios from 'axios';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const app = express();
const PORT = process.env.PORT || 3001;
const HOST = process.env.HOST || '127.0.0.1';
const FASTAPI_BASE = process.env.FASTAPI_BASE || 'http://localhost:8000';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DOCS_DIR = path.resolve(__dirname, '..', '..', 'docs');

app.use(cors({ origin: 'http://localhost:5173' }));
app.use(express.json());

async function proxy(req, res, targetPath) {
  try {
    const resp = await axios({
      method: req.method,
      url: `${FASTAPI_BASE}${targetPath}`,
      params: req.query,
      data: req.body,
      validateStatus: () => true,
    });
    res.status(resp.status).json(resp.data);
  } catch (err) {
    res.status(502).json({ error: 'upstream_unavailable', message: err.message });
  }
}

app.get('/api/customer/:id', (req, res) => proxy(req, res, `/api/customer/${req.params.id}`));
app.post('/api/predict', (req, res) => proxy(req, res, '/api/predict'));
app.get('/api/customers', (req, res) => proxy(req, res, '/api/customers'));
app.get('/api/segments', (req, res) => proxy(req, res, '/api/segments'));

app.get('/api/docs/:docname', async (req, res) => {
  const allowed = new Set(['model_card', 'data_dictionary', 'synthesis_methodology', 'validation_report']);
  const docname = req.params.docname;
  if (!allowed.has(docname)) {
    return res.status(404).json({ error: 'doc_not_found', message: 'Document not found.' });
  }

  const filePath = path.join(DOCS_DIR, `${docname}.md`);
  try {
    const content = await fs.readFile(filePath, 'utf8');
    res.json({ content });
  } catch {
    res.status(404).json({ error: 'doc_not_found', message: 'Document not found.' });
  }
});

app.listen(PORT, HOST, () => {
  console.log(`Listening on http://${HOST}:${PORT}`);
});
