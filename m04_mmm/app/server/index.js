import express from 'express';
import cors from 'cors';
import axios from 'axios';

const app = express();
const PORT = process.env.PORT || 3004;
const HOST = process.env.HOST || '127.0.0.1';
const FASTAPI_BASE = process.env.FASTAPI_BASE || 'http://localhost:8004';

app.use(cors({ origin: '*' }));
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

// Health check
app.get('/health', (req, res) => proxy(req, res, '/health'));

// Pre-computed data endpoints
app.get('/api/decomposition', (req, res) => proxy(req, res, '/api/decomposition'));
app.get('/api/roas', (req, res) => proxy(req, res, '/api/roas'));
app.get('/api/response-curves', (req, res) => proxy(req, res, '/api/response-curves'));
app.get('/api/adstock', (req, res) => proxy(req, res, '/api/adstock'));
app.get('/api/optimal-allocation', (req, res) => proxy(req, res, '/api/optimal-allocation'));
app.get('/api/metadata', (req, res) => proxy(req, res, '/api/metadata'));

// Simulator
app.post('/api/simulate', (req, res) => proxy(req, res, '/api/simulate'));

app.listen(PORT, HOST, () => {
  console.log(`MMM analytics proxy listening on http://${HOST}:${PORT}`);
  console.log(`Forwarding API requests to ${FASTAPI_BASE}`);
});
