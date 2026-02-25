import express from 'express';
import cors from 'cors';
import axios from 'axios';

const app = express();
const PORT = process.env.PORT || 3003;
const HOST = process.env.HOST || '127.0.0.1';
const FASTAPI_BASE = process.env.FASTAPI_BASE || 'http://localhost:8003';

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

// Customer endpoints
app.get('/api/customer/:id', (req, res) => proxy(req, res, `/api/customer/${req.params.id}`));
app.get('/api/customers', (req, res) => proxy(req, res, '/api/customers'));

// Prediction & simulation
app.post('/api/predict', (req, res) => proxy(req, res, '/api/predict'));
app.post('/api/simulate', (req, res) => proxy(req, res, '/api/simulate'));

// Segments overview
app.get('/api/segments', (req, res) => proxy(req, res, '/api/segments'));

app.listen(PORT, HOST, () => {
  console.log(`Churn analytics proxy listening on http://${HOST}:${PORT}`);
  console.log(`Forwarding API requests to ${FASTAPI_BASE}`);
});
