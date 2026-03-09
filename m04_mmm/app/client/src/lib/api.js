import axios from 'axios';

const api = axios.create({ baseURL: 'http://localhost:3004' });

export async function fetchDecomposition() {
  const { data } = await api.get('/api/decomposition');
  return data;
}

export async function fetchRoas() {
  const { data } = await api.get('/api/roas');
  return data;
}

export async function fetchResponseCurves() {
  const { data } = await api.get('/api/response-curves');
  return data;
}

export async function fetchAdstock() {
  const { data } = await api.get('/api/adstock');
  return data;
}

export async function fetchOptimalAllocation() {
  const { data } = await api.get('/api/optimal-allocation');
  return data;
}

export async function simulateBudget(spends) {
  const { data } = await api.post('/api/simulate', spends);
  return data;
}
