import axios from 'axios';

const api = axios.create({ baseURL: 'http://localhost:3002' });

export async function fetchCustomers(page = 1, perPage = 100, segment = null) {
  const params = { page, per_page: perPage };
  if (segment) params.segment = segment;
  const { data } = await api.get('/api/customers', { params });
  return data;
}

export async function fetchAllCustomers() {
  const first = await fetchCustomers(1, 100);
  const totalPages = first.pages;  // MUST be 'pages' — not 'total_pages'
  const customers = [...(first.items || [])];  // MUST be 'items' — not 'customers'

  const promises = [];
  for (let p = 2; p <= totalPages; p++) promises.push(fetchCustomers(p, 100));
  const rest = await Promise.all(promises);
  rest.forEach((r) => customers.push(...(r.items || [])));

  customers.sort((a, b) => a.customer_id - b.customer_id);
  return customers;
}

export async function fetchCustomer(id) {
  const { data } = await api.get(`/api/customer/${id}`);
  return data;
}

export async function fetchPrediction(params) {
  const { data } = await api.post('/api/predict', params);
  return data;
}

export async function fetchSegments() {
  const { data } = await api.get('/api/segments');
  return data;
}

export async function fetchPortfolio() {
  const { data } = await api.get('/api/portfolio');
  return data;
}
