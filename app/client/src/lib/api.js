import axios from 'axios';

const api = axios.create({ baseURL: 'http://localhost:3001' });

export async function fetchCustomer(id) {
  const resp = await api.get(`/api/customer/${id}`);
  return resp.data;
}

export async function fetchPrediction(params) {
  const resp = await api.post('/api/predict', params);
  return resp.data;
}

export async function fetchCustomers(page = 1, perPage = 100, segment = null) {
  const params = { page, per_page: perPage };
  if (segment) params.segment = segment;
  const resp = await api.get('/api/customers', { params });
  return resp.data;
}

export async function fetchAllCustomers() {
  const first = await fetchCustomers(1, 100);
  const totalPages = first.total_pages || 1;
  const customers = [...(first.customers || [])];

  for (let page = 2; page <= totalPages; page++) {
    const resp = await fetchCustomers(page, 100);
    customers.push(...(resp.customers || []));
  }

  customers.sort((a, b) => a.customer_id - b.customer_id);
  return customers;
}

export async function fetchSegments() {
  const resp = await api.get('/api/segments');
  return resp.data;
}
