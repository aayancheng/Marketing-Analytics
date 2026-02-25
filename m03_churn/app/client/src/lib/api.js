import axios from 'axios';

const api = axios.create({ baseURL: 'http://localhost:3003' });

export async function fetchCustomer(customerId) {
  const resp = await api.get(`/api/customer/${customerId}`);
  return resp.data;
}

export async function fetchPrediction(params) {
  // Map frontend convenience keys to API snake_case convenience fields
  const { Contract, InternetService, PaymentMethod, ...rest } = params;
  const payload = {
    ...rest,
    ...(Contract      && { contract_type:    Contract }),
    ...(InternetService && { internet_service: InternetService }),
    ...(PaymentMethod && { payment_method:   PaymentMethod }),
  };
  const resp = await api.post('/api/predict', payload);
  return resp.data;
}

export async function fetchSimulation(customerId, changes) {
  const resp = await api.post('/api/simulate', { customerId, changes });
  return resp.data;
}

export async function fetchCustomers(page = 1, perPage = 100, segment = null) {
  const params = { page, per_page: perPage };
  if (segment) params.segment = segment;
  const resp = await api.get('/api/customers', { params });
  return resp.data;
}

export async function fetchAllCustomers(segment = '') {
  const first = await fetchCustomers(1, 100, segment || null);
  const totalPages = first.pages || 1;
  const customers = [...(first.items || [])];

  for (let page = 2; page <= totalPages; page++) {
    const resp = await fetchCustomers(page, 100, segment || null);
    customers.push(...(resp.items || []));
  }

  customers.sort((a, b) => String(a.customerID).localeCompare(String(b.customerID)));
  return customers;
}

export async function fetchSegments() {
  const resp = await api.get('/api/segments');
  return resp.data;
}
