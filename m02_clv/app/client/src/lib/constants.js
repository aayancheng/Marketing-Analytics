export const SEGMENTS = ['Champions', 'High Value', 'Growing', 'Occasional', 'Dormant'];

export const SEGMENT_COLORS = {
  'Champions': '#10b981',
  'High Value': '#3b82f6',
  'Growing': '#8b5cf6',
  'Occasional': '#f59e0b',
  'Dormant': '#ef4444',
};

export const SLIDER_CONFIG = {
  recency_days: { label: 'Recency (days)', min: 1, max: 365, step: 1, default: 30 },
  frequency: { label: 'Purchase Frequency', min: 1, max: 50, step: 1, default: 10 },
  monetary_total: { label: 'Total Spend (\u00a3)', min: 10, max: 50000, step: 100, default: 1000 },
  purchase_velocity: { label: 'Orders / Month', min: 0.1, max: 5.0, step: 0.1, default: 1.0 },
  cancellation_rate: { label: 'Cancellation Rate', min: 0, max: 0.5, step: 0.01, default: 0 },
};

export const DEFAULT_WHATIF = {
  recency_days: 30,
  frequency: 10,
  monetary_total: 1000,
  purchase_velocity: 1.0,
  cancellation_rate: 0,
};

export function formatCurrency(value) {
  if (value == null) return '\u00a30';
  return `\u00a3${Number(value).toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;
}

export function formatError(error) {
  const messages = {
    customer_not_found: 'Customer ID not found. Please check and try again.',
    insufficient_history: 'This customer has fewer than 2 purchases.',
    invalid_customer_id: 'Customer ID must be numeric.',
    cold_start: 'This customer has too few purchases for a reliable prediction.',
  };
  return messages[error?.error] || error?.message || 'Request failed.';
}
