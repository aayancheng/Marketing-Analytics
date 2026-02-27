export const DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

export const SEGMENTS = ['Champions', 'Loyal', 'At Risk', 'Hibernating'];

export const SLIDER_CONFIG = {
  recency_days: { label: 'Recency (days)', min: 1, max: 365, step: 1, default: 30 },
  frequency: { label: 'Frequency', min: 1, max: 100, step: 1, default: 10 },
  modal_hour: { label: 'Modal Hour', min: 0, max: 23, step: 1, default: 10 },
  purchase_hour_entropy: { label: 'Entropy', min: 0, max: 1, step: 0.01, default: 0.3 },
};

export const DEFAULT_WHATIF = {
  recency_days: 30,
  frequency: 10,
  modal_hour: 10,
  purchase_hour_entropy: 0.3,
};

export function interpolateHeatColor(value) {
  // indigo-900 (#312e81) → violet-500 (#8b5cf6) → pink-400 (#f472b6)
  const clamp = Math.max(0, Math.min(1, value));
  if (clamp < 0.5) {
    const t = clamp * 2;
    const r = Math.round(49 + t * (139 - 49));
    const g = Math.round(46 + t * (92 - 46));
    const b = Math.round(129 + t * (246 - 129));
    return `rgb(${r}, ${g}, ${b})`;
  }
  const t = (clamp - 0.5) * 2;
  const r = Math.round(139 + t * (244 - 139));
  const g = Math.round(92 + t * (114 - 92));
  const b = Math.round(246 + t * (182 - 246));
  return `rgb(${r}, ${g}, ${b})`;
}

export function formatError(error) {
  const messages = {
    customer_not_found: 'Customer ID not found. Please check and try again.',
    insufficient_history: 'This customer has fewer than 5 transactions.',
    invalid_customer_id: 'Customer ID must be numeric.',
  };
  return messages[error?.error] || error?.message || 'Request failed.';
}
