export const CHANNEL_COLORS = {
  tv: '#e74c3c',
  ooh: '#3498db',
  print: '#2ecc71',
  facebook: '#9b59b6',
  search: '#f39c12',
  base: '#95a5a6',
};

export const CHANNELS = ['tv', 'ooh', 'print', 'facebook', 'search'];

export const CHANNEL_LABELS = {
  tv: 'TV',
  ooh: 'OOH',
  print: 'Print',
  facebook: 'Facebook',
  search: 'Search',
  base: 'Base',
};

export const DEFAULT_SPENDS = {
  tv_spend: 5000,
  ooh_spend: 3000,
  print_spend: 2000,
  facebook_spend: 4000,
  search_spend: 3000,
};

export function formatEUR(val) {
  if (val == null) return '--';
  return `EUR ${Number(val).toLocaleString('en-GB', { maximumFractionDigits: 0 })}`;
}

export function formatPct(val, decimals = 1) {
  if (val == null) return '--';
  return `${Number(val).toFixed(decimals)}%`;
}
