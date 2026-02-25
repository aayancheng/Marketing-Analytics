import { RISK_COLORS } from '../lib/constants';

const DEFAULT_COLUMNS = [
  { key: 'customerID', label: 'Customer ID' },
  {
    key: 'churn_probability',
    label: 'Churn Risk',
    render: (v) => {
      if (v == null) return '—';
      const pct = (v * 100).toFixed(1);
      return <span className="font-semibold">{pct}%</span>;
    },
  },
  {
    key: 'risk_tier',
    label: 'Risk Tier',
    render: (v) => {
      if (!v) return '—';
      const color = RISK_COLORS[v] || '#94a3b8';
      return (
        <span
          className="inline-block px-2 py-0.5 rounded-full text-xs font-semibold text-white"
          style={{ backgroundColor: color }}
        >
          {v}
        </span>
      );
    },
  },
  {
    key: 'tenure',
    label: 'Tenure',
    render: (v) => (v != null ? `${v}mo` : '—'),
  },
  {
    key: 'MonthlyCharges',
    label: 'Monthly ($)',
    render: (v) => (v != null ? `$${Number(v).toFixed(2)}` : '—'),
  },
];

export default function DataTable({ rows, columns }) {
  const cols = columns || DEFAULT_COLUMNS;
  if (!rows || rows.length === 0)
    return <p className="text-sm text-slate-400 py-4">No data available.</p>;

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-200">
            {cols.map((col) => (
              <th
                key={col.key}
                className="text-left py-2.5 px-3 text-xs font-semibold text-slate-500 uppercase tracking-wide"
              >
                {col.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={i}
              className={`border-b border-slate-100 ${i % 2 === 1 ? 'bg-slate-50/50' : ''}`}
            >
              {cols.map((col) => (
                <td key={col.key} className="py-2.5 px-3 text-slate-700">
                  {col.render ? col.render(row[col.key], row) : row[col.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
