import { TrendingUp, ArrowUpRight, ArrowDownRight, Minus } from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import Card from '../components/Card';
import StatCard from '../components/StatCard';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorBanner from '../components/ErrorBanner';
import { useOptimalAllocation } from '../lib/hooks';
import { CHANNEL_LABELS, CHANNELS, formatEUR, formatPct } from '../lib/constants';

export default function OptimalAllocationView() {
  const { data, loading, error } = useOptimalAllocation();

  if (loading) return <LoadingSpinner message="Loading optimal allocation..." />;
  if (error) return <ErrorBanner message={error} />;
  if (!data) return null;

  const { current, optimal, current_revenue, optimal_revenue, lift_abs, lift_pct, recommendations } = data;

  // Grouped bar data
  const barData = CHANNELS
    .filter((ch) => current[ch] != null || optimal[ch] != null)
    .map((ch) => ({
      channel: CHANNEL_LABELS[ch] || ch,
      Current: current[ch] || 0,
      Optimal: optimal[ch] || 0,
    }));

  const actionIcon = (action) => {
    if (action === 'increase') return <ArrowUpRight size={14} className="text-emerald-600" />;
    if (action === 'decrease') return <ArrowDownRight size={14} className="text-red-500" />;
    return <Minus size={14} className="text-slate-400" />;
  };

  const actionColor = (action) => {
    if (action === 'increase') return 'text-emerald-600';
    if (action === 'decrease') return 'text-red-500';
    return 'text-slate-500';
  };

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-slate-800">Optimal Allocation</h2>

      {/* Lift card */}
      <Card wide>
        <div className="flex items-center gap-4 flex-wrap">
          <div className="flex items-center gap-3 bg-emerald-50 border border-emerald-200 rounded-xl px-5 py-4 flex-1 min-w-[280px]">
            <TrendingUp size={28} className="text-emerald-600" />
            <div>
              <p className="text-sm text-emerald-700 font-medium">
                Optimal allocation could lift weekly revenue by
              </p>
              <p className="text-2xl font-bold text-emerald-800">
                {formatEUR(lift_abs)}{' '}
                <span className="text-lg font-semibold">
                  (+{formatPct(lift_pct)})
                </span>
              </p>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3 flex-1 min-w-[200px]">
            <StatCard label="Current Revenue" value={formatEUR(current_revenue)} color="amber" />
            <StatCard label="Optimal Revenue" value={formatEUR(optimal_revenue)} color="emerald" />
          </div>
        </div>
      </Card>

      {/* Grouped bar chart */}
      <Card wide>
        <h3 className="text-sm font-semibold text-slate-700 mb-4">Current vs Optimal Spend</h3>
        <ResponsiveContainer width="100%" height={340}>
          <BarChart data={barData} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="channel" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`} />
            <Tooltip formatter={(val) => formatEUR(val)} />
            <Legend />
            <Bar dataKey="Current" fill="#94a3b8" radius={[4, 4, 0, 0]} />
            <Bar dataKey="Optimal" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Recommendations table */}
      {recommendations && recommendations.length > 0 && (
        <Card wide>
          <h3 className="text-sm font-semibold text-slate-700 mb-4">Recommendations</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200">
                  <th className="text-left py-2 px-3 text-xs font-semibold text-slate-500 uppercase">Channel</th>
                  <th className="text-right py-2 px-3 text-xs font-semibold text-slate-500 uppercase">Current</th>
                  <th className="text-right py-2 px-3 text-xs font-semibold text-slate-500 uppercase">Optimal</th>
                  <th className="text-right py-2 px-3 text-xs font-semibold text-slate-500 uppercase">Delta</th>
                  <th className="text-right py-2 px-3 text-xs font-semibold text-slate-500 uppercase">Change</th>
                  <th className="text-center py-2 px-3 text-xs font-semibold text-slate-500 uppercase">Action</th>
                </tr>
              </thead>
              <tbody>
                {recommendations.map((rec) => (
                  <tr key={rec.channel} className="border-b border-slate-100 hover:bg-slate-50">
                    <td className="py-2.5 px-3 font-medium text-slate-700">
                      {CHANNEL_LABELS[rec.channel] || rec.channel}
                    </td>
                    <td className="py-2.5 px-3 text-right text-slate-600">{formatEUR(rec.current)}</td>
                    <td className="py-2.5 px-3 text-right text-slate-600">{formatEUR(rec.optimal)}</td>
                    <td className={`py-2.5 px-3 text-right font-medium ${actionColor(rec.action)}`}>
                      {rec.delta >= 0 ? '+' : ''}{formatEUR(rec.delta)}
                    </td>
                    <td className={`py-2.5 px-3 text-right font-medium ${actionColor(rec.action)}`}>
                      {rec.delta_pct >= 0 ? '+' : ''}{formatPct(rec.delta_pct)}
                    </td>
                    <td className="py-2.5 px-3 text-center">
                      <span className="inline-flex items-center gap-1">
                        {actionIcon(rec.action)}
                        <span className={`text-xs font-medium capitalize ${actionColor(rec.action)}`}>
                          {rec.action}
                        </span>
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}
