import { DollarSign, TrendingUp, AlertTriangle, Lock, Unlock } from 'lucide-react';
import Card from '../components/Card';
import StatCard from '../components/StatCard';
import SliderControl from '../components/SliderControl';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorBanner from '../components/ErrorBanner';
import { useSimulator } from '../lib/hooks';
import { CHANNEL_LABELS, formatEUR, formatPct } from '../lib/constants';

const SLIDER_CONFIG = [
  { key: 'tv_spend', label: 'TV' },
  { key: 'ooh_spend', label: 'OOH' },
  { key: 'print_spend', label: 'Print' },
  { key: 'facebook_spend', label: 'Facebook' },
  { key: 'search_spend', label: 'Search' },
];

export default function BudgetSimulatorView() {
  const {
    spends,
    result,
    loading,
    initialLoading,
    error,
    budgetLock,
    setBudgetLock,
    updateSpend,
    totalBudget,
  } = useSimulator();

  if (initialLoading) return <LoadingSpinner message="Loading budget data..." />;
  if (error) return <ErrorBanner message={error} />;
  if (!spends) return null;

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-slate-800">Budget Simulator</h2>

      {/* Total budget + lock toggle */}
      <Card>
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div>
            <p className="text-xs font-medium uppercase tracking-wide text-slate-500">Total Weekly Budget</p>
            <p className="text-2xl font-bold text-slate-800">{formatEUR(totalBudget)}</p>
          </div>
          <label className="flex items-center gap-2 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={budgetLock}
              onChange={(e) => setBudgetLock(e.target.checked)}
              className="w-4 h-4 accent-brand-600 rounded"
            />
            <span className="flex items-center gap-1.5 text-sm text-slate-600">
              {budgetLock ? <Lock size={14} /> : <Unlock size={14} />}
              Lock total budget
            </span>
          </label>
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Sliders */}
        <Card>
          <h3 className="text-sm font-semibold text-slate-700 mb-4">Channel Budgets</h3>
          <div className="space-y-5">
            {SLIDER_CONFIG.map(({ key, label }) => {
              const current = spends[key] || 0;
              return (
                <SliderControl
                  key={key}
                  label={label}
                  value={current}
                  min={0}
                  max={Math.max(current * 2, 1000)}
                  step={100}
                  onChange={(v) => updateSpend(key, v)}
                />
              );
            })}
          </div>
        </Card>

        {/* Results */}
        <div className="space-y-4">
          {/* Revenue prediction */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <StatCard
              icon={DollarSign}
              label="Predicted Revenue"
              value={result ? formatEUR(result.predicted_revenue) : '--'}
              color="brand"
            />
            <StatCard
              icon={TrendingUp}
              label="Delta vs Current"
              value={
                result
                  ? `${result.delta >= 0 ? '+' : ''}${formatEUR(result.delta)} (${result.delta_pct >= 0 ? '+' : ''}${formatPct(result.delta_pct)})`
                  : '--'
              }
              color={result && result.delta >= 0 ? 'emerald' : 'red'}
            />
          </div>

          {/* Channel contributions */}
          {result?.channel_contributions && (
            <Card>
              <h3 className="text-sm font-semibold text-slate-700 mb-3">Channel Contributions</h3>
              <div className="space-y-2">
                {Object.entries(result.channel_contributions).map(([ch, val]) => {
                  const total = Object.values(result.channel_contributions).reduce((s, v) => s + v, 0);
                  const pct = total > 0 ? (val / total) * 100 : 0;
                  return (
                    <div key={ch} className="flex items-center gap-3">
                      <span className="w-16 text-xs font-medium text-slate-600">
                        {CHANNEL_LABELS[ch] || ch}
                      </span>
                      <div className="flex-1 bg-slate-100 rounded-full h-4 overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-300"
                          style={{
                            width: `${Math.min(pct, 100)}%`,
                            backgroundColor: `var(--ch-${ch})`,
                          }}
                        />
                      </div>
                      <span className="text-xs text-slate-500 w-20 text-right">{formatEUR(val)}</span>
                    </div>
                  );
                })}
              </div>
            </Card>
          )}

          {/* Saturation warnings */}
          {result?.saturation_warnings?.length > 0 && (
            <Card>
              <h3 className="text-sm font-semibold text-slate-700 mb-3 flex items-center gap-2">
                <AlertTriangle size={16} className="text-amber-500" />
                Saturation Warnings
              </h3>
              <div className="flex flex-wrap gap-2">
                {result.saturation_warnings.map((w, i) => (
                  <span
                    key={i}
                    className={`text-xs font-medium px-3 py-1.5 rounded-full ${
                      w.level === 'high'
                        ? 'bg-red-100 text-red-700'
                        : 'bg-amber-100 text-amber-700'
                    }`}
                  >
                    {CHANNEL_LABELS[w.channel] || w.channel}: {w.level}
                  </span>
                ))}
              </div>
            </Card>
          )}

          {loading && (
            <p className="text-xs text-slate-400 text-center">Recalculating...</p>
          )}
        </div>
      </div>
    </div>
  );
}
