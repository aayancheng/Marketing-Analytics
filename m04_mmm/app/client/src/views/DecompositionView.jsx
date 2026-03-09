import { DollarSign, TrendingUp, BarChart3, Activity } from 'lucide-react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from 'recharts';
import Card from '../components/Card';
import StatCard from '../components/StatCard';
import ChannelLegend from '../components/ChannelLegend';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorBanner from '../components/ErrorBanner';
import { useDecomposition } from '../lib/hooks';
import { CHANNEL_COLORS, CHANNEL_LABELS, CHANNELS, formatEUR, formatPct } from '../lib/constants';

const STACK_KEYS = ['base', ...CHANNELS];

export default function DecompositionView() {
  const { data, loading, error } = useDecomposition();

  if (loading) return <LoadingSpinner message="Loading decomposition..." />;
  if (error) return <ErrorBanner message={error} />;
  if (!data) return null;

  const { weeks, totals, pct } = data;

  const totalRevenue = weeks.reduce((s, w) => s + (w.revenue_actual || 0), 0);
  const mediaContrib = CHANNELS.reduce((s, ch) => s + (totals[ch] || 0), 0);
  const mediaPct = totalRevenue > 0 ? (mediaContrib / totalRevenue) * 100 : 0;
  const topChannel = CHANNELS.reduce((best, ch) =>
    (totals[ch] || 0) > (totals[best] || 0) ? ch : best
  , CHANNELS[0]);
  const avgWeekly = weeks.length > 0 ? totalRevenue / weeks.length : 0;

  // Pie data from pct
  const pieData = Object.entries(pct || {})
    .filter(([k]) => k !== 'base')
    .map(([ch, val]) => ({
      name: CHANNEL_LABELS[ch] || ch,
      value: Number((val * 100).toFixed(1)),
      color: CHANNEL_COLORS[ch],
    }));

  // Add base to pie if available
  if (pct?.base != null) {
    pieData.unshift({
      name: 'Base',
      value: Number((pct.base * 100).toFixed(1)),
      color: CHANNEL_COLORS.base,
    });
  }

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-slate-800">Revenue Decomposition</h2>

      {/* KPI cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard icon={DollarSign} label="Total Revenue" value={formatEUR(totalRevenue)} color="brand" />
        <StatCard icon={TrendingUp} label="Media Contribution" value={formatPct(mediaPct)} color="accent" />
        <StatCard icon={BarChart3} label="Top Channel" value={CHANNEL_LABELS[topChannel]} color="emerald" />
        <StatCard icon={Activity} label="Avg Weekly Revenue" value={formatEUR(avgWeekly)} color="amber" />
      </div>

      {/* Stacked area chart */}
      <Card wide>
        <h3 className="text-sm font-semibold text-slate-700 mb-4">Weekly Revenue by Source</h3>
        <ResponsiveContainer width="100%" height={380}>
          <AreaChart data={weeks} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="date_week"
              tick={{ fontSize: 11 }}
              tickFormatter={(v) => {
                const d = new Date(v);
                const month = d.toLocaleString('en-GB', { month: 'short' });
                return month;
              }}
              interval="preserveStartEnd"
            />
            <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`} />
            <Tooltip
              formatter={(val, name) => [formatEUR(val), CHANNEL_LABELS[name] || name]}
              labelFormatter={(v) => new Date(v).toLocaleDateString('en-GB')}
            />
            {STACK_KEYS.map((key) => (
              <Area
                key={key}
                type="monotone"
                dataKey={key}
                stackId="1"
                fill={CHANNEL_COLORS[key]}
                stroke={CHANNEL_COLORS[key]}
                fillOpacity={0.85}
              />
            ))}
          </AreaChart>
        </ResponsiveContainer>
        <ChannelLegend showBase />
      </Card>

      {/* Pie chart */}
      <Card>
        <h3 className="text-sm font-semibold text-slate-700 mb-4">Total Attribution by Source</h3>
        <ResponsiveContainer width="100%" height={320}>
          <PieChart>
            <Pie
              data={pieData}
              dataKey="value"
              nameKey="name"
              cx="50%"
              cy="50%"
              outerRadius={110}
              label={({ name, value }) => `${name} ${value}%`}
            >
              {pieData.map((entry, i) => (
                <Cell key={i} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip formatter={(val) => `${val}%`} />
          </PieChart>
        </ResponsiveContainer>
      </Card>
    </div>
  );
}
