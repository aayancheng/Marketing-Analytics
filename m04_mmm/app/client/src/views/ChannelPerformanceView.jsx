import {
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ErrorBar,
  LineChart,
  Line,
} from 'recharts';
import Card from '../components/Card';
import ChannelLegend from '../components/ChannelLegend';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorBanner from '../components/ErrorBanner';
import { useChannelPerformance } from '../lib/hooks';
import { CHANNEL_COLORS, CHANNEL_LABELS, formatEUR } from '../lib/constants';

export default function ChannelPerformanceView() {
  const { roas, curves, adstockData, loading, error } = useChannelPerformance();

  if (loading) return <LoadingSpinner message="Loading channel performance..." />;
  if (error) return <ErrorBanner message={error} />;
  if (!roas) return null;

  // Prepare ROAS bar data with error ranges
  const roasData = roas.channels.map((ch) => ({
    channel: CHANNEL_LABELS[ch.channel] || ch.channel,
    channelKey: ch.channel,
    roas: Number(ch.roas_mean.toFixed(2)),
    errorLow: Number((ch.roas_mean - ch.roas_hdi_3).toFixed(2)),
    errorHigh: Number((ch.roas_hdi_97 - ch.roas_mean).toFixed(2)),
    spend: ch.total_spend,
    contribution: ch.total_contribution,
  }));

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-slate-800">Channel Performance</h2>

      {/* ROAS bar chart */}
      <Card wide>
        <h3 className="text-sm font-semibold text-slate-700 mb-4">
          Return on Ad Spend (ROAS) by Channel
        </h3>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={roasData} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="channel" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 11 }} label={{ value: 'ROAS', angle: -90, position: 'insideLeft', fontSize: 12 }} />
            <Tooltip
              formatter={(val, name) => {
                if (name === 'roas') return [val.toFixed(2), 'ROAS'];
                return [val, name];
              }}
              content={({ active, payload }) => {
                if (!active || !payload?.[0]) return null;
                const d = payload[0].payload;
                return (
                  <div className="bg-white shadow-lg rounded-lg p-3 border text-sm">
                    <p className="font-semibold">{d.channel}</p>
                    <p>ROAS: {d.roas.toFixed(2)}</p>
                    <p>Spend: {formatEUR(d.spend)}</p>
                    <p>Contribution: {formatEUR(d.contribution)}</p>
                  </div>
                );
              }}
            />
            <Bar dataKey="roas" radius={[6, 6, 0, 0]}>
              {roasData.map((entry, i) => (
                <Cell
                  key={i}
                  fill={CHANNEL_COLORS[entry.channelKey] || '#8b5cf6'}
                />
              ))}
              <ErrorBar dataKey="errorHigh" width={6} strokeWidth={2} stroke="#475569" direction="y" />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <ChannelLegend />
      </Card>

      {/* Response curves grid */}
      {curves && (
        <Card wide>
          <h3 className="text-sm font-semibold text-slate-700 mb-4">Response Curves</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {curves.channels.map((ch) => (
              <div key={ch.channel} className="border rounded-xl p-3">
                <p className="text-xs font-semibold text-slate-600 mb-2 flex items-center gap-2">
                  <span
                    className="w-3 h-3 rounded-full inline-block"
                    style={{ backgroundColor: CHANNEL_COLORS[ch.channel] }}
                  />
                  {CHANNEL_LABELS[ch.channel] || ch.channel}
                </p>
                <ResponsiveContainer width="100%" height={160}>
                  <LineChart data={ch.curve} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                    <XAxis
                      dataKey="spend"
                      tick={{ fontSize: 9 }}
                      tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`}
                    />
                    <YAxis tick={{ fontSize: 9 }} tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`} />
                    <Tooltip
                      formatter={(val) => [formatEUR(val), 'Contribution']}
                      labelFormatter={(v) => `Spend: ${formatEUR(v)}`}
                    />
                    <Line
                      type="monotone"
                      dataKey="contribution"
                      stroke={CHANNEL_COLORS[ch.channel]}
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Adstock decay grid */}
      {adstockData && (
        <Card wide>
          <h3 className="text-sm font-semibold text-slate-700 mb-4">Adstock Decay Patterns</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {adstockData.channels.map((ch) => {
              const decayData = ch.decay_vector.map((w, i) => ({
                week: `W${i}`,
                weight: Number(w.toFixed(3)),
              }));
              return (
                <div key={ch.channel} className="border rounded-xl p-3">
                  <p className="text-xs font-semibold text-slate-600 mb-1 flex items-center gap-2">
                    <span
                      className="w-3 h-3 rounded-full inline-block"
                      style={{ backgroundColor: CHANNEL_COLORS[ch.channel] }}
                    />
                    {CHANNEL_LABELS[ch.channel] || ch.channel}
                  </p>
                  <p className="text-[10px] text-slate-400 mb-2">
                    alpha: {ch.alpha_mean.toFixed(3)} [{ch.alpha_hdi_3.toFixed(3)} - {ch.alpha_hdi_97.toFixed(3)}]
                  </p>
                  <ResponsiveContainer width="100%" height={120}>
                    <BarChart data={decayData} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
                      <XAxis dataKey="week" tick={{ fontSize: 9 }} />
                      <YAxis tick={{ fontSize: 9 }} />
                      <Tooltip formatter={(val) => [val.toFixed(3), 'Weight']} />
                      <Bar dataKey="weight" fill={CHANNEL_COLORS[ch.channel]} radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              );
            })}
          </div>
        </Card>
      )}
    </div>
  );
}
