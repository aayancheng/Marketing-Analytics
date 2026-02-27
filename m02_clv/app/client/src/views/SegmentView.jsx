import { useMemo } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Users, DollarSign, TrendingUp } from 'lucide-react';
import Card from '../components/Card';
import StatCard from '../components/StatCard';
import SegmentPicker from '../components/SegmentPicker';
import DataTable from '../components/DataTable';
import LoadingSpinner from '../components/LoadingSpinner';
import { formatCurrency, SEGMENT_COLORS } from '../lib/constants';

const TABLE_COLUMNS = [
  { key: 'customer_id', label: 'ID' },
  { key: 'predicted_clv', label: 'Predicted CLV', render: (v) => formatCurrency(v) },
  { key: 'recency_days', label: 'Recency', render: (v) => `${(v ?? 0).toFixed(0)}d` },
  { key: 'frequency', label: 'Frequency' },
  { key: 'clv_segment', label: 'Segment' },
];

export default function SegmentView({ hook }) {
  const { segment, changeSegment, customers, summary, portfolio, loading } = hook;

  const currentSummary = summary.find((s) => s.segment === segment);

  // Scatter data: recency_days (x) vs predicted_clv (y) from portfolio
  const scatterData = useMemo(() => {
    const source = portfolio.length > 0 ? portfolio : customers;
    return source
      .filter((r) => r.clv_segment === segment || portfolio.length > 0)
      .map((r) => ({
        x: r.recency_days ?? 0,
        y: r.predicted_clv ?? 0,
        id: r.customer_id,
        segment: r.clv_segment,
      }));
  }, [portfolio, customers, segment]);

  // Group scatter by segment for color coding
  const segmentGroups = useMemo(() => {
    const groups = {};
    scatterData.forEach((d) => {
      const seg = d.segment || 'Unknown';
      if (!groups[seg]) groups[seg] = [];
      groups[seg].push(d);
    });
    return groups;
  }, [scatterData]);

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h2 className="text-2xl font-bold text-slate-800 mb-1">Portfolio Explorer</h2>
          <p className="text-sm text-slate-500">Explore CLV segments and customer distribution</p>
        </div>
        <SegmentPicker value={segment} onChange={changeSegment} />
      </div>

      {currentSummary && (
        <div className="grid grid-cols-3 gap-3">
          <StatCard icon={Users} label="Customers" value={currentSummary.count} color="brand" />
          <StatCard
            icon={DollarSign}
            label="Avg CLV"
            value={formatCurrency(currentSummary.mean_clv ?? currentSummary.avg_clv)}
            color="emerald"
          />
          <StatCard
            icon={TrendingUp}
            label="Total CLV"
            value={formatCurrency(currentSummary.total_clv ?? currentSummary.sum_clv)}
            color="accent"
          />
        </div>
      )}

      {loading ? (
        <LoadingSpinner />
      ) : (
        <>
          <Card>
            <h3 className="text-lg font-semibold text-slate-700 mb-3">Recency vs Predicted CLV</h3>
            <p className="text-xs text-slate-400 mb-3">Each dot represents a customer; color = CLV segment</p>
            <div style={{ height: 350 }}>
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ left: 10, right: 10, top: 10, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis
                    type="number"
                    dataKey="x"
                    name="Recency (days)"
                    tick={{ fontSize: 11 }}
                    label={{ value: 'Recency (days)', position: 'insideBottom', offset: -5, fontSize: 12 }}
                  />
                  <YAxis
                    type="number"
                    dataKey="y"
                    name="Predicted CLV"
                    tick={{ fontSize: 11 }}
                    label={{ value: 'Predicted CLV (\u00a3)', angle: -90, position: 'insideLeft', fontSize: 12 }}
                  />
                  <Tooltip
                    contentStyle={{ borderRadius: '8px', fontSize: '12px' }}
                    formatter={(val, name) => {
                      if (name === 'Predicted CLV') return formatCurrency(val);
                      return typeof val === 'number' ? val.toFixed(0) : val;
                    }}
                    labelFormatter={() => ''}
                  />
                  {Object.entries(segmentGroups).map(([seg, points]) => (
                    <Scatter
                      key={seg}
                      name={seg}
                      data={points}
                      fill={SEGMENT_COLORS[seg] || '#7c3aed'}
                      fillOpacity={0.6}
                    />
                  ))}
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <Card>
            <h3 className="text-lg font-semibold text-slate-700 mb-3">
              {segment} Customers
            </h3>
            <DataTable rows={customers.slice(0, 20)} columns={TABLE_COLUMNS} />
          </Card>
        </>
      )}
    </div>
  );
}
