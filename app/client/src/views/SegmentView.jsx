import { useMemo } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Users, TrendingUp, Clock } from 'lucide-react';
import Card from '../components/Card';
import StatCard from '../components/StatCard';
import SegmentPicker from '../components/SegmentPicker';
import DataTable from '../components/DataTable';
import LoadingSpinner from '../components/LoadingSpinner';

const TABLE_COLUMNS = [
  { key: 'customer_id', label: 'ID' },
  { key: 'modal_purchase_hour', label: 'Modal Hour' },
  { key: 'open_rate', label: 'Open Rate', render: (v) => `${((v ?? 0) * 100).toFixed(1)}%` },
  { key: 'recency_days', label: 'Recency', render: (v) => `${v.toFixed(0)}d` },
];

export default function SegmentView({ hook }) {
  const { segment, changeSegment, customers, summary, loading } = hook;

  const currentSummary = summary.find((s) => s.segment === segment);

  const scatterData = useMemo(
    () => customers.map((r) => ({ x: r.modal_purchase_hour, y: r.open_rate ?? 0, id: r.customer_id })),
    [customers]
  );

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-slate-800 mb-1">Segment Explorer</h2>
          <p className="text-sm text-slate-500">Explore RFM segments and customer patterns</p>
        </div>
        <SegmentPicker value={segment} onChange={changeSegment} />
      </div>

      {currentSummary && (
        <div className="grid grid-cols-3 gap-3">
          <StatCard icon={Users} label="Customers" value={currentSummary.count} color="brand" />
          <StatCard icon={TrendingUp} label="Avg Open Rate" value={`${(currentSummary.mean_open_rate * 100).toFixed(1)}%`} color="accent" />
          <StatCard icon={Clock} label="Avg Recency" value={`${currentSummary.mean_recency_days.toFixed(0)}d`} color="emerald" />
        </div>
      )}

      {loading ? (
        <LoadingSpinner />
      ) : (
        <>
          <Card>
            <h3 className="text-lg font-semibold text-slate-700 mb-3">Modal Hour vs Open Rate</h3>
            <div style={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ left: 10, right: 10, top: 10, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis type="number" dataKey="x" name="Modal Hour" domain={[0, 23]} tick={{ fontSize: 11 }} />
                  <YAxis type="number" dataKey="y" name="Open Rate" domain={[0, 1]} tick={{ fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ borderRadius: '8px', fontSize: '12px' }}
                    formatter={(val) => typeof val === 'number' ? val.toFixed(2) : val}
                  />
                  <Scatter data={scatterData} fill="#7c3aed" fillOpacity={0.6} />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <Card>
            <h3 className="text-lg font-semibold text-slate-700 mb-3">Top 10 Customers</h3>
            <DataTable rows={customers.slice(0, 10)} columns={TABLE_COLUMNS} />
          </Card>
        </>
      )}
    </div>
  );
}
