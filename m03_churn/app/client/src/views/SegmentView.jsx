import { useMemo } from 'react';
import { Users, TrendingUp, Clock, DollarSign } from 'lucide-react';
import Card from '../components/Card';
import StatCard from '../components/StatCard';
import SegmentPicker from '../components/SegmentPicker';
import DataTable from '../components/DataTable';
import LoadingSpinner from '../components/LoadingSpinner';

// Contract type churn summary table
function ContractSummary({ customers }) {
  if (!customers || customers.length === 0) return null;

  const groups = {};
  for (const c of customers) {
    const contract = c.Contract || 'Unknown';
    if (!groups[contract]) groups[contract] = { count: 0, churnSum: 0 };
    groups[contract].count += 1;
    groups[contract].churnSum += c.churn_probability ?? 0;
  }

  const rows = Object.entries(groups)
    .map(([contract, { count, churnSum }]) => ({
      contract,
      count,
      meanChurnPct: ((churnSum / count) * 100).toFixed(1),
    }))
    .sort((a, b) => b.meanChurnPct - a.meanChurnPct);

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-200">
            <th className="text-left py-2.5 px-3 text-xs font-semibold text-slate-500 uppercase tracking-wide">
              Contract Type
            </th>
            <th className="text-left py-2.5 px-3 text-xs font-semibold text-slate-500 uppercase tracking-wide">
              Customers
            </th>
            <th className="text-left py-2.5 px-3 text-xs font-semibold text-slate-500 uppercase tracking-wide">
              Mean Churn Score
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={row.contract}
              className={`border-b border-slate-100 ${i % 2 === 1 ? 'bg-slate-50/50' : ''}`}
            >
              <td className="py-2.5 px-3 text-slate-700 font-medium">{row.contract}</td>
              <td className="py-2.5 px-3 text-slate-700">{row.count}</td>
              <td className="py-2.5 px-3">
                <span
                  className={`font-semibold ${
                    parseFloat(row.meanChurnPct) >= 60
                      ? 'text-red-600'
                      : parseFloat(row.meanChurnPct) >= 40
                      ? 'text-orange-500'
                      : parseFloat(row.meanChurnPct) >= 20
                      ? 'text-yellow-600'
                      : 'text-green-600'
                  }`}
                >
                  {row.meanChurnPct}%
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function SegmentView({ hook }) {
  const { segment, changeSegment, customers, summary, loading } = hook;

  const currentSummary = summary.find((s) => s.segment === segment || s.risk_tier === segment);

  const mrrAtRisk = useMemo(() => {
    if (!customers.length) return null;
    const count = customers.length;
    const meanMonthly =
      customers.reduce((sum, c) => sum + (c.MonthlyCharges ?? 0), 0) / count;
    return (count * meanMonthly).toFixed(0);
  }, [customers]);

  const meanChurnScore = useMemo(() => {
    if (!customers.length) return null;
    return (
      (customers.reduce((sum, c) => sum + (c.churn_probability ?? 0), 0) / customers.length) *
      100
    ).toFixed(1);
  }, [customers]);

  const meanTenure = useMemo(() => {
    if (!customers.length) return null;
    return (
      customers.reduce((sum, c) => sum + (c.tenure ?? 0), 0) / customers.length
    ).toFixed(1);
  }, [customers]);

  return (
    <div className="space-y-5">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h2 className="text-2xl font-bold text-slate-800 mb-1">Segment Explorer</h2>
          <p className="text-sm text-slate-500">Explore risk tier segments and subscriber patterns</p>
        </div>
        <SegmentPicker value={segment} onChange={changeSegment} />
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <StatCard
          icon={Users}
          label="Customers"
          value={
            currentSummary?.count ?? customers.length
          }
          color="brand"
        />
        <StatCard
          icon={TrendingUp}
          label="Mean Churn Score"
          value={meanChurnScore != null ? `${meanChurnScore}%` : '—'}
          color="accent"
        />
        <StatCard
          icon={Clock}
          label="Mean Tenure"
          value={meanTenure != null ? `${meanTenure}mo` : '—'}
          color="emerald"
        />
        <StatCard
          icon={DollarSign}
          label="MRR at Risk"
          value={mrrAtRisk != null ? `$${Number(mrrAtRisk).toLocaleString()}` : '—'}
          color="amber"
        />
      </div>

      {loading ? (
        <LoadingSpinner />
      ) : (
        <>
          <Card>
            <h3 className="text-lg font-semibold text-slate-700 mb-3">
              Subscribers in Segment
            </h3>
            <DataTable rows={customers.slice(0, 25)} />
          </Card>

          <Card>
            <h3 className="text-lg font-semibold text-slate-700 mb-3">
              Churn Rate by Contract Type
            </h3>
            <p className="text-xs text-slate-400 mb-3">
              Average churn probability grouped by contract type within this segment
            </p>
            <ContractSummary customers={customers} />
          </Card>
        </>
      )}
    </div>
  );
}
