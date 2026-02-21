import { Clock, Hash, DollarSign, Tag } from 'lucide-react';
import Card from '../components/Card';
import StatCard from '../components/StatCard';
import CustomerSelect from '../components/CustomerSelect';
import TopSlots from '../components/TopSlots';
import Heatmap from '../components/Heatmap';
import ErrorBanner from '../components/ErrorBanner';
import LoadingSpinner from '../components/LoadingSpinner';

export default function LookupView({ hook }) {
  const { customerId, setCustomerId, data, error, loading, lookup, allCustomers, customersLoading } = hook;

  const handleSelect = (id) => {
    setCustomerId(id);
    lookup(id);
  };

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-2xl font-bold text-slate-800 mb-1">Customer Lookup</h2>
        <p className="text-sm text-slate-500">Search for a customer to view their optimal send times</p>
      </div>

      <Card>
        <CustomerSelect
          customers={allCustomers}
          loading={customersLoading}
          value={customerId}
          onChange={handleSelect}
        />
      </Card>

      <ErrorBanner error={error} />

      {loading && <LoadingSpinner />}

      {data?.profile && (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <StatCard icon={Clock} label="Recency" value={`${data.profile.recency_days.toFixed(0)}d`} color="brand" />
            <StatCard icon={Hash} label="Frequency" value={data.profile.frequency} color="accent" />
            <StatCard icon={DollarSign} label="Monetary" value={`£${data.profile.monetary_total.toLocaleString()}`} color="emerald" />
            <StatCard icon={Tag} label="Segment" value={data.profile.rfm_segment} color="amber" />
          </div>

          <Card>
            <h3 className="text-lg font-semibold text-slate-700 mb-3">Top Send Windows</h3>
            <TopSlots slots={data.top_3_slots} loading={false} />
          </Card>

          <Card>
            <h3 className="text-lg font-semibold text-slate-700 mb-3">Engagement Heatmap</h3>
            <p className="text-xs text-slate-400 mb-3">7-day × 24-hour probability grid — brighter = higher engagement</p>
            <Heatmap heatmap={data.heatmap} topSlots={data.top_3_slots} />
          </Card>

          {data.out_of_distribution_warning && (
            <div className="bg-amber-50 border border-amber-200 text-amber-700 rounded-xl px-4 py-3 text-sm">
              This customer's profile is outside the training distribution. Predictions may be less reliable.
            </div>
          )}
        </>
      )}
    </div>
  );
}
