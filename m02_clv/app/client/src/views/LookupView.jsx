import { Clock, Hash, DollarSign, TrendingUp } from 'lucide-react';
import Card from '../components/Card';
import StatCard from '../components/StatCard';
import CustomerSelect from '../components/CustomerSelect';
import CLVBadge from '../components/CLVBadge';
import ShapChart from '../components/ShapChart';
import ErrorBanner from '../components/ErrorBanner';
import LoadingSpinner from '../components/LoadingSpinner';
import { formatCurrency } from '../lib/constants';

export default function LookupView({ hook }) {
  const { customerId, setCustomerId, data, error, loading, lookup, allCustomers, customersLoading } = hook;

  const handleSelect = (id) => {
    setCustomerId(id);
    lookup(id);
  };

  // CRITICAL: read clv_segment directly from data, NOT from data.prediction
  const clvSegment = data?.clv_segment;
  const profile = data?.profile;

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-2xl font-bold text-slate-800 mb-1">Customer Lookup</h2>
        <p className="text-sm text-slate-500">Search for a customer to view their predicted lifetime value</p>
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

      {/* Render when profile AND clvSegment — NOT profile && prediction */}
      {profile && clvSegment && (
        <>
          {/* CLV Prediction Card */}
          <Card className="text-center">
            <p className="text-sm text-slate-500 mb-2">Predicted 12-Month CLV</p>
            <p className="text-4xl font-extrabold text-slate-800 mb-3">
              {formatCurrency(data?.predicted_clv)}
            </p>
            <CLVBadge segment={clvSegment} />
          </Card>

          {/* Profile Stats */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <StatCard icon={Clock} label="Recency" value={`${profile.recency_days?.toFixed(0) ?? 0}d`} color="brand" />
            <StatCard icon={Hash} label="Frequency" value={profile.frequency ?? 0} color="accent" />
            <StatCard icon={DollarSign} label="Total Spend" value={formatCurrency(profile.monetary_total)} color="emerald" />
            <StatCard icon={TrendingUp} label="Velocity" value={`${(profile.purchase_velocity ?? 0).toFixed(1)}/mo`} color="amber" />
          </div>

          {/* SHAP explanations */}
          {data?.shap_factors && (
            <Card>
              <h3 className="text-lg font-semibold text-slate-700 mb-3">Feature Contributions (SHAP)</h3>
              <p className="text-xs text-slate-400 mb-2">How each feature influenced the CLV prediction</p>
              <ShapChart shapValues={data.shap_factors} />
            </Card>
          )}

          {data?.cold_start_warning && (
            <div className="bg-amber-50 border border-amber-200 text-amber-700 rounded-xl px-4 py-3 text-sm">
              This customer has limited purchase history. The prediction uses the median CLV as a fallback.
            </div>
          )}
        </>
      )}
    </div>
  );
}
