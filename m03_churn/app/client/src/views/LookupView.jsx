import { Clock, DollarSign, FileText, Layers } from 'lucide-react';
import Card from '../components/Card';
import StatCard from '../components/StatCard';
import CustomerSelect from '../components/CustomerSelect';
import ChurnGauge from '../components/ChurnGauge';
import RiskFactors from '../components/RiskFactors';
import ErrorBanner from '../components/ErrorBanner';
import LoadingSpinner from '../components/LoadingSpinner';
import { RISK_COLORS } from '../lib/constants';

const ACTION_MAP = {
  'High Risk': {
    label: 'Immediate Intervention Required',
    bg: 'bg-red-50 border-red-300 text-red-700',
    dot: 'bg-red-500',
  },
  'Medium-High Risk': {
    label: 'Proactive Retention Outreach',
    bg: 'bg-orange-50 border-orange-300 text-orange-700',
    dot: 'bg-orange-500',
  },
  'Medium-Low Risk': {
    label: 'Monitor and Nurture',
    bg: 'bg-yellow-50 border-yellow-300 text-yellow-700',
    dot: 'bg-yellow-500',
  },
  'Low Risk': {
    label: 'Standard Engagement',
    bg: 'bg-green-50 border-green-300 text-green-700',
    dot: 'bg-green-500',
  },
};

export default function LookupView({ hook }) {
  const {
    customerId,
    setCustomerId,
    data,
    error,
    loading,
    lookup,
    allCustomers,
    customersLoading,
  } = hook;

  const handleSelect = (id) => {
    setCustomerId(id);
    lookup(id);
  };

  const profile = data?.profile;
  const shapFactors = data?.shap_factors || data?.shap_values;
  const riskTier = data?.risk_tier;
  const churnScore = data?.churn_probability ?? 0;
  const action = ACTION_MAP[riskTier];

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-2xl font-bold text-slate-800 mb-1">Customer Lookup</h2>
        <p className="text-sm text-slate-500">
          Search for a subscriber to view their churn risk profile
        </p>
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

      {profile && riskTier && (
        <>
          {/* Gauge centered */}
          <Card>
            <div className="flex flex-col items-center py-2">
              <p className="text-xs text-slate-400 mb-4 uppercase tracking-wide font-medium">
                Churn Propensity Score
              </p>
              <ChurnGauge score={churnScore} size={220} />
            </div>
          </Card>

          {/* Stat cards */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <StatCard
              icon={Clock}
              label="Tenure"
              value={`${profile.tenure ?? '—'}mo`}
              color="brand"
            />
            <StatCard
              icon={DollarSign}
              label="Monthly Charges"
              value={profile.MonthlyCharges != null ? `$${Number(profile.MonthlyCharges).toFixed(2)}` : '—'}
              color="accent"
            />
            <StatCard
              icon={FileText}
              label="Contract"
              value={profile.Contract ?? '—'}
              color="amber"
            />
            <StatCard
              icon={Layers}
              label="Active Services"
              value={profile.num_services ?? '—'}
              color="emerald"
            />
          </div>

          {/* Risk factors SHAP */}
          {shapFactors && shapFactors.length > 0 && (
            <Card>
              <RiskFactors shapFactors={shapFactors} />
            </Card>
          )}

          {/* Recommended action */}
          {action && (
            <div
              className={`flex items-center gap-3 border rounded-xl px-4 py-3 ${action.bg}`}
            >
              <span className={`w-3 h-3 rounded-full flex-shrink-0 ${action.dot}`} />
              <div>
                <span className="text-xs font-semibold uppercase tracking-wide opacity-70">
                  Recommended Action
                </span>
                <p className="text-sm font-medium mt-0.5">{action.label}</p>
              </div>
            </div>
          )}

          {/* OOD warning */}
          {data?.out_of_distribution_warning && (
            <div className="bg-amber-50 border border-amber-200 text-amber-700 rounded-xl px-4 py-3 text-sm">
              This customer's profile is outside the training distribution. Predictions may be
              less reliable.
            </div>
          )}
        </>
      )}
    </div>
  );
}
