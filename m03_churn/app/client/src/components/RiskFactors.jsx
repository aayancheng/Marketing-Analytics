import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from 'recharts';

// Map technical feature names to human-readable labels
const FEATURE_LABELS = {
  'Contract_Month-to-month': 'Month-to-month Contract',
  'Contract_One year': 'One Year Contract',
  'Contract_Two year': 'Two Year Contract',
  'tenure': 'Tenure (months)',
  'MonthlyCharges': 'Monthly Charges',
  'TotalCharges': 'Total Charges',
  'InternetService_Fiber optic': 'Fiber Optic Internet',
  'InternetService_DSL': 'DSL Internet',
  'InternetService_No': 'No Internet Service',
  'PaymentMethod_Electronic check': 'Electronic Check Payment',
  'PaymentMethod_Mailed check': 'Mailed Check Payment',
  'PaymentMethod_Bank transfer (automatic)': 'Bank Transfer (Auto)',
  'PaymentMethod_Credit card (automatic)': 'Credit Card (Auto)',
  'OnlineSecurity_No': 'No Online Security',
  'OnlineSecurity_Yes': 'Has Online Security',
  'TechSupport_No': 'No Tech Support',
  'TechSupport_Yes': 'Has Tech Support',
  'OnlineBackup_No': 'No Online Backup',
  'OnlineBackup_Yes': 'Has Online Backup',
  'DeviceProtection_No': 'No Device Protection',
  'DeviceProtection_Yes': 'Has Device Protection',
  'StreamingTV_Yes': 'Streaming TV',
  'StreamingMovies_Yes': 'Streaming Movies',
  'MultipleLines_Yes': 'Multiple Lines',
  'Dependents_Yes': 'Has Dependents',
  'Partner_Yes': 'Has Partner',
  'SeniorCitizen': 'Senior Citizen',
  'PaperlessBilling_Yes': 'Paperless Billing',
  'num_services': 'Active Services',
};

function humanize(feature) {
  return FEATURE_LABELS[feature] || feature.replace(/_/g, ' ');
}

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload || !payload.length) return null;
  const { feature, contribution, value } = payload[0].payload;
  const sign = contribution >= 0 ? '+' : '';
  return (
    <div className="bg-white border border-slate-200 rounded-xl px-3 py-2 shadow text-xs">
      <p className="font-semibold text-slate-700 mb-1">{humanize(feature)}</p>
      <p className="text-slate-500">
        Feature value: <span className="font-medium text-slate-800">{value}</span>
      </p>
      <p className={contribution >= 0 ? 'text-red-600 font-semibold' : 'text-green-600 font-semibold'}>
        SHAP: {sign}{Number(contribution).toFixed(4)}
      </p>
      <p className="text-slate-400 text-[11px] mt-1">
        {contribution >= 0 ? 'Increases churn risk' : 'Decreases churn risk'}
      </p>
    </div>
  );
};

export default function RiskFactors({ shapFactors }) {
  if (!shapFactors || shapFactors.length === 0) {
    return (
      <p className="text-sm text-slate-400 py-4 text-center">No risk factor data available.</p>
    );
  }

  const data = shapFactors
    .slice(0, 5)
    .map((s) => ({
      feature: s.feature,
      label: humanize(s.feature),
      contribution: parseFloat(Number(s.contribution).toFixed(4)),
      value: s.value,
    }))
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

  return (
    <div>
      <h3 className="text-lg font-semibold text-slate-700 mb-3">Top Risk Drivers</h3>
      <p className="text-xs text-slate-400 mb-3">
        SHAP contributions — red increases churn risk, green decreases it
      </p>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ left: 10, right: 30, top: 5, bottom: 5 }}
        >
          <XAxis
            type="number"
            tick={{ fontSize: 11 }}
            tickFormatter={(v) => (v > 0 ? `+${v.toFixed(3)}` : v.toFixed(3))}
          />
          <YAxis
            type="category"
            dataKey="label"
            tick={{ fontSize: 11 }}
            width={160}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine x={0} stroke="#cbd5e1" strokeWidth={1} />
          <Bar dataKey="contribution" radius={[0, 4, 4, 0]}>
            {data.map((entry, i) => (
              <Cell
                key={i}
                fill={entry.contribution >= 0 ? '#ef4444' : '#22c55e'}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
