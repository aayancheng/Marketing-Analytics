import Card from '../components/Card';
import SliderControl from '../components/SliderControl';
import CLVBadge from '../components/CLVBadge';
import ShapChart from '../components/ShapChart';
import ErrorBanner from '../components/ErrorBanner';
import LoadingSpinner from '../components/LoadingSpinner';
import { SLIDER_CONFIG, formatCurrency } from '../lib/constants';

export default function WhatIfView({ hook }) {
  const { params, data, error, loading, pending, update } = hook;

  const handleSlider = (key, value) => {
    update({ ...params, [key]: value });
  };

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-2xl font-bold text-slate-800 mb-1">CLV Simulator</h2>
        <p className="text-sm text-slate-500">Adjust customer features to explore predicted lifetime value</p>
      </div>

      <Card>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
          {Object.entries(SLIDER_CONFIG).map(([key, cfg]) => (
            <SliderControl
              key={key}
              label={cfg.label}
              value={params[key]}
              min={cfg.min}
              max={cfg.max}
              step={cfg.step}
              onChange={(v) => handleSlider(key, v)}
            />
          ))}
        </div>
      </Card>

      <ErrorBanner error={error} />

      {/* CLV Prediction Result */}
      <Card className={`text-center transition-opacity ${pending ? 'opacity-50' : ''}`}>
        <p className="text-sm text-slate-500 mb-2">Predicted 12-Month CLV</p>
        {loading ? (
          <LoadingSpinner size="sm" />
        ) : (
          <>
            <p className="text-4xl font-extrabold text-slate-800 mb-3">
              {formatCurrency(data?.predicted_clv)}
            </p>
            {data?.clv_segment && <CLVBadge segment={data.clv_segment} />}
          </>
        )}
      </Card>

      {data?.shap_factors && (
        <Card>
          <h3 className="text-lg font-semibold text-slate-700 mb-3">Feature Contributions (SHAP)</h3>
          <p className="text-xs text-slate-400 mb-2">How each feature influenced the CLV prediction</p>
          <ShapChart shapValues={data.shap_factors} />
        </Card>
      )}
    </div>
  );
}
