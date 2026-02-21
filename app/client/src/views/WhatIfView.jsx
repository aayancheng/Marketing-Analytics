import Card from '../components/Card';
import SliderControl from '../components/SliderControl';
import TopSlots from '../components/TopSlots';
import ShapChart from '../components/ShapChart';
import Heatmap from '../components/Heatmap';
import ErrorBanner from '../components/ErrorBanner';
import { SLIDER_CONFIG } from '../lib/constants';

export default function WhatIfView({ hook }) {
  const { params, data, error, loading, pending, update } = hook;

  const handleSlider = (key, value) => {
    update({ ...params, [key]: value });
  };

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-2xl font-bold text-slate-800 mb-1">What-If Simulator</h2>
        <p className="text-sm text-slate-500">Adjust customer features to explore predicted engagement</p>
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

      <Card>
        <h3 className="text-lg font-semibold text-slate-700 mb-3">Top Send Windows</h3>
        <TopSlots slots={data?.top_3_slots} loading={loading} />
      </Card>

      {data?.shap_values && (
        <Card>
          <h3 className="text-lg font-semibold text-slate-700 mb-3">Feature Contributions (SHAP)</h3>
          <p className="text-xs text-slate-400 mb-2">How each feature influenced the top prediction</p>
          <ShapChart shapValues={data.shap_values} />
        </Card>
      )}

      <Card>
        <h3 className="text-lg font-semibold text-slate-700 mb-3">Engagement Heatmap</h3>
        <Heatmap heatmap={data?.heatmap} dim={pending} topSlots={data?.top_3_slots || []} />
      </Card>
    </div>
  );
}
