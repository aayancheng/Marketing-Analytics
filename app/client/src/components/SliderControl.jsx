export default function SliderControl({ label, value, min, max, step, onChange }) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium text-slate-600">{label}</label>
        <span className="text-sm font-bold bg-brand-100 text-brand-700 px-2.5 py-0.5 rounded-full">
          {typeof value === 'number' && step < 1 ? value.toFixed(2) : value}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-2 bg-brand-100 rounded-full appearance-none cursor-pointer accent-brand-600"
      />
    </div>
  );
}
