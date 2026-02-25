export default function TopSlots({ slots, loading }) {
  if (loading) {
    return (
      <div className="grid grid-cols-3 gap-3">
        {[1, 2, 3].map((i) => (
          <div key={i} className="bg-white rounded-xl p-4 animate-pulse">
            <div className="h-4 bg-slate-200 rounded w-1/3 mb-2" />
            <div className="h-6 bg-slate-200 rounded w-2/3" />
          </div>
        ))}
      </div>
    );
  }

  if (!slots || slots.length === 0) return null;

  return (
    <div className="grid grid-cols-3 gap-3">
      {slots.map((slot, i) => (
        <div key={i} className="bg-white rounded-xl p-4 shadow-sm border border-slate-100">
          <div className="flex items-center gap-2 mb-2">
            <span className="flex items-center justify-center w-7 h-7 rounded-full bg-gradient-to-br from-brand-500 to-accent-500 text-white text-xs font-bold">
              {i + 1}
            </span>
            <span className="text-sm font-semibold text-slate-700">{slot.day_name}</span>
          </div>
          <p className="text-2xl font-bold text-slate-800">
            {String(slot.send_hour).padStart(2, '0')}:00
          </p>
          <div className="mt-2">
            <div className="flex justify-between text-xs text-slate-500 mb-1">
              <span>Confidence</span>
              <span className="font-semibold text-brand-600">{slot.confidence_pct}%</span>
            </div>
            <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-brand-500 to-accent-400 rounded-full transition-all"
                style={{ width: `${slot.confidence_pct}%` }}
              />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
