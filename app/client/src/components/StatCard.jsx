export default function StatCard({ icon: Icon, label, value, color = 'brand' }) {
  const colorMap = {
    brand: 'border-brand-500 bg-brand-50 text-brand-600',
    accent: 'border-accent-500 bg-pink-50 text-accent-600',
    emerald: 'border-emerald-500 bg-emerald-50 text-emerald-600',
    amber: 'border-amber-500 bg-amber-50 text-amber-600',
  };
  return (
    <div className={`rounded-xl border-l-4 p-4 ${colorMap[color] || colorMap.brand}`}>
      <div className="flex items-center gap-2 mb-1">
        {Icon && <Icon size={16} />}
        <span className="text-xs font-medium uppercase tracking-wide opacity-70">{label}</span>
      </div>
      <p className="text-xl font-bold text-slate-800">{value}</p>
    </div>
  );
}
