import { Search, SlidersHorizontal, Users, Menu, X } from 'lucide-react';

const NAV_ITEMS = [
  { id: 'lookup', label: 'Customer Lookup', icon: Search },
  { id: 'whatif', label: 'Risk Simulator', icon: SlidersHorizontal },
  { id: 'segments', label: 'Segment Explorer', icon: Users },
];

export default function Sidebar({ activeView, onNavigate, open, onToggle }) {
  return (
    <>
      {/* Mobile hamburger */}
      <button
        onClick={onToggle}
        className="fixed top-4 left-4 z-50 md:hidden bg-brand-950 text-white p-2 rounded-xl shadow-lg"
      >
        {open ? <X size={20} /> : <Menu size={20} />}
      </button>

      {/* Backdrop */}
      {open && (
        <div className="fixed inset-0 bg-black/40 z-30 md:hidden" onClick={onToggle} />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed left-0 top-0 h-screen w-56 bg-gradient-to-b from-brand-950 to-brand-900 flex flex-col text-white z-40 transition-transform md:translate-x-0 ${
          open ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="px-5 py-6">
          <h1 className="text-lg font-bold tracking-tight">
            <span className="bg-gradient-to-r from-brand-400 to-accent-400 bg-clip-text text-transparent">
              Churn Analytics
            </span>
          </h1>
          <p className="text-xs text-brand-300 mt-1">Churn propensity model</p>
        </div>

        <nav className="flex-1 px-3 space-y-1">
          {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => {
                onNavigate(id);
                onToggle();
              }}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-colors ${
                activeView === id
                  ? 'bg-white/10 border-l-2 border-accent-400 text-white'
                  : 'text-brand-200 hover:bg-white/5 hover:text-white border-l-2 border-transparent'
              }`}
            >
              <Icon size={18} />
              {label}
            </button>
          ))}
        </nav>

        <div className="px-5 py-4 text-[11px] text-brand-400">
          Churn Propensity ML
        </div>
      </aside>
    </>
  );
}
