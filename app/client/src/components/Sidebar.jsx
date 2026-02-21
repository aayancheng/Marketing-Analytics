import { Search, SlidersHorizontal, Users } from 'lucide-react';

const NAV_ITEMS = [
  { id: 'lookup', label: 'Lookup', icon: Search },
  { id: 'whatif', label: 'What-If', icon: SlidersHorizontal },
  { id: 'segments', label: 'Segments', icon: Users },
];

export default function Sidebar({ activeView, onNavigate }) {
  return (
    <aside className="fixed left-0 top-0 h-screen w-56 bg-gradient-to-b from-brand-950 to-brand-900 flex flex-col text-white">
      <div className="px-5 py-6">
        <h1 className="text-lg font-bold tracking-tight">
          <span className="bg-gradient-to-r from-brand-400 to-accent-400 bg-clip-text text-transparent">
            TimeToEngage
          </span>
        </h1>
        <p className="text-xs text-brand-300 mt-1">Send-time optimizer</p>
      </div>

      <nav className="flex-1 px-3 space-y-1">
        {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => onNavigate(id)}
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
        Email Send-Time ML
      </div>
    </aside>
  );
}
