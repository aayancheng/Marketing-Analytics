import { useState } from 'react';
import Sidebar from './components/Sidebar';
import DecompositionView from './views/DecompositionView';
import ChannelPerformanceView from './views/ChannelPerformanceView';
import BudgetSimulatorView from './views/BudgetSimulatorView';
import OptimalAllocationView from './views/OptimalAllocationView';

export default function App() {
  const [view, setView] = useState('decomposition');
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="flex min-h-screen bg-slate-50">
      <Sidebar
        activeView={view}
        onNavigate={setView}
        open={sidebarOpen}
        onToggle={() => setSidebarOpen((o) => !o)}
      />
      <main className="flex-1 md:ml-56 p-6 md:p-8 max-w-6xl pt-16 md:pt-8">
        {view === 'decomposition' && <DecompositionView />}
        {view === 'performance' && <ChannelPerformanceView />}
        {view === 'simulator' && <BudgetSimulatorView />}
        {view === 'allocation' && <OptimalAllocationView />}
      </main>
    </div>
  );
}
