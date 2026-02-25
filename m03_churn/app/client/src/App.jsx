import { useState } from 'react';
import Sidebar from './components/Sidebar';
import LookupView from './views/LookupView';
import WhatIfView from './views/WhatIfView';
import SegmentView from './views/SegmentView';
import { useCustomer, usePredict, useSegments } from './lib/hooks';

export default function App() {
  const [view, setView] = useState('lookup');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const customerHook = useCustomer();
  const predictHook = usePredict();
  const segmentsHook = useSegments();

  return (
    <div className="flex min-h-screen bg-slate-50">
      <Sidebar
        activeView={view}
        onNavigate={setView}
        open={sidebarOpen}
        onToggle={() => setSidebarOpen((o) => !o)}
      />
      <main className="flex-1 md:ml-56 p-6 md:p-8 max-w-5xl pt-16 md:pt-8">
        {view === 'lookup' && <LookupView hook={customerHook} />}
        {view === 'whatif' && <WhatIfView hook={predictHook} />}
        {view === 'segments' && <SegmentView hook={segmentsHook} />}
      </main>
    </div>
  );
}
