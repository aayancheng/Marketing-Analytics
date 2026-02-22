# Frontend Rebuild — Bold & Colorful Dashboard

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the monolithic 387-line App.jsx with a component-based React app using TailwindCSS, Headless UI, and a vibrant violet/pink/indigo color palette with sidebar navigation.

**Architecture:** Sidebar layout with 3 views (Lookup, What-If, Segments). State managed via custom hooks per view. Shared UI components for heatmap, stat cards, top slots. All API calls extracted to a shared api module.

**Tech Stack:** React 18, Vite 5, TailwindCSS 3, @headlessui/react 2, lucide-react, Recharts, axios

---

## Progress (2026-02-21)

**Tasks 1–15: COMPLETE** — All code implemented, build passes, committed as `5ce9137`.
- Tailwind + PostCSS configured
- Shared lib layer (api.js, hooks.js, constants.js)
- All 12 components built
- All 3 views built (LookupView, WhatIfView, SegmentView)
- App.jsx rewritten as sidebar layout shell
- Verified in Chrome via Playwright: all 3 views render correctly with live API data

**Task 16: COMPLETE**
- [x] Step 4: Responsive check — mobile hamburger sidebar added, tested at 375px/768px/1280px
- [x] Step 5: Integration smoke test — all 10 API endpoints return correct status codes
- [x] Step 6: Final polish commit

**All 16 tasks COMPLETE.** Frontend rebuild finished.

---

### Task 1: Install autoprefixer and create Tailwind config

**Files:**
- Create: `project/app/client/tailwind.config.js`
- Create: `project/app/client/postcss.config.js`

**Step 1: Install autoprefixer**

Run: `cd /Users/aayan/MarketingAnalytics/project && npm install -D autoprefixer --workspace=app/client`

**Step 2: Create tailwind.config.js**

```js
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#f5f3ff',
          100: '#ede9fe',
          200: '#ddd6fe',
          300: '#c4b5fd',
          400: '#a78bfa',
          500: '#8b5cf6',
          600: '#7c3aed',
          700: '#6d28d9',
          800: '#5b21b6',
          900: '#4c1d95',
          950: '#1e1b4b',
        },
        accent: {
          300: '#f9a8d4',
          400: '#f472b6',
          500: '#ec4899',
          600: '#db2777',
        },
      },
      fontFamily: {
        sans: ['"Inter"', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
```

**Step 3: Create postcss.config.js**

```js
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
};
```

**Step 4: Verify Vite picks up PostCSS**

Run: `cd /Users/aayan/MarketingAnalytics/project/app/client && npx vite build --mode development 2>&1 | head -20`
Expected: Build succeeds without PostCSS errors.

**Step 5: Commit**

```bash
git add project/app/client/tailwind.config.js project/app/client/postcss.config.js project/app/client/package.json
git commit -m "chore: configure tailwindcss and postcss for frontend"
```

---

### Task 2: Replace styles.css with Tailwind entry CSS

**Files:**
- Delete: `project/app/client/src/styles.css`
- Create: `project/app/client/src/index.css`
- Modify: `project/app/client/src/main.jsx` (line 4: change import)

**Step 1: Create index.css**

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  body {
    @apply bg-slate-50 text-slate-800 antialiased;
    font-family: 'Inter', system-ui, sans-serif;
  }
}

@layer components {
  .heatmap-grid {
    display: grid;
    grid-template-columns: auto repeat(24, 1fr);
    gap: 2px;
  }
  .heat-cell {
    aspect-ratio: 1 / 1;
    border-radius: 3px;
    transition: transform 0.1s;
  }
  .heat-cell:hover {
    transform: scale(1.3);
    z-index: 10;
  }
}
```

**Step 2: Update main.jsx import**

Change line 4 from:
```js
import './styles.css';
```
To:
```js
import './index.css';
```

**Step 3: Delete old styles.css**

Run: `rm /Users/aayan/MarketingAnalytics/project/app/client/src/styles.css`

**Step 4: Verify**

Run: `cd /Users/aayan/MarketingAnalytics/project/app/client && npx vite build --mode development 2>&1 | tail -5`
Expected: Build succeeds.

**Step 5: Commit**

```bash
git add project/app/client/src/index.css project/app/client/src/main.jsx
git rm project/app/client/src/styles.css
git commit -m "chore: replace custom CSS with Tailwind entry file"
```

---

### Task 3: Create shared API module and constants

**Files:**
- Create: `project/app/client/src/lib/api.js`
- Create: `project/app/client/src/lib/constants.js`

**Step 1: Create api.js**

```js
import axios from 'axios';

const api = axios.create({ baseURL: 'http://localhost:3001' });

export async function fetchCustomer(id) {
  const resp = await api.get(`/api/customer/${id}`);
  return resp.data;
}

export async function fetchPrediction(params) {
  const resp = await api.post('/api/predict', params);
  return resp.data;
}

export async function fetchCustomers(page = 1, perPage = 100, segment = null) {
  const params = { page, per_page: perPage };
  if (segment) params.segment = segment;
  const resp = await api.get('/api/customers', { params });
  return resp.data;
}

export async function fetchAllCustomers() {
  const first = await fetchCustomers(1, 100);
  const totalPages = first.total_pages || 1;
  const customers = [...(first.customers || [])];

  for (let page = 2; page <= totalPages; page++) {
    const resp = await fetchCustomers(page, 100);
    customers.push(...(resp.customers || []));
  }

  customers.sort((a, b) => a.customer_id - b.customer_id);
  return customers;
}

export async function fetchSegments() {
  const resp = await api.get('/api/segments');
  return resp.data;
}
```

**Step 2: Create constants.js**

```js
export const DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

export const SEGMENTS = ['Champions', 'Loyal', 'At Risk', 'Hibernating'];

export const SLIDER_CONFIG = {
  recency_days: { label: 'Recency (days)', min: 1, max: 365, step: 1, default: 30 },
  frequency: { label: 'Frequency', min: 1, max: 100, step: 1, default: 10 },
  modal_hour: { label: 'Modal Hour', min: 0, max: 23, step: 1, default: 10 },
  purchase_hour_entropy: { label: 'Entropy', min: 0, max: 1, step: 0.01, default: 0.3 },
};

export const DEFAULT_WHATIF = {
  recency_days: 30,
  frequency: 10,
  modal_hour: 10,
  purchase_hour_entropy: 0.3,
};

export function interpolateHeatColor(value) {
  // indigo-900 (#312e81) → violet-500 (#8b5cf6) → pink-400 (#f472b6)
  const clamp = Math.max(0, Math.min(1, value));
  if (clamp < 0.5) {
    const t = clamp * 2;
    const r = Math.round(49 + t * (139 - 49));
    const g = Math.round(46 + t * (92 - 46));
    const b = Math.round(129 + t * (246 - 129));
    return `rgb(${r}, ${g}, ${b})`;
  }
  const t = (clamp - 0.5) * 2;
  const r = Math.round(139 + t * (244 - 139));
  const g = Math.round(92 + t * (114 - 92));
  const b = Math.round(246 + t * (182 - 246));
  return `rgb(${r}, ${g}, ${b})`;
}

export function formatError(error) {
  const messages = {
    customer_not_found: 'Customer ID not found. Please check and try again.',
    insufficient_history: 'This customer has fewer than 5 transactions.',
    invalid_customer_id: 'Customer ID must be numeric.',
  };
  return messages[error?.error] || error?.message || 'Request failed.';
}
```

**Step 3: Commit**

```bash
git add project/app/client/src/lib/api.js project/app/client/src/lib/constants.js
git commit -m "feat: add shared API module and constants"
```

---

### Task 4: Create custom hooks

**Files:**
- Create: `project/app/client/src/lib/hooks.js`

**Step 1: Create hooks.js**

```js
import { useCallback, useEffect, useRef, useState } from 'react';
import { fetchCustomer, fetchAllCustomers, fetchPrediction, fetchCustomers, fetchSegments } from './api';
import { DEFAULT_WHATIF } from './constants';

export function useCustomer() {
  const [customerId, setCustomerId] = useState('');
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [allCustomers, setAllCustomers] = useState([]);
  const [customersLoading, setCustomersLoading] = useState(false);

  const lookup = useCallback(async (id) => {
    if (!id) return;
    setLoading(true);
    setError(null);
    try {
      const result = await fetchCustomer(id);
      setData(result);
    } catch (e) {
      setData(null);
      setError(e.response?.data?.detail || { error: 'unknown', message: 'Request failed' });
    } finally {
      setLoading(false);
    }
  }, []);

  const loadAll = useCallback(async () => {
    setCustomersLoading(true);
    try {
      const customers = await fetchAllCustomers();
      setAllCustomers(customers);
    } finally {
      setCustomersLoading(false);
    }
  }, []);

  useEffect(() => { loadAll(); }, [loadAll]);

  return { customerId, setCustomerId, data, error, loading, lookup, allCustomers, customersLoading };
}

export function usePredict() {
  const [params, setParams] = useState(DEFAULT_WHATIF);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [pending, setPending] = useState(false);
  const timerRef = useRef(null);

  const update = useCallback((next) => {
    setParams(next);
    setPending(true);
    setError(null);
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(async () => {
      setPending(false);
      setLoading(true);
      try {
        const result = await fetchPrediction(next);
        setData(result);
      } catch {
        setError({ message: 'Prediction failed. Please try again.' });
      } finally {
        setLoading(false);
      }
    }, 300);
  }, []);

  // Initial prediction on mount
  useEffect(() => { update(DEFAULT_WHATIF); }, [update]);

  return { params, data, error, loading, pending, update };
}

export function useSegments() {
  const [segment, setSegment] = useState('Champions');
  const [customers, setCustomers] = useState([]);
  const [summary, setSummary] = useState([]);
  const [loading, setLoading] = useState(false);

  const loadCustomers = useCallback(async (seg) => {
    setLoading(true);
    try {
      const resp = await fetchCustomers(1, 100, seg);
      setCustomers(resp.customers || []);
    } finally {
      setLoading(false);
    }
  }, []);

  const loadSummary = useCallback(async () => {
    try {
      const resp = await fetchSegments();
      setSummary(resp.segments || []);
    } catch {
      // non-critical
    }
  }, []);

  useEffect(() => { loadCustomers(segment); }, [segment, loadCustomers]);
  useEffect(() => { loadSummary(); }, [loadSummary]);

  const changeSegment = useCallback((seg) => setSegment(seg), []);

  return { segment, changeSegment, customers, summary, loading };
}
```

**Step 2: Commit**

```bash
git add project/app/client/src/lib/hooks.js
git commit -m "feat: add custom hooks for customer, predict, and segments"
```

---

### Task 5: Build Sidebar component

**Files:**
- Create: `project/app/client/src/components/Sidebar.jsx`

**Step 1: Create Sidebar.jsx**

```jsx
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
```

**Step 2: Commit**

```bash
git add project/app/client/src/components/Sidebar.jsx
git commit -m "feat: add sidebar navigation component"
```

---

### Task 6: Build shared UI components (Card, StatCard, ErrorBanner, LoadingSpinner, SliderControl)

**Files:**
- Create: `project/app/client/src/components/Card.jsx`
- Create: `project/app/client/src/components/StatCard.jsx`
- Create: `project/app/client/src/components/ErrorBanner.jsx`
- Create: `project/app/client/src/components/LoadingSpinner.jsx`
- Create: `project/app/client/src/components/SliderControl.jsx`

**Step 1: Create Card.jsx**

```jsx
export default function Card({ children, className = '', wide = false }) {
  return (
    <div className={`bg-white rounded-2xl shadow-md p-5 ${wide ? 'col-span-full' : ''} ${className}`}>
      {children}
    </div>
  );
}
```

**Step 2: Create StatCard.jsx**

```jsx
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
```

**Step 3: Create ErrorBanner.jsx**

```jsx
import { AlertTriangle, X } from 'lucide-react';
import { useState } from 'react';
import { formatError } from '../lib/constants';

export default function ErrorBanner({ error }) {
  const [dismissed, setDismissed] = useState(false);
  if (!error || dismissed) return null;

  return (
    <div className="flex items-center gap-3 bg-red-50 border border-red-200 text-red-700 rounded-xl px-4 py-3">
      <AlertTriangle size={18} />
      <span className="flex-1 text-sm">{formatError(error)}</span>
      <button onClick={() => setDismissed(true)} className="hover:bg-red-100 rounded-lg p-1">
        <X size={16} />
      </button>
    </div>
  );
}
```

**Step 4: Create LoadingSpinner.jsx**

```jsx
export default function LoadingSpinner({ size = 'md' }) {
  const sizeClass = size === 'sm' ? 'w-5 h-5' : size === 'lg' ? 'w-10 h-10' : 'w-7 h-7';
  return (
    <div className="flex items-center justify-center py-8">
      <div className={`${sizeClass} rounded-full border-2 border-brand-200 border-t-brand-600 animate-spin`} />
    </div>
  );
}
```

**Step 5: Create SliderControl.jsx**

```jsx
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
```

**Step 6: Commit**

```bash
git add project/app/client/src/components/Card.jsx project/app/client/src/components/StatCard.jsx project/app/client/src/components/ErrorBanner.jsx project/app/client/src/components/LoadingSpinner.jsx project/app/client/src/components/SliderControl.jsx
git commit -m "feat: add shared UI components (Card, StatCard, ErrorBanner, LoadingSpinner, SliderControl)"
```

---

### Task 7: Build Heatmap component

**Files:**
- Create: `project/app/client/src/components/Heatmap.jsx`

**Step 1: Create Heatmap.jsx**

```jsx
import { useState } from 'react';
import { DAY_NAMES, interpolateHeatColor } from '../lib/constants';

const HOURS = Array.from({ length: 24 }, (_, i) => i);

export default function Heatmap({ heatmap, dim = false, topSlots = [] }) {
  const [tooltip, setTooltip] = useState(null);

  if (!heatmap || heatmap.length !== 7) return null;

  const topSet = new Set(topSlots.map((s) => `${s.send_dow}-${s.send_hour}`));

  return (
    <div className={`relative transition-opacity ${dim ? 'opacity-40' : ''}`}>
      {/* Hour labels */}
      <div className="grid gap-[2px] mb-1" style={{ gridTemplateColumns: 'auto repeat(24, 1fr)' }}>
        <div />
        {HOURS.map((h) => (
          <div key={h} className="text-[10px] text-slate-400 text-center font-medium">
            {h}
          </div>
        ))}
      </div>

      {/* Grid */}
      <div className="heatmap-grid">
        {heatmap.map((row, day) => (
          <div key={day} className="contents">
            <div className="text-xs text-slate-500 font-medium flex items-center pr-2 justify-end">
              {DAY_NAMES[day]}
            </div>
            {row.map((v, hour) => {
              const isTop = topSet.has(`${day}-${hour}`);
              return (
                <div
                  key={`${day}-${hour}`}
                  className={`heat-cell relative cursor-pointer ${isTop ? 'ring-2 ring-accent-400 ring-offset-1 z-10' : ''}`}
                  style={{ backgroundColor: interpolateHeatColor(v) }}
                  onMouseEnter={() => setTooltip({ day, hour, value: v })}
                  onMouseLeave={() => setTooltip(null)}
                />
              );
            })}
          </div>
        ))}
      </div>

      {/* Tooltip */}
      {tooltip && (
        <div className="absolute -top-10 left-1/2 -translate-x-1/2 bg-slate-800 text-white text-xs px-3 py-1.5 rounded-lg shadow-lg pointer-events-none whitespace-nowrap z-20">
          {DAY_NAMES[tooltip.day]} {tooltip.hour}:00 — {(tooltip.value * 100).toFixed(1)}%
        </div>
      )}
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add project/app/client/src/components/Heatmap.jsx
git commit -m "feat: add vibrant heatmap component with labels and tooltips"
```

---

### Task 8: Build TopSlots component

**Files:**
- Create: `project/app/client/src/components/TopSlots.jsx`

**Step 1: Create TopSlots.jsx**

```jsx
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
```

**Step 2: Commit**

```bash
git add project/app/client/src/components/TopSlots.jsx
git commit -m "feat: add top slots recommendation cards"
```

---

### Task 9: Build ShapChart component

**Files:**
- Create: `project/app/client/src/components/ShapChart.jsx`

**Step 1: Create ShapChart.jsx**

```jsx
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

export default function ShapChart({ shapValues }) {
  if (!shapValues || shapValues.length === 0) return null;

  const data = shapValues.map((s) => ({
    feature: s.feature.replace(/_/g, ' '),
    contribution: parseFloat(s.contribution.toFixed(4)),
    value: s.value,
  }));

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} layout="vertical" margin={{ left: 120, right: 20, top: 5, bottom: 5 }}>
        <XAxis type="number" tick={{ fontSize: 11 }} />
        <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={110} />
        <Tooltip
          formatter={(val, name, props) => [
            `${val > 0 ? '+' : ''}${val.toFixed(4)}`,
            `Value: ${props.payload.value.toFixed(2)}`,
          ]}
          contentStyle={{ borderRadius: '8px', fontSize: '12px' }}
        />
        <Bar dataKey="contribution" radius={[0, 4, 4, 0]}>
          {data.map((entry, i) => (
            <Cell key={i} fill={entry.contribution >= 0 ? '#7c3aed' : '#ec4899'} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
```

**Step 2: Commit**

```bash
git add project/app/client/src/components/ShapChart.jsx
git commit -m "feat: add SHAP horizontal bar chart component"
```

---

### Task 10: Build CustomerSelect component (Headless UI Combobox)

**Files:**
- Create: `project/app/client/src/components/CustomerSelect.jsx`

**Step 1: Create CustomerSelect.jsx**

```jsx
import { useState } from 'react';
import { Combobox, ComboboxInput, ComboboxOptions, ComboboxOption, ComboboxButton } from '@headlessui/react';
import { Search, ChevronDown } from 'lucide-react';

export default function CustomerSelect({ customers, loading, value, onChange }) {
  const [query, setQuery] = useState('');

  const filtered =
    query === ''
      ? customers.slice(0, 50)
      : customers
          .filter((c) => String(c.customer_id).includes(query))
          .slice(0, 50);

  const selected = customers.find((c) => String(c.customer_id) === String(value)) || null;

  return (
    <Combobox value={selected} onChange={(c) => c && onChange(String(c.customer_id))}>
      <div className="relative">
        <div className="flex items-center bg-white border border-slate-200 rounded-xl shadow-sm focus-within:ring-2 focus-within:ring-brand-500 focus-within:border-brand-500">
          <Search size={16} className="ml-3 text-slate-400" />
          <ComboboxInput
            className="w-full border-none bg-transparent py-2.5 pl-2 pr-8 text-sm text-slate-800 focus:outline-none"
            placeholder={loading ? 'Loading customers...' : 'Search by customer ID...'}
            displayValue={(c) => (c ? String(c.customer_id) : '')}
            onChange={(e) => setQuery(e.target.value)}
          />
          <ComboboxButton className="absolute right-2 text-slate-400">
            <ChevronDown size={16} />
          </ComboboxButton>
        </div>

        <ComboboxOptions className="absolute z-30 mt-1 w-full max-h-60 overflow-auto bg-white rounded-xl shadow-lg border border-slate-100 py-1">
          {filtered.length === 0 ? (
            <div className="px-4 py-2 text-sm text-slate-400">No customers found</div>
          ) : (
            filtered.map((c) => (
              <ComboboxOption
                key={c.customer_id}
                value={c}
                className="flex items-center justify-between px-4 py-2 text-sm cursor-pointer data-[focus]:bg-brand-50 data-[selected]:bg-brand-100"
              >
                <span className="font-medium">{c.customer_id}</span>
                <span className="text-xs px-2 py-0.5 rounded-full bg-brand-100 text-brand-700">
                  {c.rfm_segment}
                </span>
              </ComboboxOption>
            ))
          )}
        </ComboboxOptions>
      </div>
    </Combobox>
  );
}
```

**Step 2: Commit**

```bash
git add project/app/client/src/components/CustomerSelect.jsx
git commit -m "feat: add searchable customer combobox with Headless UI"
```

---

### Task 11: Build SegmentPicker and DataTable components

**Files:**
- Create: `project/app/client/src/components/SegmentPicker.jsx`
- Create: `project/app/client/src/components/DataTable.jsx`

**Step 1: Create SegmentPicker.jsx**

```jsx
import { Listbox, ListboxButton, ListboxOptions, ListboxOption } from '@headlessui/react';
import { ChevronDown, Check } from 'lucide-react';
import { SEGMENTS } from '../lib/constants';

export default function SegmentPicker({ value, onChange }) {
  return (
    <Listbox value={value} onChange={onChange}>
      <div className="relative w-64">
        <ListboxButton className="w-full flex items-center justify-between bg-white border border-slate-200 rounded-xl px-4 py-2.5 text-sm font-medium text-slate-700 shadow-sm hover:border-brand-400 focus:outline-none focus:ring-2 focus:ring-brand-500">
          {value}
          <ChevronDown size={16} className="text-slate-400" />
        </ListboxButton>
        <ListboxOptions className="absolute z-30 mt-1 w-full bg-white rounded-xl shadow-lg border border-slate-100 py-1">
          {SEGMENTS.map((seg) => (
            <ListboxOption
              key={seg}
              value={seg}
              className="flex items-center justify-between px-4 py-2 text-sm cursor-pointer data-[focus]:bg-brand-50 data-[selected]:bg-brand-100"
            >
              {seg}
              {seg === value && <Check size={14} className="text-brand-600" />}
            </ListboxOption>
          ))}
        </ListboxOptions>
      </div>
    </Listbox>
  );
}
```

**Step 2: Create DataTable.jsx**

```jsx
export default function DataTable({ rows, columns }) {
  if (!rows || rows.length === 0) return <p className="text-sm text-slate-400 py-4">No data available.</p>;

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-200">
            {columns.map((col) => (
              <th key={col.key} className="text-left py-2.5 px-3 text-xs font-semibold text-slate-500 uppercase tracking-wide">
                {col.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className={`border-b border-slate-100 ${i % 2 === 1 ? 'bg-slate-50/50' : ''}`}>
              {columns.map((col) => (
                <td key={col.key} className="py-2.5 px-3 text-slate-700">
                  {col.render ? col.render(row[col.key], row) : row[col.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

**Step 3: Commit**

```bash
git add project/app/client/src/components/SegmentPicker.jsx project/app/client/src/components/DataTable.jsx
git commit -m "feat: add SegmentPicker listbox and DataTable components"
```

---

### Task 12: Build LookupView

**Files:**
- Create: `project/app/client/src/views/LookupView.jsx`

**Step 1: Create LookupView.jsx**

```jsx
import { Clock, Hash, DollarSign, Tag } from 'lucide-react';
import Card from '../components/Card';
import StatCard from '../components/StatCard';
import CustomerSelect from '../components/CustomerSelect';
import TopSlots from '../components/TopSlots';
import Heatmap from '../components/Heatmap';
import ErrorBanner from '../components/ErrorBanner';
import LoadingSpinner from '../components/LoadingSpinner';

export default function LookupView({ hook }) {
  const { customerId, setCustomerId, data, error, loading, lookup, allCustomers, customersLoading } = hook;

  const handleSelect = (id) => {
    setCustomerId(id);
    lookup(id);
  };

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-2xl font-bold text-slate-800 mb-1">Customer Lookup</h2>
        <p className="text-sm text-slate-500">Search for a customer to view their optimal send times</p>
      </div>

      <Card>
        <CustomerSelect
          customers={allCustomers}
          loading={customersLoading}
          value={customerId}
          onChange={handleSelect}
        />
      </Card>

      <ErrorBanner error={error} />

      {loading && <LoadingSpinner />}

      {data?.profile && (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <StatCard icon={Clock} label="Recency" value={`${data.profile.recency_days.toFixed(0)}d`} color="brand" />
            <StatCard icon={Hash} label="Frequency" value={data.profile.frequency} color="accent" />
            <StatCard icon={DollarSign} label="Monetary" value={`£${data.profile.monetary_total.toLocaleString()}`} color="emerald" />
            <StatCard icon={Tag} label="Segment" value={data.profile.rfm_segment} color="amber" />
          </div>

          <Card>
            <h3 className="text-lg font-semibold text-slate-700 mb-3">Top Send Windows</h3>
            <TopSlots slots={data.top_3_slots} loading={false} />
          </Card>

          <Card>
            <h3 className="text-lg font-semibold text-slate-700 mb-3">Engagement Heatmap</h3>
            <p className="text-xs text-slate-400 mb-3">7-day × 24-hour probability grid — brighter = higher engagement</p>
            <Heatmap heatmap={data.heatmap} topSlots={data.top_3_slots} />
          </Card>

          {data.out_of_distribution_warning && (
            <div className="bg-amber-50 border border-amber-200 text-amber-700 rounded-xl px-4 py-3 text-sm">
              This customer's profile is outside the training distribution. Predictions may be less reliable.
            </div>
          )}
        </>
      )}
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add project/app/client/src/views/LookupView.jsx
git commit -m "feat: add LookupView with customer search, profile stats, and heatmap"
```

---

### Task 13: Build WhatIfView

**Files:**
- Create: `project/app/client/src/views/WhatIfView.jsx`

**Step 1: Create WhatIfView.jsx**

```jsx
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
```

**Step 2: Commit**

```bash
git add project/app/client/src/views/WhatIfView.jsx
git commit -m "feat: add WhatIfView with sliders, SHAP chart, and heatmap"
```

---

### Task 14: Build SegmentView

**Files:**
- Create: `project/app/client/src/views/SegmentView.jsx`

**Step 1: Create SegmentView.jsx**

```jsx
import { useMemo } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Users, TrendingUp, Clock } from 'lucide-react';
import Card from '../components/Card';
import StatCard from '../components/StatCard';
import SegmentPicker from '../components/SegmentPicker';
import DataTable from '../components/DataTable';
import LoadingSpinner from '../components/LoadingSpinner';

const TABLE_COLUMNS = [
  { key: 'customer_id', label: 'ID' },
  { key: 'modal_purchase_hour', label: 'Modal Hour' },
  { key: 'open_rate', label: 'Open Rate', render: (v) => `${((v ?? 0) * 100).toFixed(1)}%` },
  { key: 'recency_days', label: 'Recency', render: (v) => `${v.toFixed(0)}d` },
];

export default function SegmentView({ hook }) {
  const { segment, changeSegment, customers, summary, loading } = hook;

  const currentSummary = summary.find((s) => s.segment === segment);

  const scatterData = useMemo(
    () => customers.map((r) => ({ x: r.modal_purchase_hour, y: r.open_rate ?? 0, id: r.customer_id })),
    [customers]
  );

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-slate-800 mb-1">Segment Explorer</h2>
          <p className="text-sm text-slate-500">Explore RFM segments and customer patterns</p>
        </div>
        <SegmentPicker value={segment} onChange={changeSegment} />
      </div>

      {currentSummary && (
        <div className="grid grid-cols-3 gap-3">
          <StatCard icon={Users} label="Customers" value={currentSummary.count} color="brand" />
          <StatCard icon={TrendingUp} label="Avg Open Rate" value={`${(currentSummary.mean_open_rate * 100).toFixed(1)}%`} color="accent" />
          <StatCard icon={Clock} label="Avg Recency" value={`${currentSummary.mean_recency_days.toFixed(0)}d`} color="emerald" />
        </div>
      )}

      {loading ? (
        <LoadingSpinner />
      ) : (
        <>
          <Card>
            <h3 className="text-lg font-semibold text-slate-700 mb-3">Modal Hour vs Open Rate</h3>
            <div style={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ left: 10, right: 10, top: 10, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis type="number" dataKey="x" name="Modal Hour" domain={[0, 23]} tick={{ fontSize: 11 }} />
                  <YAxis type="number" dataKey="y" name="Open Rate" domain={[0, 1]} tick={{ fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ borderRadius: '8px', fontSize: '12px' }}
                    formatter={(val) => typeof val === 'number' ? val.toFixed(2) : val}
                  />
                  <Scatter data={scatterData} fill="#7c3aed" fillOpacity={0.6} />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <Card>
            <h3 className="text-lg font-semibold text-slate-700 mb-3">Top 10 Customers</h3>
            <DataTable rows={customers.slice(0, 10)} columns={TABLE_COLUMNS} />
          </Card>
        </>
      )}
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add project/app/client/src/views/SegmentView.jsx
git commit -m "feat: add SegmentView with summary stats, scatter plot, and table"
```

---

### Task 15: Rewrite App.jsx (app shell)

**Files:**
- Modify: `project/app/client/src/App.jsx` (complete rewrite)

**Step 1: Rewrite App.jsx**

```jsx
import { useState } from 'react';
import Sidebar from './components/Sidebar';
import LookupView from './views/LookupView';
import WhatIfView from './views/WhatIfView';
import SegmentView from './views/SegmentView';
import { useCustomer, usePredict, useSegments } from './lib/hooks';

export default function App() {
  const [view, setView] = useState('lookup');
  const customerHook = useCustomer();
  const predictHook = usePredict();
  const segmentsHook = useSegments();

  return (
    <div className="flex min-h-screen bg-slate-50">
      <Sidebar activeView={view} onNavigate={setView} />
      <main className="flex-1 ml-56 p-8 max-w-5xl">
        {view === 'lookup' && <LookupView hook={customerHook} />}
        {view === 'whatif' && <WhatIfView hook={predictHook} />}
        {view === 'segments' && <SegmentView hook={segmentsHook} />}
      </main>
    </div>
  );
}
```

**Step 2: Verify build**

Run: `cd /Users/aayan/MarketingAnalytics/project/app/client && npx vite build 2>&1 | tail -10`
Expected: Build succeeds with no errors.

**Step 3: Commit**

```bash
git add project/app/client/src/App.jsx
git commit -m "feat: rewrite App.jsx as sidebar layout shell"
```

---

### Task 16: Manual verification and polish

**Step 1: Start the frontend dev server**

Run: `cd /Users/aayan/MarketingAnalytics/project/app/client && npm run dev`

**Step 2: Visual check in browser**

Open `http://localhost:5173` and verify:
- Sidebar renders with dark indigo gradient and 3 nav items
- Clicking nav items switches between views
- All Tailwind classes are applying (colors, shadows, rounded corners)

**Step 3: Test with backend running**

In separate terminals:
```bash
source .venv/bin/activate && uvicorn src.api.main:app --port 8000 --reload
node app/server/index.js
```

Verify:
- **Lookup:** Search combobox loads customers, selecting one shows stats + heatmap
- **What-If:** Sliders trigger debounced predictions, SHAP chart renders, heatmap updates
- **Segments:** Picker changes segment, summary stats show, scatter plot and table populate

**Step 4: Responsive check**

Resize browser to < 768px. Sidebar should still be visible (full collapse is a future enhancement — for now the sidebar is fixed width).

**Step 5: Run integration smoke test**

Run: `cd /Users/aayan/MarketingAnalytics/project && ./scripts/integration_smoke.sh`
Expected: All HTTP contract tests pass (backend unchanged).

**Step 6: Final commit if any polish tweaks needed**

```bash
git add -A
git commit -m "polish: responsive tweaks and final cleanup"
```
