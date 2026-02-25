import { useState } from 'react';
import {
  Combobox,
  ComboboxInput,
  ComboboxOptions,
  ComboboxOption,
  ComboboxButton,
} from '@headlessui/react';
import { Search, ChevronDown } from 'lucide-react';
import { RISK_COLORS } from '../lib/constants';

export default function CustomerSelect({ customers, loading, value, onChange }) {
  const [query, setQuery] = useState('');

  const filtered =
    query === ''
      ? customers.slice(0, 50)
      : customers
          .filter((c) =>
            String(c.customerID).toLowerCase().includes(query.toLowerCase())
          )
          .slice(0, 50);

  const selected =
    customers.find((c) => String(c.customerID) === String(value)) || null;

  return (
    <Combobox
      value={selected}
      onChange={(c) => c && onChange(String(c.customerID))}
    >
      <div className="relative">
        <div className="flex items-center bg-white border border-slate-200 rounded-xl shadow-sm focus-within:ring-2 focus-within:ring-brand-500 focus-within:border-brand-500">
          <Search size={16} className="ml-3 text-slate-400" />
          <ComboboxInput
            className="w-full border-none bg-transparent py-2.5 pl-2 pr-8 text-sm text-slate-800 focus:outline-none"
            placeholder={loading ? 'Loading customers...' : 'Search by customer ID (e.g. 7590-VHVEG)...'}
            displayValue={(c) => (c ? String(c.customerID) : '')}
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
            filtered.map((c) => {
              const tierColor = RISK_COLORS[c.risk_tier] || '#94a3b8';
              return (
                <ComboboxOption
                  key={c.customerID}
                  value={c}
                  className="flex items-center justify-between px-4 py-2 text-sm cursor-pointer data-[focus]:bg-brand-50 data-[selected]:bg-brand-100"
                >
                  <span className="font-medium">{c.customerID}</span>
                  {c.risk_tier && (
                    <span
                      className="text-xs px-2 py-0.5 rounded-full text-white font-semibold"
                      style={{ backgroundColor: tierColor }}
                    >
                      {c.risk_tier}
                    </span>
                  )}
                </ComboboxOption>
              );
            })
          )}
        </ComboboxOptions>
      </div>
    </Combobox>
  );
}
