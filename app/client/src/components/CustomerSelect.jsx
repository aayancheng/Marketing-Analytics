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
