import { Listbox, ListboxButton, ListboxOptions, ListboxOption } from '@headlessui/react';
import { ChevronDown, Check } from 'lucide-react';
import { SEGMENTS, SEGMENT_COLORS } from '../lib/constants';

export default function SegmentPicker({ value, onChange }) {
  return (
    <Listbox value={value} onChange={onChange}>
      <div className="relative w-64">
        <ListboxButton className="w-full flex items-center justify-between bg-white border border-slate-200 rounded-xl px-4 py-2.5 text-sm font-medium text-slate-700 shadow-sm hover:border-brand-400 focus:outline-none focus:ring-2 focus:ring-brand-500">
          <span className="flex items-center gap-2">
            <span
              className="w-2.5 h-2.5 rounded-full"
              style={{ backgroundColor: SEGMENT_COLORS[value] || '#6b7280' }}
            />
            {value}
          </span>
          <ChevronDown size={16} className="text-slate-400" />
        </ListboxButton>
        <ListboxOptions className="absolute z-30 mt-1 w-full bg-white rounded-xl shadow-lg border border-slate-100 py-1">
          {SEGMENTS.map((seg) => (
            <ListboxOption
              key={seg}
              value={seg}
              className="flex items-center justify-between px-4 py-2 text-sm cursor-pointer data-[focus]:bg-brand-50 data-[selected]:bg-brand-100"
            >
              <span className="flex items-center gap-2">
                <span
                  className="w-2.5 h-2.5 rounded-full"
                  style={{ backgroundColor: SEGMENT_COLORS[seg] || '#6b7280' }}
                />
                {seg}
              </span>
              {seg === value && <Check size={14} className="text-brand-600" />}
            </ListboxOption>
          ))}
        </ListboxOptions>
      </div>
    </Listbox>
  );
}
