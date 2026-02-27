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
