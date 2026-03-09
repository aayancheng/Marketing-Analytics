import { CHANNEL_COLORS, CHANNEL_LABELS } from '../lib/constants';

export default function ChannelLegend({ channels = null, showBase = false }) {
  const entries = Object.entries(CHANNEL_COLORS).filter(([key]) => {
    if (key === 'base') return showBase;
    if (channels) return channels.includes(key);
    return true;
  });

  return (
    <div className="flex flex-wrap gap-3 mt-3">
      {entries.map(([key, color]) => (
        <div key={key} className="flex items-center gap-1.5 text-xs text-slate-600">
          <span
            className="w-3 h-3 rounded-full inline-block"
            style={{ backgroundColor: color }}
          />
          {CHANNEL_LABELS[key] || key}
        </div>
      ))}
    </div>
  );
}
