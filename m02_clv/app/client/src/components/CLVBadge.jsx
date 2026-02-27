import { SEGMENT_COLORS } from '../lib/constants';

export default function CLVBadge({ segment }) {
  if (!segment) return null;

  const color = SEGMENT_COLORS[segment] || '#6b7280';

  return (
    <span
      className="inline-flex items-center px-3 py-1 rounded-full text-sm font-semibold text-white"
      style={{ backgroundColor: color }}
    >
      {segment}
    </span>
  );
}
