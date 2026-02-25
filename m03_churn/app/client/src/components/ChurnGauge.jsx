import { interpolateRiskColor, getRiskTier, RISK_COLORS } from '../lib/constants';

// Color zones along the arc (each zone covers a 36-degree segment of the 180° arc)
const ZONES = [
  { from: 0.0, to: 0.2, color: '#22c55e' },   // green
  { from: 0.2, to: 0.4, color: '#84cc16' },   // lime
  { from: 0.4, to: 0.6, color: '#eab308' },   // yellow/amber
  { from: 0.6, to: 0.8, color: '#f97316' },   // orange
  { from: 0.8, to: 1.0, color: '#ef4444' },   // red
];

/**
 * Convert a 0-1 score to an angle on the semicircular arc.
 * Arc goes from 180° (leftmost, score=0) to 0° (rightmost, score=1).
 * In SVG coordinates: leftmost = angle 180°, rightmost = 0°.
 */
function scoreToAngle(score) {
  // Map score 0→180deg, score 1→0deg (left to right sweep over top)
  return Math.PI - score * Math.PI;
}

function polarToCartesian(cx, cy, r, angleRad) {
  return {
    x: cx + r * Math.cos(angleRad),
    y: cy - r * Math.sin(angleRad),
  };
}

function arcPath(cx, cy, r, startScore, endScore) {
  const startAngle = scoreToAngle(endScore);   // reversed because sweep direction
  const endAngle = scoreToAngle(startScore);
  const start = polarToCartesian(cx, cy, r, startAngle);
  const end = polarToCartesian(cx, cy, r, endAngle);
  const largeArc = endScore - startScore > 0.5 ? 1 : 0;
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 0 ${end.x} ${end.y}`;
}

export default function ChurnGauge({ score = 0, size = 200 }) {
  const clampedScore = Math.max(0, Math.min(1, score));
  const cx = size / 2;
  const cy = size * 0.6;  // slightly below center so arc fits nicely
  const outerR = size * 0.42;
  const innerR = size * 0.30;
  const strokeWidth = outerR - innerR;
  const midR = (outerR + innerR) / 2;

  const riskTier = getRiskTier(clampedScore);
  const tierColor = RISK_COLORS[riskTier] || '#94a3b8';

  // Needle: points from center toward the score position
  const needleAngle = scoreToAngle(clampedScore);
  const needleLength = outerR + 4;
  const needleTip = polarToCartesian(cx, cy, needleLength, needleAngle);
  const needleBase1 = polarToCartesian(cx, cy, 8, needleAngle + Math.PI / 2);
  const needleBase2 = polarToCartesian(cx, cy, 8, needleAngle - Math.PI / 2);

  const pct = Math.round(clampedScore * 100);

  return (
    <div className="flex flex-col items-center">
      <svg
        width={size}
        height={size * 0.72}
        viewBox={`0 0 ${size} ${size * 0.72}`}
        aria-label={`Churn risk gauge showing ${pct}%`}
      >
        {/* Background gray arc */}
        <path
          d={arcPath(cx, cy, midR, 0, 1)}
          fill="none"
          stroke="#e2e8f0"
          strokeWidth={strokeWidth}
          strokeLinecap="butt"
        />

        {/* Colored zone arcs */}
        {ZONES.map((zone, i) => (
          <path
            key={i}
            d={arcPath(cx, cy, midR, zone.from, zone.to)}
            fill="none"
            stroke={zone.color}
            strokeWidth={strokeWidth}
            strokeLinecap="butt"
            opacity={0.25}
          />
        ))}

        {/* Filled colored arc up to score */}
        {clampedScore > 0 && (
          <path
            d={arcPath(cx, cy, midR, 0, clampedScore)}
            fill="none"
            stroke={interpolateRiskColor(clampedScore)}
            strokeWidth={strokeWidth}
            strokeLinecap="butt"
          />
        )}

        {/* Zone tick marks */}
        {[0.2, 0.4, 0.6, 0.8].map((tick) => {
          const angle = scoreToAngle(tick);
          const inner = polarToCartesian(cx, cy, innerR - 2, angle);
          const outer = polarToCartesian(cx, cy, outerR + 2, angle);
          return (
            <line
              key={tick}
              x1={inner.x}
              y1={inner.y}
              x2={outer.x}
              y2={outer.y}
              stroke="#fff"
              strokeWidth={2}
            />
          );
        })}

        {/* Needle */}
        <polygon
          points={`${needleTip.x},${needleTip.y} ${needleBase1.x},${needleBase1.y} ${needleBase2.x},${needleBase2.y}`}
          fill={tierColor}
          opacity={0.9}
        />

        {/* Center hub circle */}
        <circle cx={cx} cy={cy} r={9} fill={tierColor} />
        <circle cx={cx} cy={cy} r={5} fill="white" />

        {/* Score label */}
        <text
          x={cx}
          y={cy - outerR * 0.45}
          textAnchor="middle"
          fontSize={size * 0.13}
          fontWeight="bold"
          fill="#1e293b"
        >
          {pct}%
        </text>

        {/* Min/max labels */}
        <text
          x={cx - midR - strokeWidth / 2}
          y={cy + 14}
          textAnchor="middle"
          fontSize={size * 0.06}
          fill="#94a3b8"
        >
          0%
        </text>
        <text
          x={cx + midR + strokeWidth / 2}
          y={cy + 14}
          textAnchor="middle"
          fontSize={size * 0.06}
          fill="#94a3b8"
        >
          100%
        </text>
      </svg>

      {/* Risk tier badge */}
      <span
        className="mt-1 inline-block px-3 py-1 rounded-full text-sm font-semibold text-white"
        style={{ backgroundColor: tierColor }}
      >
        {riskTier}
      </span>
    </div>
  );
}
