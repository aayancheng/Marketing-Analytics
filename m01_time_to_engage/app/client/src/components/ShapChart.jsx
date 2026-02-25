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
