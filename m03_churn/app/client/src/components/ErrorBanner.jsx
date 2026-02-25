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
