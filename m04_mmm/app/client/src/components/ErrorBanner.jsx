import { AlertTriangle } from 'lucide-react';

export default function ErrorBanner({ message }) {
  if (!message) return null;
  return (
    <div className="flex items-center gap-3 bg-red-50 border border-red-200 text-red-700 rounded-xl px-4 py-3 mb-4">
      <AlertTriangle size={18} />
      <p className="text-sm">{message}</p>
    </div>
  );
}
