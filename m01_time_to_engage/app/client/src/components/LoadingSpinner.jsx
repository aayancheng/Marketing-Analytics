export default function LoadingSpinner({ size = 'md' }) {
  const sizeClass = size === 'sm' ? 'w-5 h-5' : size === 'lg' ? 'w-10 h-10' : 'w-7 h-7';
  return (
    <div className="flex items-center justify-center py-8">
      <div className={`${sizeClass} rounded-full border-2 border-brand-200 border-t-brand-600 animate-spin`} />
    </div>
  );
}
