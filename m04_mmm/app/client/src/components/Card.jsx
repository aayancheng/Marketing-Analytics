export default function Card({ children, className = '', wide = false }) {
  return (
    <div className={`bg-white rounded-2xl shadow-md p-5 ${wide ? 'col-span-full' : ''} ${className}`}>
      {children}
    </div>
  );
}
