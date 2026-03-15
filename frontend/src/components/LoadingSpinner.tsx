export default function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center py-12">
      <div className="relative w-12 h-12">
        <div className="absolute inset-0 rounded-full border-4 border-dark-700"></div>
        <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-accent-primary animate-spin"></div>
      </div>
      <span className="ml-4 text-gray-400">Analyzing...</span>
    </div>
  )
}
