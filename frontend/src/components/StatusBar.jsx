export default function StatusBar({ error }) {
  return (
    <div className="mt-8">
      {error && (
        <div className="bg-red-900 border border-red-700 text-red-200 p-4 rounded-lg mb-4">
          <p className="font-bold">An error occurred:</p>
          <p>{error}</p>
        </div>
      )}
    </div>
  )
}