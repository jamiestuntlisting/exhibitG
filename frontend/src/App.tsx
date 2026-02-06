import { useState } from 'react'
import type { ExhibitGDetail } from './api/client'
import UploadPage from './pages/UploadPage'

function App() {
  const [result, setResult] = useState<ExhibitGDetail | null>(null)

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="max-w-[1800px] mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold text-gray-900">Exhibit G Transcriber</h1>
            <p className="text-sm text-gray-500">SAG-AFTRA Actors Production Time Report</p>
          </div>
          {result && (
            <button
              onClick={() => setResult(null)}
              className="text-sm text-blue-600 hover:text-blue-800 font-medium"
            >
              + New Upload
            </button>
          )}
        </div>
      </header>

      <main className="max-w-[1800px] mx-auto px-6 py-6">
        <UploadPage result={result} setResult={setResult} />
      </main>
    </div>
  )
}

export default App
