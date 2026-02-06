import { useState, useEffect } from 'react'
import type { CellDisagreement, EngineReading } from '../api/client'
import { getEngineReadings, resolveDisagreement } from '../api/client'

interface ReviewModalProps {
  exhibitId: number
  disagreement: CellDisagreement
  onClose: () => void
  onResolved: () => void
}

const ENGINE_LABELS: Record<string, string> = {
  paddleocr: 'PaddleOCR',
  easyocr: 'EasyOCR',
  claude: 'Claude Vision',
}

const ENGINE_COLORS: Record<string, string> = {
  paddleocr: 'bg-purple-100 text-purple-800 border-purple-200',
  easyocr: 'bg-green-100 text-green-800 border-green-200',
  claude: 'bg-orange-100 text-orange-800 border-orange-200',
}

export default function ReviewModal({ exhibitId, disagreement, onClose, onResolved }: ReviewModalProps) {
  const [readings, setReadings] = useState<EngineReading[]>([])
  const [selectedValue, setSelectedValue] = useState('')
  const [customValue, setCustomValue] = useState('')
  const [useCustom, setUseCustom] = useState(false)
  const [isResolving, setIsResolving] = useState(false)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    loadReadings()
  }, [disagreement])

  const loadReadings = async () => {
    setIsLoading(true)
    try {
      const data = await getEngineReadings(exhibitId, disagreement.row_number, disagreement.column_name)
      setReadings(data)

      // Default to first non-empty value
      const firstValue = data.find(r => r.value.trim())?.value || ''
      setSelectedValue(firstValue)
    } catch (err) {
      console.error('Failed to load readings:', err)
    } finally {
      setIsLoading(false)
    }
  }

  const handleResolve = async () => {
    const value = useCustom ? customValue : selectedValue
    if (!value.trim() && !confirm('Are you sure you want to set this cell to empty?')) {
      return
    }

    setIsResolving(true)
    try {
      await resolveDisagreement(exhibitId, disagreement.id, value)
      onResolved()
    } catch (err) {
      console.error('Failed to resolve:', err)
      alert('Failed to resolve disagreement')
    } finally {
      setIsResolving(false)
    }
  }

  const columnLabel = disagreement.column_name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase())

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={onClose}>
      <div
        className="bg-white rounded-xl shadow-2xl w-full max-w-lg mx-4"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-gray-900">Resolve Disagreement</h3>
              <p className="text-sm text-gray-500 mt-0.5">
                Row {disagreement.row_number + 1} &middot; {columnLabel}
              </p>
            </div>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 text-xl leading-none"
            >
              &times;
            </button>
          </div>
        </div>

        {/* Body */}
        <div className="px-6 py-4 space-y-4">
          {isLoading ? (
            <div className="text-center py-8 text-gray-500">Loading engine readings...</div>
          ) : (
            <>
              <p className="text-sm text-gray-600">
                The OCR engines produced different readings for this cell. Select the correct value or enter a custom one:
              </p>

              {/* Engine readings */}
              <div className="space-y-2">
                {readings.map((reading) => (
                  <label
                    key={reading.engine}
                    className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                      !useCustom && selectedValue === reading.value
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                    onClick={() => {
                      setSelectedValue(reading.value)
                      setUseCustom(false)
                    }}
                  >
                    <input
                      type="radio"
                      name="engine_value"
                      checked={!useCustom && selectedValue === reading.value}
                      onChange={() => {
                        setSelectedValue(reading.value)
                        setUseCustom(false)
                      }}
                      className="text-blue-600"
                    />
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className={`text-xs px-2 py-0.5 rounded border font-medium ${ENGINE_COLORS[reading.engine] || 'bg-gray-100 text-gray-700'}`}>
                          {ENGINE_LABELS[reading.engine] || reading.engine}
                        </span>
                        {reading.confidence > 0 && (
                          <span className="text-xs text-gray-400">
                            {(reading.confidence * 100).toFixed(0)}% conf
                          </span>
                        )}
                      </div>
                      <div className="mt-1 text-lg font-mono text-gray-900">
                        {reading.value || <span className="text-gray-400 italic text-sm">(empty)</span>}
                      </div>
                    </div>
                  </label>
                ))}
              </div>

              {/* Custom value */}
              <div
                className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                  useCustom ? 'border-blue-500 bg-blue-50' : 'border-gray-200'
                }`}
                onClick={() => setUseCustom(true)}
              >
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="radio"
                    name="engine_value"
                    checked={useCustom}
                    onChange={() => setUseCustom(true)}
                    className="text-blue-600"
                  />
                  <span className="text-sm font-medium text-gray-700">Custom value:</span>
                </label>
                {useCustom && (
                  <input
                    type="text"
                    value={customValue}
                    onChange={(e) => setCustomValue(e.target.value)}
                    autoFocus
                    placeholder="Type the correct value..."
                    className="mt-2 w-full border border-gray-300 rounded px-3 py-2 text-lg font-mono focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                )}
              </div>
            </>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-gray-200 flex justify-end gap-3">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50"
          >
            Cancel
          </button>
          <button
            onClick={handleResolve}
            disabled={isResolving || isLoading}
            className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isResolving ? 'Resolving...' : 'Resolve'}
          </button>
        </div>
      </div>
    </div>
  )
}
