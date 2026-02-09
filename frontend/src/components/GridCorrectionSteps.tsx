import { useState } from 'react'

interface StepData {
  [key: string]: unknown
}

interface OrientationMethod {
  method: string
  orientation: number | null
  confidence: number
  error: string | null
  details: Record<string, unknown>
}

interface OrientationAnalysisData {
  recommended_rotation: number
  overall_confidence: number
  voting_breakdown: Record<string, number>
  methods: OrientationMethod[]
}

function isOrientationAnalysis(data: StepData): boolean {
  return 'methods' in data && 'voting_breakdown' in data && 'recommended_rotation' in data
}

function OrientationAnalysisPanel({ data }: { data: OrientationAnalysisData }) {
  const [expandedMethod, setExpandedMethod] = useState<string | null>(null)

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.7) return 'text-green-600 bg-green-100'
    if (confidence >= 0.4) return 'text-yellow-600 bg-yellow-100'
    return 'text-red-600 bg-red-100'
  }

  const getMethodIcon = (method: string) => {
    switch (method) {
      case 'ocr': return '📝'
      case 'logo': return '🎯'
      case 'template': return '📐'
      case 'claude_vision': return '🤖'
      case 'lines': return '📏'
      default: return '❓'
    }
  }

  const formatRotation = (rotation: number | null) => {
    if (rotation === null) return 'Unknown'
    if (rotation === 0) return 'Correct (0°)'
    return `${rotation}° CCW`
  }

  return (
    <div className="space-y-4">
      {/* Overall Result */}
      <div className="bg-blue-50 rounded-lg p-3 border border-blue-200">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-semibold text-blue-800">Recommended Rotation</span>
          <span className={`px-2 py-0.5 rounded text-xs font-medium ${getConfidenceColor(data.overall_confidence)}`}>
            {(data.overall_confidence * 100).toFixed(0)}% confident
          </span>
        </div>
        <div className="text-2xl font-bold text-blue-900">
          {formatRotation(data.recommended_rotation)}
        </div>
      </div>

      {/* Voting Breakdown */}
      <div>
        <h5 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
          Vote Distribution
        </h5>
        <div className="grid grid-cols-4 gap-1 text-center">
          {Object.entries(data.voting_breakdown).map(([rotation, votes]) => (
            <div
              key={rotation}
              className={`p-2 rounded text-xs ${
                parseInt(rotation) === data.recommended_rotation
                  ? 'bg-blue-100 border-2 border-blue-400'
                  : 'bg-gray-100'
              }`}
            >
              <div className="font-semibold">{rotation}°</div>
              <div className="text-gray-600">{(votes as number).toFixed(2)}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Individual Methods */}
      <div>
        <h5 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
          Detection Methods
        </h5>
        <div className="space-y-2">
          {data.methods.map((method) => (
            <div
              key={method.method}
              className="border border-gray-200 rounded-lg overflow-hidden"
            >
              <button
                onClick={() => setExpandedMethod(
                  expandedMethod === method.method ? null : method.method
                )}
                className="w-full px-3 py-2 flex items-center justify-between bg-gray-50 hover:bg-gray-100 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <span>{getMethodIcon(method.method)}</span>
                  <span className="font-medium text-sm capitalize">
                    {method.method.replace(/_/g, ' ')}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  {method.error ? (
                    <span className="text-xs text-red-500">Error</span>
                  ) : method.orientation !== null ? (
                    <>
                      <span className="text-xs text-gray-600">
                        {formatRotation(method.orientation)}
                      </span>
                      <span className={`px-1.5 py-0.5 rounded text-xs ${getConfidenceColor(method.confidence)}`}>
                        {(method.confidence * 100).toFixed(0)}%
                      </span>
                    </>
                  ) : (
                    <span className="text-xs text-gray-400">No result</span>
                  )}
                  <span className="text-gray-400">
                    {expandedMethod === method.method ? '▼' : '▶'}
                  </span>
                </div>
              </button>

              {expandedMethod === method.method && (
                <div className="px-3 py-2 bg-white border-t border-gray-200 text-xs">
                  {method.error && (
                    <div className="text-red-600 mb-2">
                      <strong>Error:</strong> {method.error}
                    </div>
                  )}
                  <div className="space-y-1 max-h-48 overflow-y-auto">
                    {Object.entries(method.details).map(([key, value]) => (
                      <div key={key} className="flex justify-between">
                        <span className="text-gray-500">{key}:</span>
                        <span className="text-gray-800 font-mono ml-2 text-right max-w-[60%] truncate">
                          {typeof value === 'object'
                            ? JSON.stringify(value).slice(0, 50) + '...'
                            : String(value).slice(0, 50)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

interface CorrectionStep {
  step: number
  name: string
  description: string
  image_url: string
  data: StepData
}

interface GridCorrectionStepsProps {
  sessionId: string
  filename: string
  steps: CorrectionStep[]
  onClose?: () => void
}

export default function GridCorrectionSteps({
  sessionId,
  filename,
  steps,
  onClose,
}: GridCorrectionStepsProps) {
  const [selectedStep, setSelectedStep] = useState<number>(0)
  const [expandedImage, setExpandedImage] = useState(false)
  const [showGridOverlay, setShowGridOverlay] = useState(false)

  const currentStep = steps[selectedStep]

  const formatDataValue = (value: unknown): string => {
    if (typeof value === 'number') {
      return value.toFixed(2)
    }
    if (Array.isArray(value)) {
      if (value.length <= 5) {
        return value.map(v => typeof v === 'number' ? v.toFixed(2) : String(v)).join(', ')
      }
      return `[${value.length} items]`
    }
    return String(value)
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-gradient-to-r from-blue-50 to-indigo-50 border-b border-gray-200">
        <div>
          <h3 className="text-sm font-semibold text-gray-700">
            Grid Correction Analysis
          </h3>
          <p className="text-xs text-gray-500 mt-0.5">
            Session: {sessionId} &bull; {filename}
          </p>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 text-xl leading-none"
          >
            &times;
          </button>
        )}
      </div>

      {/* Step Navigation */}
      <div className="flex overflow-x-auto bg-gray-50 border-b border-gray-200 px-2 py-2 gap-1">
        {steps.map((step, idx) => (
          <button
            key={step.step}
            onClick={() => setSelectedStep(idx)}
            className={`flex-shrink-0 px-3 py-1.5 text-xs rounded-md transition-colors ${
              idx === selectedStep
                ? 'bg-blue-600 text-white font-medium'
                : 'bg-white border border-gray-200 text-gray-600 hover:bg-gray-100'
            }`}
          >
            {step.step}. {step.name.replace(/_/g, ' ')}
          </button>
        ))}
      </div>

      {/* Current Step Content */}
      {currentStep && (
        <div className="flex flex-col lg:flex-row">
          {/* Image */}
          <div className={`${expandedImage ? 'lg:w-3/4' : 'lg:w-2/3'} p-4 bg-gray-100`}>
            <div className="relative">
              <img
                src={currentStep.image_url}
                alt={currentStep.description}
                className={`w-full h-auto rounded-lg shadow-sm border border-gray-200 ${
                  expandedImage ? 'max-h-[800px]' : 'max-h-[500px]'
                } object-contain bg-white`}
              />
              {/* Grid overlay */}
              {showGridOverlay && (
                <svg
                  className="absolute inset-0 w-full h-full pointer-events-none"
                  viewBox="0 0 100 100"
                  preserveAspectRatio="none"
                >
                  {/* Vertical lines (10 divisions = 11 lines) */}
                  {[...Array(11)].map((_, i) => (
                    <line
                      key={`v-${i}`}
                      x1={i * 10}
                      y1={0}
                      x2={i * 10}
                      y2={100}
                      stroke="rgba(255, 0, 0, 0.5)"
                      strokeWidth="0.3"
                    />
                  ))}
                  {/* Horizontal lines (10 divisions = 11 lines) */}
                  {[...Array(11)].map((_, i) => (
                    <line
                      key={`h-${i}`}
                      x1={0}
                      y1={i * 10}
                      x2={100}
                      y2={i * 10}
                      stroke="rgba(255, 0, 0, 0.5)"
                      strokeWidth="0.3"
                    />
                  ))}
                </svg>
              )}
              {/* Controls */}
              <div className="absolute top-2 right-2 flex items-center gap-2">
                <label className="flex items-center gap-1 bg-white/90 hover:bg-white px-2 py-1 rounded text-xs text-gray-600 shadow cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showGridOverlay}
                    onChange={(e) => setShowGridOverlay(e.target.checked)}
                    className="w-3 h-3"
                  />
                  Grid
                </label>
                <button
                  onClick={() => setExpandedImage(!expandedImage)}
                  className="bg-white/90 hover:bg-white px-2 py-1 rounded text-xs text-gray-600 shadow"
                >
                  {expandedImage ? 'Collapse' : 'Expand'}
                </button>
              </div>
            </div>
          </div>

          {/* Step Details */}
          <div className={`${expandedImage ? 'lg:w-1/4' : 'lg:w-1/3'} p-4 border-t lg:border-t-0 lg:border-l border-gray-200`}>
            <div className="mb-4">
              <h4 className="text-sm font-semibold text-gray-800">
                Step {currentStep.step}: {currentStep.name.replace(/_/g, ' ')}
              </h4>
              <p className="text-sm text-gray-600 mt-1">{currentStep.description}</p>
            </div>

            {/* Step Data */}
            {Object.keys(currentStep.data).length > 0 && (
              <div>
                {/* Special rendering for orientation analysis step */}
                {currentStep.name === 'orientation_analysis' && isOrientationAnalysis(currentStep.data) ? (
                  <OrientationAnalysisPanel data={currentStep.data as unknown as OrientationAnalysisData} />
                ) : (
                  <>
                    <h5 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
                      Measurements
                    </h5>
                    <div className="space-y-2">
                      {Object.entries(currentStep.data).map(([key, value]) => (
                        <div key={key} className="flex justify-between items-start text-sm">
                          <span className="text-gray-500">{key.replace(/_/g, ' ')}:</span>
                          <span className="text-gray-800 font-mono text-right ml-2">
                            {formatDataValue(value)}
                            {key.includes('angle') || key.includes('rotation') ||
                             key.includes('keystone') || key.includes('avg') ||
                             key.includes('std') || key.includes('dev') ? '°' : ''}
                          </span>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            )}

            {/* Navigation buttons */}
            <div className="flex gap-2 mt-6">
              <button
                onClick={() => setSelectedStep(Math.max(0, selectedStep - 1))}
                disabled={selectedStep === 0}
                className="flex-1 px-3 py-2 text-sm rounded bg-gray-100 hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                &larr; Previous
              </button>
              <button
                onClick={() => setSelectedStep(Math.min(steps.length - 1, selectedStep + 1))}
                disabled={selectedStep === steps.length - 1}
                className="flex-1 px-3 py-2 text-sm rounded bg-gray-100 hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Next &rarr;
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
