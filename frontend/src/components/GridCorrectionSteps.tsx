import { useState } from 'react'

interface StepData {
  [key: string]: unknown
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
              <button
                onClick={() => setExpandedImage(!expandedImage)}
                className="absolute top-2 right-2 bg-white/90 hover:bg-white px-2 py-1 rounded text-xs text-gray-600 shadow"
              >
                {expandedImage ? 'Collapse' : 'Expand'}
              </button>
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
