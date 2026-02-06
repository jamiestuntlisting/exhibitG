import { useState } from 'react'

interface CorrectedImageViewProps {
  correctedImageUrl: string
  originalImageUrl: string
  skewAngle: number | null
  filename: string
}

export default function CorrectedImageView({
  correctedImageUrl,
  originalImageUrl,
  skewAngle,
  filename,
}: CorrectedImageViewProps) {
  const [showOriginal, setShowOriginal] = useState(false)
  const [isExpanded, setIsExpanded] = useState(false)

  const imageUrl = showOriginal ? originalImageUrl : correctedImageUrl

  return (
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 bg-gray-50 border-b border-gray-200">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-semibold text-gray-700">
            {showOriginal ? 'Original Image' : 'Corrected Image'}
          </h3>
          {skewAngle !== null && skewAngle !== 0 && (
            <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full">
              Skew corrected: {skewAngle.toFixed(2)}&deg;
            </span>
          )}
          <span className="text-xs text-gray-500">{filename}</span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowOriginal(!showOriginal)}
            className="text-xs px-3 py-1 rounded border border-gray-300 bg-white hover:bg-gray-50 text-gray-600"
          >
            {showOriginal ? 'Show Corrected' : 'Show Original'}
          </button>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="text-xs px-3 py-1 rounded border border-gray-300 bg-white hover:bg-gray-50 text-gray-600"
          >
            {isExpanded ? 'Collapse' : 'Expand'}
          </button>
        </div>
      </div>

      <div className={`overflow-auto bg-gray-100 ${isExpanded ? 'max-h-[800px]' : 'max-h-[400px]'}`}>
        <img
          src={imageUrl}
          alt="Exhibit G form"
          className="w-full h-auto"
        />
      </div>
    </div>
  )
}
