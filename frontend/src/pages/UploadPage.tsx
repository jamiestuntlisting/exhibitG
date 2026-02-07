import { useState, useCallback } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import type { ExhibitGDetail, GridCorrectionResult } from '../api/client'
import { uploadExhibitG, getTranscription, testGridCorrection } from '../api/client'
import ImageUploader from '../components/ImageUploader'
import CorrectedImageView from '../components/CorrectedImageView'
import HeaderFields from '../components/HeaderFields'
import TranscriptionTable from '../components/TranscriptionTable'
import ExportButton from '../components/ExportButton'
import GridCorrectionSteps from '../components/GridCorrectionSteps'

interface UploadPageProps {
  result: ExhibitGDetail | null
  setResult: (result: ExhibitGDetail | null) => void
}

export default function UploadPage({ result, setResult }: UploadPageProps) {
  const [error, setError] = useState<string | null>(null)
  const [gridCorrectionResult, setGridCorrectionResult] = useState<GridCorrectionResult | null>(null)
  const [testMode, setTestMode] = useState(false)

  // Mutation for uploading
  const uploadMutation = useMutation({
    mutationFn: (file: File) => uploadExhibitG(file),
    onSuccess: (data) => {
      setResult(data)
      setError(null)
    },
    onError: (err: Error) => {
      setError(err.message)
    },
  })

  // Mutation for grid correction testing
  const gridCorrectionMutation = useMutation({
    mutationFn: (file: File) => testGridCorrection(file),
    onSuccess: (data) => {
      setGridCorrectionResult(data)
      setError(null)
    },
    onError: (err: Error) => {
      setError(err.message)
    },
  })

  // Refetch current result (after edits)
  const { refetch } = useQuery({
    queryKey: ['transcription', result?.id],
    queryFn: () => result ? getTranscription(result.id) : null,
    enabled: false,
  })

  const handleUpload = useCallback((file: File) => {
    setError(null)
    if (testMode) {
      setGridCorrectionResult(null)
      gridCorrectionMutation.mutate(file)
    } else {
      uploadMutation.mutate(file)
    }
  }, [uploadMutation, gridCorrectionMutation, testMode])

  const handleRefresh = useCallback(async () => {
    if (!result) return
    try {
      const updated = await getTranscription(result.id)
      setResult(updated)
    } catch (err) {
      console.error('Failed to refresh:', err)
    }
  }, [result, setResult])

  const unresolvedCount = result?.disagreements.filter(d => !d.is_resolved).length || 0

  const isProcessing = uploadMutation.isPending || gridCorrectionMutation.isPending

  return (
    <div className="space-y-6">
      {/* Upload section - show when no result */}
      {!result && !gridCorrectionResult && (
        <>
          {/* Test Mode Toggle */}
          <div className="flex items-center justify-end gap-4 px-2">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={testMode}
                onChange={(e) => setTestMode(e.target.checked)}
                className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <span className="text-sm text-gray-600">
                Test Grid Correction (show steps)
              </span>
            </label>
          </div>

          <ImageUploader
            onUpload={handleUpload}
            isProcessing={isProcessing}
          />
        </>
      )}

      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center gap-2">
            <span className="text-red-600 text-lg">&#9888;</span>
            <div>
              <p className="text-sm font-medium text-red-800">Processing Error</p>
              <p className="text-sm text-red-600 mt-0.5">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Grid Correction Test Results */}
      {gridCorrectionResult && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-gray-800">
              Grid Correction Analysis Results
            </h2>
            <button
              onClick={() => {
                setGridCorrectionResult(null)
                setTestMode(false)
              }}
              className="text-sm px-3 py-1.5 rounded bg-gray-100 hover:bg-gray-200 text-gray-600"
            >
              &larr; Upload Another
            </button>
          </div>
          <GridCorrectionSteps
            sessionId={gridCorrectionResult.session_id}
            filename={gridCorrectionResult.filename}
            steps={gridCorrectionResult.steps}
          />
        </div>
      )}

      {/* Results */}
      {result && (
        <>
          {/* Corrected image */}
          <CorrectedImageView
            correctedImageUrl={result.corrected_image_url}
            originalImageUrl={result.original_image_url}
            skewAngle={result.skew_angle}
            filename={result.original_filename}
          />

          {/* Header fields */}
          <HeaderFields exhibit={result} onUpdate={handleRefresh} />

          {/* Action bar */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <ExportButton
                exhibitId={result.id}
                disabled={unresolvedCount > 0}
              />
              {unresolvedCount > 0 && (
                <span className="text-xs text-amber-600">
                  Resolve {unresolvedCount} disagreement{unresolvedCount !== 1 ? 's' : ''} before exporting
                </span>
              )}
            </div>
            <button
              onClick={() => refetch()}
              className="text-xs text-gray-500 hover:text-gray-700"
            >
              Refresh
            </button>
          </div>

          {/* Transcription table */}
          <TranscriptionTable exhibit={result} onUpdate={handleRefresh} />
        </>
      )}
    </div>
  )
}
