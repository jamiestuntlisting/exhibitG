import { useState, useCallback } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import type { ExhibitGDetail } from '../api/client'
import { uploadExhibitG, getTranscription } from '../api/client'
import ImageUploader from '../components/ImageUploader'
import CorrectedImageView from '../components/CorrectedImageView'
import HeaderFields from '../components/HeaderFields'
import TranscriptionTable from '../components/TranscriptionTable'
import ExportButton from '../components/ExportButton'

interface UploadPageProps {
  result: ExhibitGDetail | null
  setResult: (result: ExhibitGDetail | null) => void
}

export default function UploadPage({ result, setResult }: UploadPageProps) {
  const [error, setError] = useState<string | null>(null)

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

  // Refetch current result (after edits)
  const { refetch } = useQuery({
    queryKey: ['transcription', result?.id],
    queryFn: () => result ? getTranscription(result.id) : null,
    enabled: false,
  })

  const handleUpload = useCallback((file: File) => {
    setError(null)
    uploadMutation.mutate(file)
  }, [uploadMutation])

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

  return (
    <div className="space-y-6">
      {/* Upload section - show when no result */}
      {!result && (
        <ImageUploader
          onUpload={handleUpload}
          isProcessing={uploadMutation.isPending}
        />
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
