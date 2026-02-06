import { useState, useRef, useCallback } from 'react'

interface ImageUploaderProps {
  onUpload: (file: File) => void
  isProcessing: boolean
}

export default function ImageUploader({ onUpload, isProcessing }: ImageUploaderProps) {
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/heic', 'image/heif']
  const ACCEPTED_EXTENSIONS = '.jpg,.jpeg,.png,.heic,.heif'

  const handleFile = useCallback((file: File) => {
    const ext = file.name.toLowerCase().split('.').pop()
    const isValidType = ACCEPTED_TYPES.includes(file.type) ||
      ['jpg', 'jpeg', 'png', 'heic', 'heif'].includes(ext || '')

    if (!isValidType) {
      alert('Please upload a JPG, PNG, or HEIC image file.')
      return
    }

    onUpload(file)
  }, [onUpload])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    const files = e.dataTransfer.files
    if (files.length > 0) {
      handleFile(files[0])
    }
  }, [handleFile])

  const handleClick = () => {
    fileInputRef.current?.click()
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      handleFile(files[0])
    }
  }

  if (isProcessing) {
    return (
      <div className="border-2 border-blue-300 bg-blue-50 rounded-xl p-12 text-center">
        <div className="flex flex-col items-center gap-4">
          <svg className="spinner w-10 h-10 text-blue-600" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
          <div>
            <p className="text-lg font-semibold text-blue-800">Processing Exhibit G...</p>
            <p className="text-sm text-blue-600 mt-1">
              Running OCR engines (PaddleOCR + EasyOCR + Claude Vision). This may take 30-60 seconds.
            </p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div
      className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-colors ${
        isDragging
          ? 'border-blue-500 bg-blue-50'
          : 'border-gray-300 bg-white hover:border-gray-400 hover:bg-gray-50'
      }`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={handleClick}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept={ACCEPTED_EXTENSIONS}
        onChange={handleFileSelect}
        className="hidden"
      />

      <svg className="mx-auto w-12 h-12 text-gray-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
          d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
        />
      </svg>

      <p className="text-lg font-medium text-gray-700">
        Drop an Exhibit G image here or click to browse
      </p>
      <p className="text-sm text-gray-500 mt-2">
        Supports JPG, PNG, and HEIC files
      </p>
    </div>
  )
}
