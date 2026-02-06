import { getExportCsvUrl } from '../api/client'

interface ExportButtonProps {
  exhibitId: number
  disabled?: boolean
}

export default function ExportButton({ exhibitId, disabled }: ExportButtonProps) {
  const handleExport = () => {
    window.open(getExportCsvUrl(exhibitId), '_blank')
  }

  return (
    <button
      onClick={handleExport}
      disabled={disabled}
      className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
      title={disabled ? 'Resolve all disagreements before exporting' : 'Download as CSV'}
    >
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
        />
      </svg>
      Export CSV
    </button>
  )
}
