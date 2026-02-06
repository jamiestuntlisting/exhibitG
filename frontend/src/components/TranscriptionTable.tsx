import { useState, useMemo } from 'react'
import type { ExhibitGDetail, ExhibitGRow, CellDisagreement } from '../api/client'
import { COLUMN_DEFS, updateRow } from '../api/client'
import ReviewModal from './ReviewModal'

interface TranscriptionTableProps {
  exhibit: ExhibitGDetail
  onUpdate: () => void
}

export default function TranscriptionTable({ exhibit, onUpdate }: TranscriptionTableProps) {
  const [editingCell, setEditingCell] = useState<{ rowId: number; column: string } | null>(null)
  const [editValue, setEditValue] = useState('')
  const [reviewDisagreement, setReviewDisagreement] = useState<CellDisagreement | null>(null)
  const [isSaving, setIsSaving] = useState(false)

  // Build a lookup for disagreements by (row_number, column_name)
  const disagreementMap = useMemo(() => {
    const map = new Map<string, CellDisagreement>()
    for (const d of exhibit.disagreements) {
      if (!d.is_resolved) {
        map.set(`${d.row_number}:${d.column_name}`, d)
      }
    }
    return map
  }, [exhibit.disagreements])

  const unresolvedCount = exhibit.disagreements.filter(d => !d.is_resolved).length

  // Group columns for header rendering
  const columnGroups = useMemo(() => {
    const groups: { label: string; columns: typeof COLUMN_DEFS; span: number }[] = []
    let currentGroup = ''

    for (const col of COLUMN_DEFS) {
      const group = col.group || ''
      if (group !== currentGroup) {
        groups.push({ label: group, columns: [col], span: 1 })
        currentGroup = group
      } else if (groups.length > 0 && group) {
        groups[groups.length - 1].columns.push(col)
        groups[groups.length - 1].span++
      } else {
        groups.push({ label: '', columns: [col], span: 1 })
      }
    }
    return groups
  }, [])

  const handleCellClick = (row: ExhibitGRow, columnKey: string) => {
    // Check if this cell has an unresolved disagreement
    const disagreement = disagreementMap.get(`${row.row_number}:${columnKey}`)
    if (disagreement) {
      setReviewDisagreement(disagreement)
      return
    }

    // Start inline editing
    const value = (row as Record<string, unknown>)[columnKey] as string || ''
    setEditingCell({ rowId: row.id, column: columnKey })
    setEditValue(value)
  }

  const handleSave = async () => {
    if (!editingCell) return

    setIsSaving(true)
    try {
      await updateRow(exhibit.id, editingCell.rowId, {
        [editingCell.column]: editValue,
      } as Partial<ExhibitGRow>)
      onUpdate()
    } catch (err) {
      console.error('Failed to save:', err)
    } finally {
      setIsSaving(false)
      setEditingCell(null)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSave()
    } else if (e.key === 'Escape') {
      setEditingCell(null)
    }
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      {/* Table header bar */}
      <div className="flex items-center justify-between px-4 py-3 bg-gray-50 border-b border-gray-200">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-semibold text-gray-700">Transcription Results</h3>
          <span className="text-xs text-gray-500">
            {exhibit.rows.length} row{exhibit.rows.length !== 1 ? 's' : ''}
          </span>
          {unresolvedCount > 0 && (
            <span className="text-xs bg-amber-100 text-amber-800 px-2 py-0.5 rounded-full font-medium">
              {unresolvedCount} cell{unresolvedCount !== 1 ? 's' : ''} need review
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className={`text-xs px-2 py-1 rounded-full font-medium ${
            exhibit.status === 'complete'
              ? 'bg-green-100 text-green-800'
              : exhibit.status === 'review'
                ? 'bg-amber-100 text-amber-800'
                : 'bg-blue-100 text-blue-800'
          }`}>
            {exhibit.status === 'complete' ? 'Complete' :
             exhibit.status === 'review' ? 'Needs Review' : 'Processing'}
          </span>
        </div>
      </div>

      {/* Scrollable table */}
      <div className="table-container">
        <table className="w-full text-xs border-collapse">
          {/* Group header row */}
          <thead>
            <tr className="bg-gray-100">
              {columnGroups.map((group, i) => (
                <th
                  key={i}
                  colSpan={group.span}
                  className="px-1 py-1 text-center text-[10px] font-bold text-gray-600 uppercase tracking-wider border-b border-r border-gray-200"
                >
                  {group.label}
                </th>
              ))}
            </tr>
            {/* Column header row */}
            <tr className="bg-gray-50">
              {COLUMN_DEFS.map((col) => (
                <th
                  key={col.key}
                  className="px-1 py-1.5 text-center text-[10px] font-semibold text-gray-700 border-b border-r border-gray-200 whitespace-nowrap"
                  style={{ minWidth: col.width }}
                >
                  {col.label}
                </th>
              ))}
            </tr>
          </thead>

          <tbody>
            {exhibit.rows.length === 0 ? (
              <tr>
                <td colSpan={COLUMN_DEFS.length} className="text-center py-8 text-gray-500">
                  No rows detected
                </td>
              </tr>
            ) : (
              exhibit.rows.map((row) => (
                <tr key={row.id} className="hover:bg-gray-50 border-b border-gray-100">
                  {COLUMN_DEFS.map((col) => {
                    const value = (row as Record<string, unknown>)[col.key] as string || ''
                    const isEditing = editingCell?.rowId === row.id && editingCell?.column === col.key
                    const disagreement = disagreementMap.get(`${row.row_number}:${col.key}`)
                    const isFlagged = !!disagreement

                    return (
                      <td
                        key={col.key}
                        className={`px-1 py-0.5 border-r border-gray-100 ${
                          isFlagged ? 'cell-flagged' : ''
                        }`}
                        style={{ minWidth: col.width }}
                      >
                        {isEditing ? (
                          <input
                            type="text"
                            value={editValue}
                            onChange={(e) => setEditValue(e.target.value)}
                            onKeyDown={handleKeyDown}
                            onBlur={handleSave}
                            autoFocus
                            disabled={isSaving}
                            className="w-full text-xs border border-blue-400 rounded px-1 py-0.5 focus:outline-none focus:ring-1 focus:ring-blue-500 font-mono"
                          />
                        ) : (
                          <div
                            onClick={() => handleCellClick(row, col.key)}
                            className={`cell-editable text-xs font-mono ${
                              isFlagged ? 'cursor-pointer' : ''
                            }`}
                            title={isFlagged ? 'Click to review disagreement' : 'Click to edit'}
                          >
                            {value}
                            {isFlagged && (
                              <span className="ml-1 text-amber-500 text-[10px]" title="Disagreement">
                                &#9888;
                              </span>
                            )}
                          </div>
                        )}
                      </td>
                    )
                  })}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Review modal */}
      {reviewDisagreement && (
        <ReviewModal
          exhibitId={exhibit.id}
          disagreement={reviewDisagreement}
          onClose={() => setReviewDisagreement(null)}
          onResolved={() => {
            setReviewDisagreement(null)
            onUpdate()
          }}
        />
      )}
    </div>
  )
}
