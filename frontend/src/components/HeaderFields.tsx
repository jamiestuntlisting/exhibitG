import { useState } from 'react'
import type { ExhibitGDetail } from '../api/client'
import { updateHeader } from '../api/client'

interface HeaderFieldsProps {
  exhibit: ExhibitGDetail
  onUpdate: () => void
}

interface FieldDef {
  key: string
  label: string
  width?: string
}

const HEADER_FIELDS: FieldDef[] = [
  { key: 'production_co', label: 'Production Co.', width: 'w-64' },
  { key: 'series_title', label: 'Series Title', width: 'w-64' },
  { key: 'episode_name', label: 'Episode Name', width: 'w-64' },
  { key: 'prod_number', label: 'Prod #', width: 'w-32' },
  { key: 'date', label: 'Date', width: 'w-40' },
  { key: 'day_x_of_y', label: 'Day X of Y', width: 'w-32' },
  { key: 'contact', label: 'Contact', width: 'w-48' },
  { key: 'phone', label: 'Phone', width: 'w-40' },
]

export default function HeaderFields({ exhibit, onUpdate }: HeaderFieldsProps) {
  const [editingField, setEditingField] = useState<string | null>(null)
  const [editValue, setEditValue] = useState('')
  const [isSaving, setIsSaving] = useState(false)

  const handleEdit = (key: string, currentValue: string) => {
    setEditingField(key)
    setEditValue(currentValue)
  }

  const handleSave = async (key: string) => {
    setIsSaving(true)
    try {
      await updateHeader(exhibit.id, { [key]: editValue })
      onUpdate()
    } catch (err) {
      console.error('Failed to update header:', err)
    } finally {
      setIsSaving(false)
      setEditingField(null)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent, key: string) => {
    if (e.key === 'Enter') {
      handleSave(key)
    } else if (e.key === 'Escape') {
      setEditingField(null)
    }
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <h3 className="text-sm font-semibold text-gray-700 mb-3">Header Fields</h3>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {HEADER_FIELDS.map((field) => {
          const value = (exhibit as Record<string, unknown>)[field.key] as string || ''
          const isEditing = editingField === field.key

          return (
            <div key={field.key} className="space-y-1">
              <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                {field.label}
              </label>
              {isEditing ? (
                <div className="flex gap-1">
                  <input
                    type="text"
                    value={editValue}
                    onChange={(e) => setEditValue(e.target.value)}
                    onKeyDown={(e) => handleKeyDown(e, field.key)}
                    onBlur={() => handleSave(field.key)}
                    autoFocus
                    disabled={isSaving}
                    className="flex-1 text-sm border border-blue-400 rounded px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              ) : (
                <div
                  onClick={() => handleEdit(field.key, value)}
                  className="text-sm text-gray-900 border border-transparent hover:border-gray-300 rounded px-2 py-1 cursor-text min-h-[30px] bg-gray-50"
                >
                  {value || <span className="text-gray-400 italic">empty</span>}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
