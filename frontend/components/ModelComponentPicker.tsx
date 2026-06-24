import { FolderOpen } from 'lucide-react'
import { useCallback } from 'react'
import { Button } from './ui/button'

// ponytail: presets cover safetensors & GGUF. Extend if new model formats added.
export const MODEL_FILE_FILTERS = {
  safetensors: [{ name: 'Safetensors', extensions: ['safetensors'] }],
  gguf: [{ name: 'GGUF', extensions: ['gguf'] }],
  all: [{ name: 'Model Files', extensions: ['safetensors', 'gguf', 'bin', 'pt', 'pth', 'ckpt'] }],
}

export interface ModelComponentPickerProps {
  value: string
  onChange: (value: string) => void
  label: string
  placeholder?: string
  dialogTitle?: string
  filters?: { name: string; extensions: string[] }[]
  pickDirectory?: boolean
}

export function ModelComponentPicker({
  value,
  onChange,
  label,
  placeholder = '',
  dialogTitle,
  filters,
  pickDirectory = false,
}: ModelComponentPickerProps) {
  const handleBrowse = useCallback(async () => {
    if (pickDirectory) {
      const dir = await window.electronAPI.showOpenDirectoryDialog({
        title: dialogTitle ?? `Select ${label} directory`,
      })
      if (dir) onChange(dir)
    } else {
      const paths = await window.electronAPI.showOpenFileDialog({
        title: dialogTitle ?? `Select ${label}`,
        filters,
      })
      if (paths && paths.length > 0) onChange(paths[0])
    }
  }, [pickDirectory, dialogTitle, label, filters, onChange])

  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-sm font-medium text-foreground">{label}</label>
      <div className="flex gap-2">
        <input
          type="text"
          value={value}
          onChange={e => onChange(e.target.value)}
          placeholder={placeholder}
          className="flex-1 h-9 rounded-md border border-border bg-background px-3 text-sm text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
        />
        <Button
          type="button"
          variant="outline"
          size="icon"
          onClick={handleBrowse}
          aria-label={`Browse for ${label}`}
        >
          <FolderOpen className="h-4 w-4" />
        </Button>
      </div>
    </div>
  )
}
