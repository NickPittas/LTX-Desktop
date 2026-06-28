import { useEffect, useState } from 'react'
import { Progress } from './ui/progress'

interface ActiveJob {
  jobId: string
  percent: number
  label: string
  indeterminate?: boolean
}

/**
 * Non-blocking toast that surfaces visual-asset import progress (copy /
 * transcode / finalize) streamed over `asset:importProgress`. Listens globally,
 * so every call site that imports a generated asset — including the three in
 * GenSpace (T2V/I2V/A2V, IC-LoRA, retake) — gets feedback without wiring.
 *
 * Determinate phases render the shared `Progress` primitive; indeterminate
 * phases (e.g. a single-file copy where the fraction is unknowable without a
 * slow read/write loop) render a pulsing bar. Toasts auto-dismiss when the main
 * process emits `done` (including on failure).
 */
export function ImportProgressToast() {
  const [jobs, setJobs] = useState<Record<string, ActiveJob>>({})

  useEffect(() => {
    const unsubscribe = window.electronAPI.onAssetImportProgress((e) => {
      setJobs((prev) => {
        // `done` (success or failure) removes the toast immediately.
        if (e.done) {
          if (!(e.jobId in prev)) return prev
          const next = { ...prev }
          delete next[e.jobId]
          return next
        }
        return {
          ...prev,
          [e.jobId]: {
            jobId: e.jobId,
            percent: e.percent,
            label: e.label,
            indeterminate: e.indeterminate === true,
          },
        }
      })
    })
    return unsubscribe
  }, [])

  const jobList = Object.values(jobs)
  if (jobList.length === 0) return null

  return (
    <div className="absolute bottom-4 right-4 z-50 flex flex-col gap-2 w-72">
      {jobList.map((job) => (
        <div
          key={job.jobId}
          className="rounded-xl bg-zinc-900/95 border border-zinc-700 shadow-2xl p-3 backdrop-blur-md"
        >
          <div className="flex items-center justify-between mb-2 gap-2">
            <span className="text-xs font-medium text-zinc-200 truncate">{job.label}</span>
            <span className="text-[10px] text-zinc-500 tabular-nums shrink-0">
              {job.indeterminate ? '' : `${Math.round(job.percent)}%`}
            </span>
          </div>
          {job.indeterminate ? (
            <div className="h-2 w-full overflow-hidden rounded-full bg-secondary">
              <div className="h-full w-full bg-primary/60 animate-pulse rounded-full" />
            </div>
          ) : (
            <Progress value={job.percent} className="h-2" />
          )}
        </div>
      ))}
    </div>
  )
}
