# frontend/components/ui

## Responsibility

Five headless-ish, tailwind-styled form primitives that the rest of the renderer composes: `button.tsx`, `select.tsx`, `textarea.tsx`, `progress.tsx`, `tooltip.tsx`. They are the only styling-aware building blocks reused broadly (settings, modals, editor toolbars, logs). No business logic, no context access, no network — pure presentation.

## Design Patterns

- **`class-variance-authority` (CVA) for the variant surface.** `button.tsx` is the canonical example: `buttonVariants = cva(base, { variants: { variant, size }, defaultVariants })` and `ButtonProps extends React.ButtonHTMLAttributes & VariantProps<typeof buttonVariants>`. Only `button.tsx` uses CVA; the other four hand-assemble class strings.
- **`cn` (clsx + tailwind-merge) for class composition.** Every file imports `cn` from `@/lib/utils` (path alias `@/*` → `frontend/*`) to merge consumer `className` overrides with internal classes deterministically, letting callers win on conflicts.
- **`React.forwardRef` for all form controls.** `Button`, `Select`, `Textarea`, `Progress` forward their underlying element ref (`HTMLButtonElement`, `HTMLSelectElement`, `HTMLTextAreaElement`, `HTMLDivElement`) and set `displayName` for devtools. `Tooltip` is a plain function component (no ref needed — it wraps children and portals).
- **Semantic Tailwind tokens.** Classes reference design tokens (`bg-primary`, `text-primary-foreground`, `border-border`, `bg-secondary`, `text-muted-foreground`, `ring-ring`) backed by CSS variables, so theming swaps via variables rather than class edits. A few components also use literal zinc palette values for chrome that should not re-theme.
- **Prop passthrough.** Each primitive spreads `...props` onto the native element after destructuring its own fields, so standard HTML attributes (`disabled`, `value`, `onChange`, `onClick`, `aria-*`, `id`) work transparently.

## Data & Control Flow

- **`button.tsx`** — `<Button variant size className ...props>` renders `<button className={cn(buttonVariants({variant, size, className}))}>`. Variants: `default | destructive | outline | secondary | ghost | link`; sizes: `default (h-9) | sm (h-8) | lg (h-10) | icon (h-9 w-9)`. Defaults: `variant='default'`, `size='default'`. Exports `Button` and `buttonVariants` (the latter is reused to style anchor/Link look-alikes, e.g. by `LtxUpgradePrompt`).
- **`select.tsx`** — `<Select label badge ...props>` wraps a native `<select>` with an absolutely-positioned `ChevronDown` icon (`pointer-events-none`, `appearance-none` on the select). Optional `label` renders above as uppercase zinc-500 text; optional `badge` renders a small chip next to the label (used by `SettingsPanel` for the "PREVIEW" audio badge). Option styling is forced via `[&>option]:bg-zinc-800`.
- **`textarea.tsx`** — `<Textarea label helperText charCount maxChars ...props>` renders a `min-h-[120px]` resizable textarea with a label row and a footer split between `helperText` (left) and a `{charCount}/{maxChars}` counter (right, only when `maxChars !== undefined`).
- **`progress.tsx`** — `<Progress value max showLabel className ...props>` clamps `value/max` to `[0,100]%`, animates the fill width with `transition-all duration-300`, and optionally prints the rounded percentage beneath.
- **`tooltip.tsx`** — `<Tooltip content side='top'|'bottom'|'left'|'right' className>{children}</Tooltip>`. On `mouseenter` it schedules a `DELAY_MS=500ms` timer that computes positioning from the wrapper's `getBoundingClientRect()` (`GAP_PX=6`) and sets `visible`; `mouseleave` clears the timer and hides immediately. The bubble is rendered through `ReactDOM.createPortal(... , document.body)` with `position: fixed; z-[99999]` and `pointer-events-none`, so it is never clipped by `overflow-hidden` ancestors. `useEffect` cleanup clears any pending timer on unmount.

## Integration Points

- **`frontend/lib/utils`** — `cn` (clsx + tailwind-merge), imported via the `@/lib/utils` alias by all five files.
- **`class-variance-authority`** — `cva` and `VariantProps`, imported only by `button.tsx`.
- **`lucide-react`** — `ChevronDown`, imported only by `select.tsx`.
- **`react-dom`** — `ReactDOM.createPortal`, imported only by `tooltip.tsx`.
- **Consumers** — used pervasively across `frontend/components` (e.g. `Button` in `Home`, `LogViewer`, `ModelProfileWizard`, `LtxUpgradePrompt`, `ModelComponentPicker`; `Select`/`Textarea` in `SettingsPanel`; `Tooltip` in `VideoEditor` chrome, `ToolsPanel`, `TimelineToolbar`, `VideoEditorLayoutMenu`) and the editor panels under `frontend/views/editor`.
