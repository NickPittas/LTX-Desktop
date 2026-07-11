import * as React from 'react'
import { cn } from '@/lib/utils'

const BUTTON_BASE_CLASSES = 'inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50'

const BUTTON_VARIANT_CLASSES = {
  default: 'bg-primary text-primary-foreground shadow hover:bg-primary/90',
  destructive: 'bg-red-500 text-white shadow-sm hover:bg-red-600',
  outline: 'border border-border bg-transparent shadow-sm hover:bg-secondary',
  secondary: 'bg-secondary text-secondary-foreground shadow-sm hover:bg-secondary/80',
  ghost: 'hover:bg-secondary hover:text-secondary-foreground',
  link: 'text-primary underline-offset-4 hover:underline',
} as const

const BUTTON_SIZE_CLASSES = {
  default: 'h-9 px-4 py-2',
  sm: 'h-8 rounded-md px-3 text-xs',
  lg: 'h-10 rounded-md px-8',
  icon: 'h-9 w-9',
} as const

type ButtonVariant = keyof typeof BUTTON_VARIANT_CLASSES
type ButtonSize = keyof typeof BUTTON_SIZE_CLASSES

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    Partial<{ variant: ButtonVariant | null; size: ButtonSize | null }> {}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'default', size = 'default', ...props }, ref) => {
    return (
      <button
        className={cn(
          BUTTON_BASE_CLASSES,
          variant !== null && BUTTON_VARIANT_CLASSES[variant],
          size !== null && BUTTON_SIZE_CLASSES[size],
          className
        )}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = 'Button'

export { Button }
