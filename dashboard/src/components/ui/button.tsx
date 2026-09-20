import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"
import { Slot } from "radix-ui"

const buttonVariants = cva(
  "inline-flex shrink-0 items-center justify-center rounded-full border border-transparent font-medium whitespace-nowrap transition-all outline-none select-none focus-visible:ring-2 focus-visible:ring-[#533afd]/40 active:scale-[0.985] disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4 cursor-pointer",
  {
    variants: {
      variant: {
        default:
          "bg-[#533afd] text-white hover:bg-[#4434d4] active:bg-[#2e2b8c] shadow-[0_1px_2px_rgba(0,0,0,0.08)]",
        outline:
          "border-[#e3e8ee] bg-white text-[#0d253d] hover:bg-[#f6f9fc] hover:border-[#cbd5e1] shadow-[0_1px_2px_rgba(0,0,0,0.04)]",
        secondary:
          "bg-[#f1f5f9] text-[#0d253d] hover:bg-[#e2e8f0]",
        ghost:
          "hover:bg-[#f1f5f9] text-[#273951] hover:text-[#0d253d]",
        destructive:
          "bg-rose-50 text-rose-700 border border-rose-200 hover:bg-rose-100",
        link:
          "text-[#533afd] underline-offset-4 hover:underline p-0 h-auto",
      },
      size: {
        default: "h-9 gap-1.5 px-4 text-sm tracking-tight",
        xs: "h-7 gap-1 px-2.5 text-xs",
        sm: "h-8 gap-1.5 px-3 text-sm",
        lg: "h-10 gap-2 px-5 text-base",
        icon: "size-9 rounded-full",
        "icon-sm": "size-7.5 rounded-full",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

function Button({
  className,
  variant = "default",
  size = "default",
  asChild = false,
  ...props
}: React.ComponentProps<"button"> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean
  }) {
  const Comp = asChild ? Slot.Root : "button"

  return (
    <Comp
      data-slot="button"
      data-variant={variant}
      data-size={size}
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  )
}

export { Button, buttonVariants }
