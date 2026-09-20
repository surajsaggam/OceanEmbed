import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex items-center justify-center rounded-full border px-2.5 py-0.5 text-[11px] font-medium w-fit whitespace-nowrap shrink-0 gap-1 transition-colors",
  {
    variants: {
      variant: {
        default:
          "border-transparent bg-[#eef0fd] text-[#4434d4]",
        secondary:
          "border-[#e2e8f0] bg-[#f8fafc] text-[#475569]",
        outline:
          "border-[#e2e8f0] bg-white text-[#273951]",
        amber:
          "border-amber-200 bg-amber-50 text-amber-800",
        emerald:
          "border-emerald-200 bg-emerald-50 text-emerald-800",
        rose:
          "border-rose-200 bg-rose-50 text-rose-800",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

function Badge({
  className,
  variant,
  ...props
}: React.ComponentProps<"span"> & VariantProps<typeof badgeVariants>) {
  return (
    <span
      data-slot="badge"
      className={cn(badgeVariants({ variant }), className)}
      {...props}
    />
  )
}

export { Badge, badgeVariants }
