import * as React from "react"
import { cn } from "@/lib/utils"

function Input({ className, type, ...props }: React.ComponentProps<"input">) {
  return (
    <input
      type={type}
      data-slot="input"
      className={cn(
        "flex h-9 w-full rounded-md border border-[#cbd5e1] bg-white px-3 py-1.5 text-sm text-[#0d253d] placeholder:text-[#94a3b8] shadow-xs transition-colors focus-visible:outline-none focus-visible:border-[#533afd] focus-visible:ring-2 focus-visible:ring-[#533afd]/20 disabled:cursor-not-allowed disabled:opacity-50",
        className
      )}
      {...props}
    />
  )
}

export { Input }
