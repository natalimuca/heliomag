import type { CSSProperties } from 'react'

/**
 * A self-contained sun rendered entirely with CSS radial-gradients.
 * The transparent falloff at the edge avoids the rectangular/box artifacts
 * that blend-mode or non-alpha video approaches leak around the disc.
 */
export function Sun({
  size,
  className,
  style,
}: {
  size: number
  className?: string
  style?: CSSProperties
}) {
  return (
    <div
      aria-hidden="true"
      className={className}
      style={{
        width: size,
        height: size,
        borderRadius: '50%',
        // outer corona glow (soft transparent falloff) + hot core
        background: `
          radial-gradient(circle at 50% 50%,
            #fff6e6 0%,
            #ffd27a 14%,
            #ffab3d 30%,
            #ff7a1e 44%,
            rgba(255, 106, 43, 0.55) 55%,
            rgba(255, 106, 43, 0.18) 68%,
            rgba(255, 106, 43, 0.06) 80%,
            rgba(255, 106, 43, 0) 92%)
        `,
        filter: 'saturate(1.05)',
        ...style,
      }}
    >
      {/* faint surface texture / mottling — kept well inside the disc and low
          opacity so it never reaches the edge (that was producing a jagged,
          rippled boundary on the lower-right curve). */}
      <div
        style={{
          position: 'absolute',
          inset: '30%',
          borderRadius: '50%',
          mixBlendMode: 'overlay',
          opacity: 0.28,
          background: `
            radial-gradient(circle at 40% 36%, rgba(255,255,255,0.45), transparent 40%),
            radial-gradient(circle at 60% 58%, rgba(160,70,15,0.35), transparent 42%),
            radial-gradient(circle at 34% 66%, rgba(255,220,150,0.35), transparent 38%)
          `,
        }}
      />
    </div>
  )
}
