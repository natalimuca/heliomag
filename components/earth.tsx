import type { CSSProperties } from 'react'

/**
 * Earth rendered entirely with CSS radial-gradients, the same way the Sun is.
 *
 * The element is an enormous circle whose top arc is the only part on screen,
 * so the limb is a true circle rather than a photographed horizon. Because
 * every band is a gradient stop expressed as a fraction of the radius, the
 * whole thing is resolution-independent: no source image to upscale, sharp at
 * any viewport or device pixel ratio, and the curvature is set by one number.
 *
 * The visible strip is only the outermost few percent of the radius, which is
 * why the interesting stops all sit between ~88% and 100%.
 */
export function Earth({
  diameterVw,
  style,
}: {
  diameterVw: number
  style?: CSSProperties
}) {
  return (
    <div
      aria-hidden="true"
      style={{
        position: 'absolute',
        left: '50%',
        top: 0,
        width: `${diameterVw}vw`,
        height: `${diameterVw}vw`,
        transform: 'translateX(-50%)',
        borderRadius: '50%',
        background: `
          radial-gradient(ellipse 13% 0.12% at 33% 0.42%, rgba(200,228,255,0.09), transparent 72%),
          radial-gradient(ellipse 9% 0.09% at 66% 0.3% , rgba(200,228,255,0.07), transparent 74%),
          radial-gradient(circle closest-side at 50% 50%,
            #010206 0%,
            #01030a 97%,
            #020713 98.4%,
            #051029 99.1%,
            #0a2149 99.48%,
            #1a4b86 99.72%,
            #5aa3d8 99.87%,
            #cbe8ff 99.95%,
            rgba(190,226,255,0.45) 99.99%,
            rgba(150,200,255,0) 100%)
        `,
        // Atmosphere kept INSIDE the sphere. An outward glow reached ~36px
        // above the limb and washed a grey haze across the lede and citation
        // sitting just above it — the planet cleared the text, its shadow did
        // not. An inset glow can never paint outside the circle.
        boxShadow: 'inset 0 5px 22px rgba(130,190,255,0.3)',
        ...style,
      }}
    />
  )
}
