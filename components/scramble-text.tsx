'use client'

import { useEffect, useRef, useState } from 'react'

const GLYPHS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/#*<>%'

/**
 * Reveals `text` with a per-character "scramble" effect: each character
 * cycles through random glyphs before settling on its final value.
 * Runs once when `active` first becomes true.
 */
export function ScrambleText({
  text,
  active,
  className,
  delay = 0,
}: {
  text: string
  active: boolean
  className?: string
  delay?: number
}) {
  const [display, setDisplay] = useState(text)
  const started = useRef(false)

  useEffect(() => {
    if (!active || started.current) return
    started.current = true

    const chars = text.split('')
    let raf: number
    let start: number | null = null
    const duration = 720
    const perChar = 34 // ms of lock-in stagger per character

    const tick = (ts: number) => {
      if (start === null) start = ts + delay
      const elapsed = ts - start
      if (elapsed < 0) {
        raf = requestAnimationFrame(tick)
        return
      }
      let done = true
      const out = chars.map((c, i) => {
        if (c === ' ') return ' '
        const settleAt = i * perChar + duration * 0.35
        if (elapsed >= settleAt) return c
        done = false
        return GLYPHS[Math.floor(Math.random() * GLYPHS.length)]
      })
      setDisplay(out.join(''))
      if (!done) raf = requestAnimationFrame(tick)
      else setDisplay(text)
    }

    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [active, text, delay])

  return <span className={className}>{display}</span>
}
