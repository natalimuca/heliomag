'use client'

import { useEffect, useRef, useState } from 'react'
import type { RefObject } from 'react'

/**
 * Returns scroll progress (0 -> 1) of an element through the viewport.
 *
 * mode 'pin':  progress across a tall wrapper that contains a sticky child.
 *              0 when the wrapper top hits the viewport top, 1 when its
 *              bottom reaches the viewport bottom (i.e. the pin is released).
 * mode 'enter': 0 when the element's top enters the bottom of the viewport,
 *               1 when the element's bottom leaves the top of the viewport.
 */
export function useScrollProgress(
  ref: RefObject<HTMLElement | null>,
  mode: 'pin' | 'enter' = 'pin',
) {
  const [progress, setProgress] = useState(0)
  const frame = useRef<number | null>(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return

    const compute = () => {
      frame.current = null
      const rect = el.getBoundingClientRect()
      const vh = window.innerHeight
      let p = 0
      if (mode === 'pin') {
        const distance = rect.height - vh
        p = distance > 0 ? -rect.top / distance : 0
      } else {
        const total = rect.height + vh
        p = (vh - rect.top) / total
      }
      setProgress(Math.min(1, Math.max(0, p)))
    }

    const onScroll = () => {
      if (frame.current == null) {
        frame.current = requestAnimationFrame(compute)
      }
    }

    compute()
    // Layout isn't always settled yet on the very first paint (webfont
    // swap, scrollbar-width changes, etc.), and nothing else recomputes
    // until the user actually scrolls or resizes — so the hero could get
    // stuck rendering a stale progress value on load. Recompute a couple
    // more times right after mount to catch any late layout shift.
    const raf = requestAnimationFrame(compute)
    const settleTimer = window.setTimeout(compute, 300)
    window.addEventListener('scroll', onScroll, { passive: true })
    window.addEventListener('resize', onScroll)
    window.addEventListener('load', compute)
    return () => {
      window.removeEventListener('scroll', onScroll)
      window.removeEventListener('resize', onScroll)
      window.removeEventListener('load', compute)
      cancelAnimationFrame(raf)
      window.clearTimeout(settleTimer)
      if (frame.current != null) cancelAnimationFrame(frame.current)
    }
  }, [ref, mode])

  return progress
}

export function useViewportSize() {
  const [size, setSize] = useState({ w: 1280, h: 800 })
  useEffect(() => {
    const update = () => setSize({ w: window.innerWidth, h: window.innerHeight })
    update()
    window.addEventListener('resize', update)
    return () => window.removeEventListener('resize', update)
  }, [])
  return size
}

/** cubic ease-in-out */
export function easeInOut(t: number) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2
}

export function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t
}

/** remap x from [inMin,inMax] to [0,1], clamped */
export function remap(x: number, inMin: number, inMax: number) {
  return Math.min(1, Math.max(0, (x - inMin) / (inMax - inMin)))
}
