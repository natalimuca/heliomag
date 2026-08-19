'use client'

import { useEffect, useRef, useState } from 'react'

/**
 * Tracks whether an element is in view, in BOTH directions — it does not
 * disconnect after the first hit, so animations replay when you scroll back
 * up past a section rather than firing once and staying put for the session.
 */
export function useInView<T extends HTMLElement = HTMLDivElement>(
  threshold = 0.25,
) {
  const ref = useRef<T>(null)
  const [inView, setInView] = useState(false)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    if (typeof IntersectionObserver === 'undefined') {
      setInView(true)
      return
    }
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => setInView(e.isIntersecting))
      },
      // Asymmetric margin: elements arm slightly before they reach the middle
      // of the viewport and only disarm once well clear of it, which stops the
      // state flickering while scrolling across a boundary.
      { threshold, rootMargin: '-5% 0px -10% 0px' },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [threshold])

  return { ref, inView }
}

export const reduceMotion = () =>
  typeof window !== 'undefined' &&
  window.matchMedia &&
  window.matchMedia('(prefers-reduced-motion: reduce)').matches
