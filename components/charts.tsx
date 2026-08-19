'use client'

import { useEffect, useState } from 'react'
import { useInView, reduceMotion } from './use-in-view'

/** Counts a number up once it scrolls into view. */
export function CountUp({
  to,
  decimals = 0,
  suffix = '',
  prefix = '',
  duration = 900,
}: {
  to: number
  decimals?: number
  suffix?: string
  prefix?: string
  duration?: number
}) {
  const { ref, inView } = useInView<HTMLSpanElement>(0.4)
  const [v, setV] = useState(0)

  useEffect(() => {
    if (!inView) return
    if (reduceMotion()) {
      setV(to)
      return
    }
    let raf = 0
    const start = performance.now()
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration)
      const eased = 1 - Math.pow(1 - t, 3)
      setV(to * eased)
      if (t < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [inView, to, duration])

  return (
    <span ref={ref}>
      {prefix}
      {v.toLocaleString(undefined, {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
      })}
      {suffix}
    </span>
  )
}

/** Wraps children in a fade/rise that plays when scrolled into view. */
export function Reveal({
  children,
  delay = 0,
  style,
}: {
  children: React.ReactNode
  delay?: number
  style?: React.CSSProperties
}) {
  const { ref, inView } = useInView<HTMLDivElement>(0.15)
  return (
    <div
      ref={ref}
      style={{
        opacity: inView ? 1 : 0,
        transform: inView ? 'translateY(0)' : 'translateY(18px)',
        transition: `opacity 620ms ease ${delay}ms, transform 620ms cubic-bezier(.2,.7,.3,1) ${delay}ms`,
        ...style,
      }}
    >
      {children}
    </div>
  )
}

type Series = { label: string; color: string; values: number[] }

/**
 * Grouped bar chart. Bars grow from zero on first view and report their exact
 * value on hover, so the numbers stay readable without cluttering every bar.
 */
export function GroupedBars({
  categories,
  series,
  unit,
  lowerIsBetter = true,
  height = 300,
}: {
  categories: string[]
  series: Series[]
  unit: string
  lowerIsBetter?: boolean
  height?: number
}) {
  const { ref, inView } = useInView<HTMLDivElement>(0.2)
  const [hover, setHover] = useState<{ c: number; s: number } | null>(null)
  const max = Math.max(...series.flatMap((s) => s.values)) * 1.14

  return (
    <div ref={ref}>
      <div
        style={{
          display: 'flex',
          alignItems: 'flex-end',
          gap: 'clamp(10px, 3vw, 44px)',
          height,
          borderBottom: '1px solid var(--line)',
          position: 'relative',
        }}
      >
        {categories.map((cat, ci) => {
          const best = lowerIsBetter
            ? Math.min(...series.map((s) => s.values[ci]))
            : Math.max(...series.map((s) => s.values[ci]))
          return (
            <div
              key={cat}
              style={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                height: '100%',
                justifyContent: 'flex-end',
              }}
            >
              <div
                style={{
                  display: 'flex',
                  alignItems: 'flex-end',
                  gap: 6,
                  width: '100%',
                  justifyContent: 'center',
                  height: '100%',
                }}
              >
                {series.map((s, si) => {
                  const isHot = hover && hover.c === ci && hover.s === si
                  const isBest = s.values[ci] === best
                  return (
                    <div
                      key={s.label}
                      onMouseEnter={() => setHover({ c: ci, s: si })}
                      onMouseLeave={() => setHover(null)}
                      style={{
                        position: 'relative',
                        flex: 1,
                        maxWidth: 56,
                        minWidth: 12,
                        height: inView ? `${(s.values[ci] / max) * 100}%` : '0%',
                        background: s.color,
                        borderRadius: '4px 4px 0 0',
                        transition: `height 780ms cubic-bezier(.2,.7,.3,1) ${si * 90 + ci * 60}ms, filter 160ms ease`,
                        filter: isHot ? 'brightness(1.35)' : 'none',
                        cursor: 'default',
                        outline: isBest ? '1px solid rgba(255,255,255,0.28)' : 'none',
                        outlineOffset: -1,
                      }}
                    >
                      {isHot && (
                        <div
                          className="font-mono"
                          style={{
                            position: 'absolute',
                            bottom: 'calc(100% + 8px)',
                            left: '50%',
                            transform: 'translateX(-50%)',
                            whiteSpace: 'nowrap',
                            background: '#0d0f14',
                            border: '1px solid var(--line)',
                            borderRadius: 6,
                            padding: '6px 9px',
                            fontSize: 11,
                            color: 'var(--ink)',
                            zIndex: 5,
                          }}
                        >
                          {s.label} · {s.values[ci].toFixed(2)} {unit}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
              <span
                className="font-mono"
                style={{
                  fontSize: 12,
                  letterSpacing: '0.1em',
                  color: 'var(--ink-dim)',
                  marginTop: 12,
                }}
              >
                {cat}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

/**
 * Line chart that draws itself in. Used for the persistence-decay curve, where
 * the shape (a collapse) is the point rather than any single value.
 */
export function DecayLine({
  points,
  refLevel,
  refLabel,
  width = 560,
  height = 220,
}: {
  points: { x: number; y: number }[]
  refLevel?: number
  refLabel?: string
  width?: number
  height?: number
}) {
  const { ref, inView } = useInView<HTMLDivElement>(0.3)
  const padL = 44
  const padB = 30
  const padT = 14
  const maxX = Math.max(...points.map((p) => p.x))
  const maxY = Math.max(...points.map((p) => p.y))
  const sx = (x: number) => padL + (x / maxX) * (width - padL - 12)
  const sy = (y: number) => padT + (1 - y / maxY) * (height - padT - padB)

  const d = points
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`)
    .join(' ')
  const area =
    `${d} L${sx(maxX).toFixed(1)},${sy(0).toFixed(1)} L${sx(0).toFixed(1)},${sy(0).toFixed(1)} Z`

  return (
    <div ref={ref}>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img">
        <defs>
          <linearGradient id="decayFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.28" />
            <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
          </linearGradient>
        </defs>

        {[0, 0.25, 0.5, 0.75, 1].map((f) => (
          <g key={f}>
            <line
              x1={padL}
              x2={width - 12}
              y1={sy(maxY * f)}
              y2={sy(maxY * f)}
              stroke="var(--line)"
              strokeWidth="1"
            />
            <text
              x={padL - 8}
              y={sy(maxY * f) + 3.5}
              textAnchor="end"
              fill="var(--ink-faint)"
              style={{ fontSize: 9, fontFamily: 'var(--font-mono)' }}
            >
              {(maxY * f).toFixed(2)}
            </text>
          </g>
        ))}

        {refLevel !== undefined && (
          <>
            <line
              x1={padL}
              x2={width - 12}
              y1={sy(refLevel)}
              y2={sy(refLevel)}
              stroke="var(--gold)"
              strokeWidth="1"
              strokeDasharray="4 3"
              opacity={inView ? 0.9 : 0}
              style={{ transition: 'opacity 500ms ease 700ms' }}
            />
            {refLabel && (
              <text
                x={width - 14}
                y={sy(refLevel) - 6}
                textAnchor="end"
                fill="var(--gold-text)"
                opacity={inView ? 1 : 0}
                style={{
                  fontSize: 9,
                  fontFamily: 'var(--font-mono)',
                  letterSpacing: '0.08em',
                  transition: 'opacity 500ms ease 800ms',
                }}
              >
                {refLabel}
              </text>
            )}
          </>
        )}

        <path
          d={area}
          fill="url(#decayFill)"
          opacity={inView ? 1 : 0}
          style={{ transition: 'opacity 700ms ease 300ms' }}
        />
        <path
          d={d}
          fill="none"
          stroke="var(--accent)"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          style={{
            strokeDasharray: 1200,
            strokeDashoffset: inView ? 0 : 1200,
            transition: 'stroke-dashoffset 1100ms cubic-bezier(.2,.7,.3,1)',
          }}
        />

        {points.map((p, i) => (
          <g key={i}>
            <circle
              cx={sx(p.x)}
              cy={sy(p.y)}
              r="3.2"
              fill="var(--accent)"
              opacity={inView ? 1 : 0}
              style={{ transition: `opacity 300ms ease ${400 + i * 110}ms` }}
            />
            <text
              x={sx(p.x)}
              y={height - 10}
              textAnchor="middle"
              fill="var(--ink-dim)"
              style={{ fontSize: 10, fontFamily: 'var(--font-mono)' }}
            >
              {p.x}d
            </text>
          </g>
        ))}
      </svg>
    </div>
  )
}
