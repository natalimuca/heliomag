'use client'

import { useRef } from 'react'
import { Sun } from './sun'
import { useScrollProgress, useViewportSize, lerp, remap } from './use-scroll-progress'

const STATS = [
  { label: 'Embeddings', value: '1,941' },
  { label: 'Train rows', value: '1,785' },
  { label: 'Lead times', value: '3–14d' },
  { label: 'Coverage', value: '98%' },
]

export function ReadoutSection() {
  const ref = useRef<HTMLElement>(null)
  const p = useScrollProgress(ref, 'enter')
  const { w: vw } = useViewportSize()

  const badge = 360
  // Uncapped this grows to 2.2x (792px) anchored top-right with no clipping
  // container — fine on a wide screen, but on a narrow one that pushes the
  // disc far enough past the left edge to risk unwanted horizontal scroll.
  const maxScale = Math.min(2.2, Math.max(1, (vw * 0.85) / badge))
  const scale = lerp(1, maxScale, remap(p, 0.05, 0.55))
  const opacity = remap(p, 0, 0.12) * (1 - remap(p, 0.72, 1))

  return (
    <section
      id="readout"
      ref={ref}
      style={{
        position: 'relative',
        // Pull up 1px so this section overlaps the hero's bottom edge. At
        // fractional device-pixel ratios the two boundaries can round to
        // different device pixels and leave a hairline gap between them;
        // overlapping guarantees there is no gap to show through.
        marginTop: -1,
        minHeight: '100vh',
        /* No overflow clipping here: the growing sun's radial glow must be
           free to taper to full transparency on its own instead of being
           sliced by the section's bottom edge (which read as a hard seam). */
        padding: 'clamp(28px, 5vw, 88px)',
        display: 'flex',
        alignItems: 'center',
      }}
      aria-label="The actual test"
    >
      {/* growing sun badge anchored snug to the top-right corner */}
      <div
        aria-hidden="true"
        style={{
          position: 'absolute',
          top: 16,
          right: 16,
          width: badge,
          height: badge,
          opacity,
          transform: `scale(${scale})`,
          transformOrigin: 'top right',
          pointerEvents: 'none',
          willChange: 'transform, opacity',
        }}
      >
        <Sun size={badge} style={{ position: 'relative' }} />
      </div>

      <div style={{ position: 'relative', maxWidth: 1180, width: '100%', margin: '0 auto' }}>
        <p
          className="font-mono"
          style={{
            color: 'var(--accent-text)',
            fontSize: 12,
            letterSpacing: '0.14em',
            textTransform: 'uppercase',
            marginBottom: 22,
          }}
        >
          The actual test
        </p>

        <h2
          className="display-condensed"
          style={{
            fontSize: 'clamp(30px, 5vw, 74px)',
            lineHeight: 0.98,
            color: 'var(--ink)',
            margin: 0,
            maxWidth: 900,
          }}
        >
          <span style={{ display: 'block' }}>Testing the Sun</span>
          <span style={{ display: 'block' }}>Against 90 years</span>
          <span style={{ display: 'block', color: 'var(--gold-text)' }}>
            Of ground truth
          </span>
        </h2>

        <p
          style={{
            color: 'var(--ink-dim)',
            fontSize: 16,
            lineHeight: 1.6,
            maxWidth: 560,
            marginTop: 26,
          }}
        >
          Surya&apos;s own published evaluation stops at a single 4-day lead time
          for solar wind speed, benchmarked only against other neural nets — it
          never compares against classical indices, never predicts geomagnetic
          activity directly, and never tests multiple lead times. That&apos;s the
          gap this project fills: we score the model&apos;s solar embeddings
          against the Ap and Kp geomagnetic-activity indices — the ground-truth
          planetary record kept continuously since 1932 — to ask a single
          question: does looking at the Sun beat looking at yesterday&apos;s
          number?
        </p>

        <a
          href="#arc"
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 10,
            background: 'var(--accent)',
            color: '#0a0a0a',
            fontWeight: 700,
            fontSize: 13,
            letterSpacing: '0.04em',
            textTransform: 'uppercase',
            padding: '13px 22px',
            borderRadius: 999,
            textDecoration: 'none',
            marginTop: 30,
          }}
        >
          See the probe <span aria-hidden="true">→</span>
        </a>

        {/* stat panel + rotated tag chips */}
        <div
          style={{
            display: 'flex',
            alignItems: 'stretch',
            gap: 14,
            marginTop: 52,
            flexWrap: 'wrap',
          }}
        >
          <div
            style={{
              flex: '1 1 520px',
              display: 'grid',
              gridTemplateColumns: 'repeat(2, 1fr)',
              border: '1px solid var(--line)',
              borderRadius: 14,
              background: 'rgba(13,15,20,0.5)',
              backdropFilter: 'blur(2px)',
              overflow: 'hidden',
            }}
          >
            {STATS.map((s, i) => (
              <div
                key={s.label}
                style={{
                  padding: '26px 26px 22px',
                  borderRight: i % 2 === 0 ? '1px solid var(--line)' : 'none',
                  borderTop: i >= 2 ? '1px solid var(--line)' : 'none',
                }}
              >
                <p
                  className="font-mono"
                  style={{
                    fontSize: 10,
                    letterSpacing: '0.14em',
                    textTransform: 'uppercase',
                    color: 'var(--ink-faint)',
                    marginBottom: 12,
                  }}
                >
                  {s.label}
                </p>
                <p
                  className="font-mono"
                  style={{
                    fontSize: 'clamp(28px, 3.4vw, 44px)',
                    fontWeight: 600,
                    color: 'var(--ink)',
                    margin: 0,
                    lineHeight: 1,
                  }}
                >
                  {s.value}
                </p>
              </div>
            ))}
          </div>

          {/* vertical rotated tag chips */}
          <div style={{ display: 'flex', gap: 10 }}>
            {['AP', 'KP'].map((t) => (
              <div
                key={t}
                className="font-mono"
                style={{
                  border: '1px solid var(--line)',
                  borderRadius: 10,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  padding: '0 6px',
                  minWidth: 40,
                }}
              >
                <span
                  style={{
                    writingMode: 'vertical-rl',
                    transform: 'rotate(180deg)',
                    fontSize: 11,
                    letterSpacing: '0.3em',
                    color: 'var(--gold-text)',
                    padding: '14px 0',
                  }}
                >
                  {t}
                </span>
              </div>
            ))}
          </div>
        </div>

        <p
          className="font-mono"
          style={{
            color: 'var(--ink-faint)',
            fontSize: 11.5,
            letterSpacing: '0.02em',
            lineHeight: 1.5,
            maxWidth: 620,
            marginTop: 16,
          }}
        >
          ~2% of weeks are missing — confirmed genuine SDO source-data gaps, not a
          pipeline issue. 2025 is blocked upstream: NASA&apos;s Surya benchmark
          bucket has no 2025 data yet.
        </p>
      </div>
    </section>
  )
}
