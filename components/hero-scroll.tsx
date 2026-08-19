'use client'

import { useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import { Sun } from './sun'
import { Earth } from './earth'
import { ScrambleText } from './scramble-text'
import {
  useScrollProgress,
  useViewportSize,
  easeInOut,
  lerp,
  remap,
} from './use-scroll-progress'

const navLinkStyle: CSSProperties = {
  color: 'inherit',
  textDecoration: 'none',
}

const MENU_ITEMS = [
  { label: 'Index', href: '#top' },
  { label: 'Method', href: '#arc' },
  { label: 'Data', href: '#data' },
  { label: 'Verdict', href: '#verdict' },
  { label: 'Residual correction', href: '#residual' },
  { label: 'Where this leaves it', href: '#where-next' },
]

export function HeroScroll() {
  const wrapperRef = useRef<HTMLElement>(null)
  const progress = useScrollProgress(wrapperRef, 'pin')
  const { w: vw, h: vh } = useViewportSize()
  const [menuOpen, setMenuOpen] = useState(false)

  // Below ~768px the pinned scroll-jack has no room to work with and the
  // desktop-tuned tx/ty math (built around wide viewports) falls apart, so
  // skip the animation entirely and render the fully-docked end state as a
  // normal static layout instead — no pin, no 230vh dead scroll zone.
  const isMobile = vw > 0 && vw < 768
  const effProgress = isMobile ? 1 : progress

  const e = easeInOut(effProgress)

  // Sun: large & centered -> ~260px badge docked top-right corner. The
  // "large & centered" scale is capped to the viewport so the disc never
  // gets clipped top/bottom by the pinned container's overflow:hidden on
  // shorter screens — 3x was tuned for a tall viewport and clipped on
  // anything under ~900px tall.
  const badge = 260
  const margin = 24
  const startScale = Math.max(1.4, Math.min(3, (Math.min(vw, vh) * 0.82) / badge))
  const sunScale = lerp(startScale, 1, e)
  const tx = lerp(0, vw / 2 - badge / 2 - margin, e)
  const ty = lerp(0, badge / 2 + margin - vh / 2, e)

  // Earth's crop has to be computed, not a fixed percentage. The element is
  // full-bleed (124vw) but only 70vh tall, so its aspect ratio swings wildly
  // with viewport width — and `object-fit: cover` scales by the larger axis.
  // At 1920px wide the 1024px source is scaled ~2.3x and only a ~310px slice
  // of the photo is visible (the dark upper limb); at 700px wide it scales
  // ~0.76x and an ~800px slice shows (the bright, detailed surface). A single
  // objectPosition therefore cannot look the same at both. Solving for the
  // position that keeps the limb edge (~y=470 in the source) near the top of
  // whatever slice is visible keeps the bright surface in frame at any width.
  // Strip height as a fraction of the viewport. This also sets where the limb
  // sits (top of the strip = 1 - EARTH_H), so it must stay below the verdict
  // block on the right — at 0.28 the limb cut straight through the quote's
  // last line and swallowed the citation underneath it.
  const EARTH_H = 0.19
  // Sphere diameter in vw — the single curvature knob. Nothing is resized by
  // it (the planet is drawn, not sampled), so sharpness is unaffected either
  // way: higher = flatter horizon, lower = rounder.
  const EARTH_DISC_VW = 680

  // Earth rises + fades in during the middle of the scroll.
  const earthP = isMobile ? 1 : remap(progress, 0.12, 0.82)
  const earthE = easeInOut(earthP)

  // Chrome staged reveal.
  const guideP = isMobile ? 1 : remap(progress, 0.28, 0.5)
  const chromeP = isMobile ? 1 : remap(progress, 0.5, 0.82)
  const scrambleActive = isMobile ? true : progress > 0.52

  return (
    <section
      id="top"
      ref={wrapperRef}
      style={{ height: isMobile ? 'auto' : '230vh', position: 'relative' }}
      aria-label="heliomag hero"
    >
      <div
        style={{
          position: isMobile ? 'relative' : 'sticky',
          top: 0,
          height: isMobile ? 'auto' : '100vh',
          minHeight: '100vh',
          overflow: 'hidden',
        }}
      >
        {/* SCRIM — sits UNDER Earth, and covers the full height uniformly.
            Order matters and was the root of a long-running seam: while the
            scrim painted over Earth it had to fade out just above the limb so
            it wouldn't dim the planet, and that fade made the starfield jump
            from hidden to fully visible across a ~30px band — a bright strip
            between the dark sky and the planet. With Earth drawn on top, the
            scrim can dim the stars evenly everywhere and never touch the
            planet, so there is no fade and therefore no band. */}
        <div
          aria-hidden="true"
          style={{
            position: 'absolute',
            inset: 0,
            opacity: Math.max(earthP * 0.9, chromeP),
            background:
              'linear-gradient(105deg, rgba(0,0,0,0.92) 0%, rgba(0,0,0,0.86) 30%, rgba(0,0,0,0.8) 60%, rgba(0,0,0,0.76) 100%)',
            // The scrim MUST reach alpha 0 by the container's bottom edge.
            // It dims the page-wide starfield, so if it stopped at full
            // strength there, stars would be suppressed above the boundary and
            // full-brightness below it — that brightness step is the seam.
            // Because Earth now draws on top, this fade can span the strip
            // (81%→100%) instead of having to finish above the limb: it is
            // hidden behind the planet for most of its length, and over the
            // last stretch both fade out together, so the transition ramps to
            // zero instead of stepping.
            // Also lands on 0 before the edge, for the same rounding reason.
            maskImage: `linear-gradient(to bottom, #000 0%, #000 ${((1 - EARTH_H) * 100).toFixed(0)}%, transparent 95%, transparent 100%)`,
            WebkitMaskImage: `linear-gradient(to bottom, #000 0%, #000 ${((1 - EARTH_H) * 100).toFixed(0)}%, transparent 95%, transparent 100%)`,
            pointerEvents: 'none',
          }}
        />

        {/* EARTH — a viewport-clipping window holding one enormous circle.
            Curvature and sharpness are decoupled: the circle's radius sets how
            gently the limb arcs, and being drawn rather than photographed
            there are no pixels to stretch. Rendered after the scrim so it
            keeps its full brightness. */}
        <div
          aria-hidden={earthP < 0.05}
          style={{
            position: 'absolute',
            left: 0,
            right: 0,
            bottom: 0,
            height: `${EARTH_H * 100}vh`,
            opacity: earthP,
            // NO transform here. Translating the masked strip pushed its
            // fade-out region below the container's clip edge, so during the
            // reveal the boundary met the strip's still-opaque middle — a hard
            // cut that only disappeared once the animation finished. The
            // rise now happens on the inner wrapper, leaving the mask pinned
            // to the edge at every scroll position.
            willChange: 'opacity',
            overflow: 'hidden',
            // Only the bottom needs to reach alpha 0 by the container edge so
            // bright pixels never butt into the next section. No top fade —
            // the arc is the top edge and fading it would erase the curve.
            maskImage:
              // Reaches alpha 0 at 90%, not 100%. The strip's height is a vh
              // value, so on a fractional device-pixel ratio (a 125%-scaled
              // display at 100% browser zoom) the clip edge can round onto a
              // row that is still faintly opaque — a 1px cut that disappears
              // at other zoom levels. Finishing the fade early leaves the last
              // ~10% as dead transparent space for rounding to land in.
              'linear-gradient(to bottom, rgba(0,0,0,1) 0%, rgba(0,0,0,1) 22%, rgba(0,0,0,0.88) 36%, rgba(0,0,0,0.66) 50%, rgba(0,0,0,0.42) 63%, rgba(0,0,0,0.2) 76%, rgba(0,0,0,0.06) 85%, rgba(0,0,0,0) 90%, rgba(0,0,0,0) 100%)',
            WebkitMaskImage:
              // Reaches alpha 0 at 90%, not 100%. The strip's height is a vh
              // value, so on a fractional device-pixel ratio (a 125%-scaled
              // display at 100% browser zoom) the clip edge can round onto a
              // row that is still faintly opaque — a 1px cut that disappears
              // at other zoom levels. Finishing the fade early leaves the last
              // ~10% as dead transparent space for rounding to land in.
              'linear-gradient(to bottom, rgba(0,0,0,1) 0%, rgba(0,0,0,1) 22%, rgba(0,0,0,0.88) 36%, rgba(0,0,0,0.66) 50%, rgba(0,0,0,0.42) 63%, rgba(0,0,0,0.2) 76%, rgba(0,0,0,0.06) 85%, rgba(0,0,0,0) 90%, rgba(0,0,0,0) 100%)',
          }}
        >
          <div
            style={{
              position: 'absolute',
              inset: 0,
              transform: `translateY(${lerp(18, 0, earthE)}%)`,
              willChange: 'transform',
            }}
          >
            <Earth diameterVw={EARTH_DISC_VW} />
          </div>
        </div>

        {/* GUIDE LINES — crosshair + vertical divider */}
        <div
          aria-hidden="true"
          style={{ opacity: guideP, pointerEvents: 'none' }}
        >
          <div
            style={{
              position: 'absolute',
              top: 0,
              bottom: 0,
              left: '62%',
              width: 1,
              background: 'var(--line)',
            }}
          />
          <div
            style={{
              position: 'absolute',
              left: 0,
              right: 0,
              top: '46%',
              height: 1,
              background:
                'linear-gradient(to right, transparent, var(--line) 12%, var(--line) 88%, transparent)',
            }}
          />
          <CrosshairMark left="62%" top="46%" />
          <CrosshairMark left="12%" top="46%" />
        </div>

        {/* SUN */}
        <div
          style={{
            position: 'absolute',
            left: '50%',
            top: '50%',
            width: badge,
            height: badge,
            transform: `translate(-50%, -50%) translate(${tx}px, ${ty}px) scale(${sunScale})`,
            transformOrigin: 'center center',
            willChange: 'transform',
          }}
        >
          <Sun size={badge} style={{ position: 'relative' }} />
        </div>

        {/* CONTENT LAYER */}
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
            padding: 'clamp(20px, 3vw, 40px)',
          }}
        >
          {/* NAV */}
          <nav
            style={{
              opacity: chromeP,
              transform: `translateY(${lerp(-12, 0, chromeP)}px)`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 16,
            }}
          >
            <span
              className="font-mono"
              style={{
                fontSize: 13,
                letterSpacing: '0.14em',
                textTransform: 'uppercase',
                color: 'var(--ink)',
              }}
            >
              heliomag
            </span>
            <div
              className="font-mono"
              style={{
                display: 'flex',
                gap: 'clamp(14px, 2vw, 30px)',
                fontSize: 11,
                letterSpacing: '0.14em',
                textTransform: 'uppercase',
                color: 'var(--ink-dim)',
              }}
            >
              <a href="#top" style={navLinkStyle}>Index</a>
              <a href="#arc" style={navLinkStyle}>Method</a>
              <a href="#data" style={navLinkStyle}>Data</a>
              <a href="#verdict" style={{ ...navLinkStyle, color: 'var(--accent-text)' }}>Verdict</a>
            </div>
          </nav>

          {/* MIDDLE */}
          <div
            style={{
              display: 'flex',
              alignItems: 'flex-end',
              justifyContent: 'space-between',
              gap: 32,
              opacity: chromeP,
              transform: `translateY(${lerp(24, 0, chromeP)}px)`,
              flexWrap: 'wrap',
            }}
          >
            {/* headline column */}
            <div style={{ maxWidth: 720, flex: '1 1 460px' }}>
              <p
                className="font-mono"
                style={{
                  color: 'var(--accent-text)',
                  fontSize: 12,
                  letterSpacing: '0.12em',
                  textTransform: 'uppercase',
                  marginBottom: 18,
                }}
              >
                Does the Sun tell us more than a number?
              </p>

              <h1
                className="display-condensed"
                style={{
                  fontSize: 'clamp(34px, 5.4vw, 82px)',
                  lineHeight: 0.96,
                  color: 'var(--ink)',
                  margin: 0,
                }}
              >
                <span style={{ display: 'block' }}>
                  <ScrambleText text="READING THE SUN" active={scrambleActive} />
                </span>
                <span style={{ display: 'block' }}>
                  <ScrambleText
                    text="FOR WARNINGS EARTH'S"
                    active={scrambleActive}
                    delay={120}
                  />
                </span>
                <span style={{ display: 'block', color: 'var(--gold-text)' }}>
                  <ScrambleText
                    text="OWN NUMBERS MISS"
                    active={scrambleActive}
                    delay={240}
                  />
                </span>
              </h1>

              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 22,
                  marginTop: 30,
                  flexWrap: 'wrap',
                }}
              >
                <a
                  href="#readout"
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
                  }}
                >
                  See the findings <span aria-hidden="true">→</span>
                </a>
                <p
                  style={{
                    color: 'var(--ink-dim)',
                    fontSize: 14,
                    lineHeight: 1.5,
                    maxWidth: 320,
                    margin: 0,
                  }}
                >
                  A solar foundation model reads active-region imagery directly —
                  where legacy indices flatten the Sun into a single daily
                  scalar.
                </p>
              </div>
            </div>

            {/* verdict callout */}
            <div
              style={{
                flex: '0 1 300px',
                textAlign: 'right',
                paddingBottom: 4,
              }}
            >
              <p
                className="font-mono"
                style={{
                  fontSize: 11,
                  letterSpacing: '0.14em',
                  textTransform: 'uppercase',
                  color: 'var(--ink-faint)',
                  marginBottom: 12,
                }}
              >
                Verdict
              </p>
              <p
                style={{
                  fontSize: 'clamp(18px, 1.7vw, 25px)',
                  lineHeight: 1.24,
                  color: 'var(--ink)',
                  margin: 0,
                }}
              >
                <span style={{ fontWeight: 300, fontStyle: 'italic' }}>“</span>
                <span style={{ fontWeight: 700 }}>Classical indices beat</span>
                <span style={{ fontWeight: 300 }}>
                  {' '}
                  the embedding probe at every lead time tested — 3 to 14 days,
                  for both Ap and Kp.
                </span>
                <span style={{ fontWeight: 300, fontStyle: 'italic' }}>”</span>
              </p>
              <p
                className="font-mono"
                style={{
                  fontSize: 10,
                  letterSpacing: '0.1em',
                  textTransform: 'uppercase',
                  color: 'var(--ink-faint)',
                  marginTop: 14,
                }}
              >
                20 lead × target comparisons · 1,785 training rows
              </p>
            </div>
          </div>

          {/* FOOTER ROW */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 16,
              paddingTop: 18,
              // No border-top. A 1px full-width rule here read as a hard page
              // seam across the hero — it sat just above this row, spanning
              // the whole viewport, and was the horizontal line that survived
              // every Earth, scrim and mask change, because it was never part
              // of them.
              opacity: guideP,
            }}
          >
            <span
              className="font-mono"
              style={{
                fontSize: 11,
                letterSpacing: '0.1em',
                textTransform: 'uppercase',
                color: 'var(--ink-dim)',
              }}
            >
              heliomag · research log
            </span>
            <span
              className="font-mono"
              style={{
                fontSize: 11,
                letterSpacing: '0.16em',
                color: 'var(--ink-faint)',
              }}
            >
              01 / 08
            </span>
            <div style={{ position: 'relative' }}>
              <button
                className="font-mono"
                onClick={() => setMenuOpen((v) => !v)}
                aria-expanded={menuOpen}
                aria-haspopup="true"
                style={{
                  fontSize: 11,
                  letterSpacing: '0.14em',
                  textTransform: 'uppercase',
                  color: 'var(--ink)',
                  background: 'transparent',
                  border: '1px solid var(--line)',
                  borderRadius: 999,
                  padding: '7px 16px',
                  cursor: 'pointer',
                }}
              >
                {menuOpen ? 'Close' : 'Menu'}
              </button>
              {menuOpen && (
                <div
                  role="menu"
                  style={{
                    position: 'absolute',
                    bottom: 'calc(100% + 8px)',
                    right: 0,
                    minWidth: 200,
                    background: 'var(--surface-2, #0d0f14)',
                    border: '1px solid var(--line)',
                    borderRadius: 12,
                    padding: 6,
                    display: 'flex',
                    flexDirection: 'column',
                    boxShadow: '0 12px 32px rgba(0,0,0,0.5)',
                    zIndex: 20,
                  }}
                >
                  {MENU_ITEMS.map((item) => (
                    <a
                      key={item.href}
                      href={item.href}
                      onClick={() => setMenuOpen(false)}
                      className="font-mono"
                      style={{
                        fontSize: 12.5,
                        letterSpacing: '0.04em',
                        color: 'var(--ink-dim)',
                        textDecoration: 'none',
                        padding: '9px 10px',
                        borderRadius: 6,
                      }}
                    >
                      {item.label}
                    </a>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

function CrosshairMark({ left, top }: { left: string; top: string }) {
  return (
    <div
      style={{
        position: 'absolute',
        left,
        top,
        width: 14,
        height: 14,
        transform: 'translate(-50%, -50%)',
      }}
    >
      <div
        style={{
          position: 'absolute',
          left: '50%',
          top: 0,
          bottom: 0,
          width: 1,
          background: 'var(--accent-text)',
          transform: 'translateX(-50%)',
        }}
      />
      <div
        style={{
          position: 'absolute',
          top: '50%',
          left: 0,
          right: 0,
          height: 1,
          background: 'var(--accent-text)',
          transform: 'translateY(-50%)',
        }}
      />
    </div>
  )
}
