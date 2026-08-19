'use client'

import { useEffect, useState } from 'react'

type Star = { x: number; y: number; r: number; accent?: boolean }
type Figure = {
  name: string
  w: number
  h: number
  stars: Star[]
  lines: [number, number][]
}

// Approximate real sky geometry, not decorative scatter — the Dipper's bowl and
// handle, Scorpius' claws/Antares/curled tail, and Cancer's faint inverted Y.
const URSA_MAJOR: Figure = {
  name: 'Ursa Major',
  w: 152,
  h: 64,
  stars: [
    { x: 4, y: 28, r: 2.1, accent: true }, // Dubhe
    { x: 7, y: 52, r: 1.9 }, // Merak
    { x: 41, y: 56, r: 1.6 }, // Phecda
    { x: 45, y: 33, r: 1.2 }, // Megrez
    { x: 75, y: 27, r: 1.9 }, // Alioth
    { x: 105, y: 21, r: 1.7 }, // Mizar
    { x: 138, y: 31, r: 2.0, accent: true }, // Alkaid
  ],
  lines: [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [3, 4],
    [4, 5],
    [5, 6],
  ],
}

const SCORPIUS: Figure = {
  name: 'Scorpius',
  w: 118,
  h: 146,
  stars: [
    { x: 8, y: 12, r: 1.5 },
    { x: 33, y: 20, r: 1.4 },
    { x: 57, y: 11, r: 1.3 },
    { x: 39, y: 47, r: 2.5, accent: true }, // Antares
    { x: 45, y: 74, r: 1.4 },
    { x: 51, y: 97, r: 1.4 },
    { x: 65, y: 116, r: 1.3 },
    { x: 87, y: 126, r: 1.5 },
    { x: 103, y: 112, r: 1.8, accent: true }, // Shaula
    { x: 97, y: 94, r: 1.3 },
  ],
  lines: [
    [0, 1],
    [1, 2],
    [1, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [8, 9],
  ],
}

const CANCER: Figure = {
  name: 'Cancer',
  w: 92,
  h: 102,
  stars: [
    { x: 47, y: 7, r: 1.3 },
    { x: 40, y: 41, r: 1.5 }, // Asellus Borealis
    { x: 49, y: 59, r: 1.4 }, // Asellus Australis
    { x: 13, y: 85, r: 1.3 },
    { x: 77, y: 79, r: 1.2 },
  ],
  lines: [
    [0, 1],
    [1, 2],
    [2, 3],
    [2, 4],
  ],
}

function Constellation({
  figure,
  left,
  top,
  scale = 1,
}: {
  figure: Figure
  left: string
  top: string
  scale?: number
}) {
  const { w, h, stars, lines, name } = figure
  return (
    <svg
      width={w * scale}
      height={(h + 18) * scale}
      viewBox={`0 0 ${w} ${h + 18}`}
      style={{ position: 'absolute', left, top, overflow: 'visible' }}
      aria-hidden="true"
    >
      {lines.map(([a, b], i) => (
        <line
          key={i}
          x1={stars[a].x}
          y1={stars[a].y}
          x2={stars[b].x}
          y2={stars[b].y}
          stroke="rgba(255,255,255,0.13)"
          strokeWidth={0.6}
        />
      ))}
      {stars.map((s, i) => (
        <g key={i}>
          <circle
            cx={s.x}
            cy={s.y}
            r={s.r * 3.2}
            fill={s.accent ? 'rgba(255,180,120,0.10)' : 'rgba(255,255,255,0.07)'}
          />
          <circle
            cx={s.x}
            cy={s.y}
            r={s.r}
            fill={s.accent ? 'rgba(255,214,170,0.95)' : 'rgba(255,255,255,0.85)'}
          />
        </g>
      ))}
      <text
        x={0}
        y={h + 12}
        fill="rgba(255,255,255,0.22)"
        style={{
          fontSize: 7,
          letterSpacing: '0.22em',
          textTransform: 'uppercase',
          fontFamily: 'var(--font-mono, ui-monospace, monospace)',
        }}
      >
        {name}
      </text>
    </svg>
  )
}

export function Constellations() {
  // The constellations sit behind the hero's sun and headline, which is too
  // busy — hold them until the hero has scrolled past. Comets stay throughout.
  const [pastHero, setPastHero] = useState(false)

  useEffect(() => {
    const update = () => {
      const hero = document.getElementById('top')
      const threshold = hero ? hero.offsetHeight - window.innerHeight * 0.6 : 0
      setPastHero(window.scrollY > threshold)
    }
    update()
    window.addEventListener('scroll', update, { passive: true })
    window.addEventListener('resize', update)
    return () => {
      window.removeEventListener('scroll', update)
      window.removeEventListener('resize', update)
    }
  }, [])

  return (
    <div
      aria-hidden="true"
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 0,
        pointerEvents: 'none',
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          position: 'absolute',
          inset: 0,
          opacity: pastHero ? 1 : 0,
          transition: 'opacity 700ms ease',
        }}
      >
        <Constellation figure={URSA_MAJOR} left="5%" top="9%" scale={1.15} />
        <Constellation figure={CANCER} left="70%" top="44%" scale={1.05} />
        <Constellation figure={SCORPIUS} left="20%" top="52%" scale={1} />
      </div>
    </div>
  )
}
