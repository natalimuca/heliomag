'use client'

import { useState } from 'react'
import { CountUp, Reveal, GroupedBars, DecayLine } from './charts'

const CARDS = [
  { k: 'Model', v: 'Surya', d: 'NASA/IBM heliophysics foundation model over solar imagery' },
  { k: 'Baseline', v: 'Ap / Kp', d: 'Persistence of the planetary geomagnetic index' },
  { k: 'Horizon', v: '3–14 days', d: 'Forecast lead window under evaluation' },
  { k: 'Record', v: 'Since 1932', d: 'Continuous ground-truth activity archive' },
]

const HEADLINE_STATS = [
  { v: 1941, d: 0, label: 'Surya embeddings extracted', sub: '2010-05 → 2024-12' },
  { v: 1785, d: 0, label: 'Aligned training rows', sub: '3.0× the original 598' },
  { v: 98.7, d: 1, suffix: '%', label: 'Weekly coverage', sub: '754 of 764 slots' },
  { v: 20, d: 0, label: 'Lead × target comparisons', sub: 'classical won every one' },
]

const STAGES = [
  {
    n: '01',
    t: 'Pilot correlation',
    d: 'Confirm the classical indices carry real signal, and that persistence decays fast enough to leave a window worth testing.',
    detail:
      'Sunspot number and F10.7 correlate with Ap at r ≈ 0.15–0.19 across the full 1932–2026 record — small, but with n in the tens of thousands it is not a sampling artifact. Ap autocorrelation starts at 0.461 for a 1-day lag and falls below the classical-index level by day 3, which is what defines the 3–14 day window.',
    result: 'Signal real but weak · ~2–3.5% of daily Ap variance',
  },
  {
    n: '02',
    t: 'Classical baseline',
    d: 'Fit the number Surya has to beat: persistence, linear SN+F10.7, the combination, and a small MLP.',
    detail:
      'Chronological split, train ≤ 2021-12-31 and test ≥ 2022-01-01. The linear SN+F10.7+persistence combination is strongest at every lead time, and an MLP on the same features does not improve on it — the classical relationship is close to linear.',
    result: 'Bar to clear · 14.34–14.64 RMSE across 3–14 days',
  },
  {
    n: '03',
    t: 'Probe & align',
    d: 'Extract embeddings on GPU, align them to the geomagnetic record, and linear-probe the frozen model.',
    detail:
      'PCA + Ridge with dimensionality and alpha selected per lead time by GridSearchCV over TimeSeriesSplit(5) on the training set only — no test leakage. The identical grid is applied to the classical features so the comparison is like for like.',
    result: '1,941 embeddings · 1,280-dim · validated on Colab + Kaggle',
  },
  {
    n: '04',
    t: 'Skill scoring',
    d: 'Compare RMSE per horizon, then stress-test the negative result three separate ways.',
    detail:
      'Beyond the head-to-head: a residual-correction variant where embeddings only explain what the classical fit gets wrong, Kp swapped in as an alternate target, and the training set tripled by densifying to every 3 days. None of the three flipped the outcome.',
    result: 'Classical wins 20 of 20 · no exceptions',
  },
]

// Ap autocorrelation by lag (SDO era) — why the 3–14 day window exists.
const PERSISTENCE = [
  { x: 1, y: 0.461 },
  { x: 2, y: 0.173 },
  { x: 3, y: 0.1 },
  { x: 4, y: 0.081 },
  { x: 7, y: 0.065 },
]

// Classical index vs Ap, Pearson r by lag (results/pilot_kp.md).
const CLASSICAL_CORR = [
  { lag: '0d', snFull: 0.158, snSdo: 0.155, f107Full: 0.168, f107Sdo: 0.168 },
  { lag: '1d', snFull: 0.162, snSdo: 0.153, f107Full: 0.175, f107Sdo: 0.174 },
  { lag: '2d', snFull: 0.165, snSdo: 0.156, f107Full: 0.184, f107Sdo: 0.18 },
  { lag: '3d', snFull: 0.165, snSdo: 0.156, f107Full: 0.183, f107Sdo: 0.188 },
  { lag: '5d', snFull: 0.162, snSdo: 0.153, f107Full: 0.17, f107Sdo: 0.167 },
  { lag: '7d', snFull: 0.155, snSdo: 0.142, f107Full: 0.162, f107Sdo: 0.15 },
  { lag: '14d', snFull: 0.132, snSdo: 0.111, f107Full: 0.141, f107Sdo: 0.123 },
]

const LEADS = ['3d', '5d', '7d', '10d', '14d']

// Held-out RMSE, densified training set (results/embedding_probe.md).
const SKILL: Record<'Ap' | 'Kp', { persistence: number[]; classical: number[]; embedding: number[] }> = {
  Ap: {
    persistence: [19.6, 19.77, 20.18, 20.13, 20.25],
    classical: [8.43, 15.32, 12.34, 8.41, 12.36],
    embedding: [8.7, 15.59, 14.46, 9.0, 12.92],
  },
  Kp: {
    persistence: [1.33, 1.47, 1.4, 1.34, 1.46],
    classical: [0.87, 1.1, 1.08, 0.88, 1.08],
    embedding: [0.94, 1.14, 1.43, 1.03, 1.27],
  },
}

// Every model variant tried, densified training set.
const VARIANTS: Record<
  'Ap' | 'Kp',
  { lead: string; classical: number; embed: number; embedCls: number; mlp: number; residual: number }[]
> = {
  Ap: [
    { lead: '3d', classical: 8.43, embed: 8.7, embedCls: 8.67, mlp: 23.76, residual: 8.56 },
    { lead: '5d', classical: 15.32, embed: 15.59, embedCls: 15.57, mlp: 19.46, residual: 15.58 },
    { lead: '7d', classical: 12.34, embed: 14.46, embedCls: 14.21, mlp: 13.75, residual: 14.6 },
    { lead: '10d', classical: 8.41, embed: 9.0, embedCls: 9.0, mlp: 12.0, residual: 9.15 },
    { lead: '14d', classical: 12.36, embed: 12.92, embedCls: 12.88, mlp: 16.91, residual: 12.64 },
  ],
  Kp: [
    { lead: '3d', classical: 0.87, embed: 0.94, embedCls: 0.93, mlp: 1.66, residual: 0.89 },
    { lead: '5d', classical: 1.1, embed: 1.14, embedCls: 1.15, mlp: 1.3, residual: 1.16 },
    { lead: '7d', classical: 1.08, embed: 1.43, embedCls: 1.44, mlp: 1.44, residual: 1.49 },
    { lead: '10d', classical: 0.88, embed: 1.03, embedCls: 1.03, mlp: 1.1, residual: 1.05 },
    { lead: '14d', classical: 1.08, embed: 1.27, embedCls: 1.26, mlp: 1.79, residual: 1.23 },
  ],
}

// Weekly-only run vs densified run — how much tripling the data moved things.
const DENSIFY = [
  { lead: '3d', weekly: 0.3, dense: 0.27 },
  { lead: '5d', weekly: 1.89, dense: 0.27 },
  { lead: '7d', weekly: 2.42, dense: 2.12 },
  { lead: '10d', weekly: -0.15, dense: 0.59 },
  { lead: '14d', weekly: 0.65, dense: 0.56 },
]

function Eyebrow({ children, index }: { children: string; index: string }) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        borderTop: '1px solid var(--line)',
        paddingTop: 16,
        marginBottom: 40,
      }}
    >
      <p
        className="font-mono"
        style={{
          color: 'var(--accent-text)',
          fontSize: 12,
          letterSpacing: '0.14em',
          textTransform: 'uppercase',
          margin: 0,
        }}
      >
        {children}
      </p>
      <span
        className="font-mono"
        style={{ fontSize: 11, letterSpacing: '0.16em', color: 'var(--ink-faint)' }}
      >
        {index}
      </span>
    </div>
  )
}

function TargetToggle({
  target,
  setTarget,
}: {
  target: 'Ap' | 'Kp'
  setTarget: (t: 'Ap' | 'Kp') => void
}) {
  return (
    <div
      role="group"
      aria-label="Choose forecast target"
      style={{
        display: 'inline-flex',
        border: '1px solid var(--line)',
        borderRadius: 999,
        padding: 4,
        gap: 4,
      }}
    >
      {(['Ap', 'Kp'] as const).map((t) => {
        const active = target === t
        return (
          <button
            key={t}
            onClick={() => setTarget(t)}
            aria-pressed={active}
            className="font-mono"
            style={{
              fontSize: 12,
              letterSpacing: '0.14em',
              textTransform: 'uppercase',
              padding: '8px 20px',
              borderRadius: 999,
              border: 'none',
              cursor: 'pointer',
              background: active ? 'var(--accent)' : 'transparent',
              color: active ? '#0a0a0a' : 'var(--ink-dim)',
              fontWeight: active ? 700 : 400,
              transition: 'background 160ms ease, color 160ms ease',
            }}
          >
            Target {t}
          </button>
        )
      })}
    </div>
  )
}

export function BelowFold() {
  const [target, setTarget] = useState<'Ap' | 'Kp'>('Ap')
  const [openStage, setOpenStage] = useState<number | null>(0)
  const [showAllCorr, setShowAllCorr] = useState(false)

  const skill = SKILL[target]
  const variants = VARIANTS[target]

  return (
    <div
      style={{
        position: 'relative',
        maxWidth: 1180,
        margin: '0 auto',
        padding: 'clamp(28px, 5vw, 72px) clamp(20px, 5vw, 88px) 120px',
      }}
    >
      {/* ---------- 03 · at a glance ---------- */}
      <div id="data" style={{ scrollMarginTop: 24 }} />
      <Eyebrow index="03 / 08">At a glance</Eyebrow>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: 14,
          marginBottom: 20,
        }}
      >
        {HEADLINE_STATS.map((s, i) => (
          <Reveal key={s.label} delay={i * 70}>
            <div
              style={{
                border: '1px solid var(--line)',
                borderRadius: 12,
                padding: '20px 20px 22px',
                background: 'rgba(13,15,20,0.4)',
                height: '100%',
              }}
            >
              <p
                className="font-mono"
                style={{ fontSize: 'clamp(26px, 3vw, 34px)', color: 'var(--gold-text)', margin: 0 }}
              >
                <CountUp to={s.v} decimals={s.d} suffix={s.suffix ?? ''} />
              </p>
              <p style={{ fontSize: 13.5, color: 'var(--ink)', margin: '10px 0 4px' }}>{s.label}</p>
              <p
                className="font-mono"
                style={{ fontSize: 11, color: 'var(--ink-faint)', margin: 0, letterSpacing: '0.04em' }}
              >
                {s.sub}
              </p>
            </div>
          </Reveal>
        ))}
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
          gap: 14,
        }}
      >
        {CARDS.map((c, i) => (
          <Reveal key={c.k} delay={i * 70}>
            <div
              style={{
                border: '1px solid var(--line)',
                borderRadius: 12,
                padding: '22px 22px 26px',
                background: 'rgba(13,15,20,0.4)',
                height: '100%',
              }}
            >
              <p
                className="font-mono"
                style={{
                  fontSize: 10,
                  letterSpacing: '0.14em',
                  textTransform: 'uppercase',
                  color: 'var(--ink-faint)',
                  margin: '0 0 14px',
                }}
              >
                {c.k}
              </p>
              <p
                className="display-condensed"
                style={{ fontSize: 26, color: 'var(--ink)', margin: '0 0 10px' }}
              >
                {c.v}
              </p>
              <p style={{ fontSize: 13, lineHeight: 1.5, color: 'var(--ink-dim)', margin: 0 }}>
                {c.d}
              </p>
            </div>
          </Reveal>
        ))}
      </div>

      {/* ---------- 04 · the arc ---------- */}
      <div id="arc" style={{ marginTop: 96, scrollMarginTop: 24 }}>
        <Eyebrow index="04 / 08">The arc</Eyebrow>
        <p
          style={{
            fontSize: 15,
            lineHeight: 1.6,
            color: 'var(--ink-dim)',
            maxWidth: 620,
            margin: '-16px 0 28px',
          }}
        >
          Four stages, each gating the next. Select one to see what it actually returned.
        </p>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {STAGES.map((s, i) => {
            const open = openStage === i
            return (
              <Reveal key={s.n} delay={i * 60}>
                <button
                  onClick={() => setOpenStage(open ? null : i)}
                  aria-expanded={open}
                  style={{
                    width: '100%',
                    textAlign: 'left',
                    border: '1px solid var(--line)',
                    borderRadius: 12,
                    background: open ? 'rgba(13,15,20,0.65)' : 'rgba(13,15,20,0.3)',
                    padding: '20px 22px',
                    cursor: 'pointer',
                    transition: 'background 200ms ease, border-color 200ms ease',
                    borderColor: open ? 'var(--accent)' : 'var(--line)',
                    color: 'inherit',
                    font: 'inherit',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'baseline', gap: 16 }}>
                    <span
                      className="font-mono"
                      style={{
                        fontSize: 22,
                        color: open ? 'var(--accent-text)' : 'var(--gold-text)',
                        letterSpacing: '0.04em',
                        transition: 'color 200ms ease',
                      }}
                    >
                      {s.n}
                    </span>
                    <span
                      className="display-condensed"
                      style={{ fontSize: 19, color: 'var(--ink)', flex: 1 }}
                    >
                      {s.t}
                    </span>
                    <span
                      className="font-mono"
                      style={{
                        fontSize: 18,
                        color: 'var(--ink-faint)',
                        transform: open ? 'rotate(45deg)' : 'none',
                        transition: 'transform 240ms cubic-bezier(.2,.7,.3,1)',
                      }}
                      aria-hidden="true"
                    >
                      +
                    </span>
                  </div>

                  <p
                    style={{
                      fontSize: 13.5,
                      lineHeight: 1.55,
                      color: 'var(--ink-dim)',
                      margin: '12px 0 0',
                      paddingLeft: 38,
                    }}
                  >
                    {s.d}
                  </p>

                  <div
                    style={{
                      display: 'grid',
                      gridTemplateRows: open ? '1fr' : '0fr',
                      transition: 'grid-template-rows 300ms cubic-bezier(.2,.7,.3,1)',
                    }}
                  >
                    <div style={{ overflow: 'hidden' }}>
                      <p
                        style={{
                          fontSize: 13.5,
                          lineHeight: 1.65,
                          color: 'var(--ink-dim)',
                          margin: '14px 0 0',
                          paddingLeft: 38,
                          borderLeft: '1px solid var(--line)',
                          marginLeft: 0,
                        }}
                      >
                        {s.detail}
                      </p>
                      <p
                        className="font-mono"
                        style={{
                          fontSize: 11,
                          letterSpacing: '0.08em',
                          textTransform: 'uppercase',
                          color: 'var(--accent-text)',
                          margin: '14px 0 2px',
                          paddingLeft: 38,
                        }}
                      >
                        → {s.result}
                      </p>
                    </div>
                  </div>
                </button>
              </Reveal>
            )
          })}
        </div>
      </div>

      {/* ---------- 05 · why this was worth trying ---------- */}
      <div style={{ marginTop: 96 }}>
        <Eyebrow index="05 / 08">Why this was worth trying</Eyebrow>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)',
            gap: 'clamp(24px, 4vw, 48px)',
            alignItems: 'start',
          }}
        >
          <Reveal>
            <h3
              className="display-condensed"
              style={{
                fontSize: 'clamp(20px, 2.2vw, 30px)',
                color: 'var(--ink)',
                margin: '0 0 14px',
                lineHeight: 1.08,
              }}
            >
              Persistence collapses. That is what opens the window.
            </h3>
            <p style={{ fontSize: 14.5, lineHeight: 1.65, color: 'var(--ink-dim)', margin: '0 0 18px' }}>
              Yesterday&apos;s Ap predicts today&apos;s well — autocorrelation is{' '}
              <span style={{ color: 'var(--ink)' }}>0.461</span> at a one-day lag. By day three it has
              fallen to <span style={{ color: 'var(--ink)' }}>0.100</span>, below what the classical
              indices themselves manage. Everything past that dashed line is forecasting the Sun, not
              echoing yesterday.
            </p>
            <DecayLine
              points={PERSISTENCE}
              refLevel={0.17}
              refLabel="CLASSICAL INDEX LEVEL ≈ 0.17"
            />
            <p
              className="font-mono"
              style={{
                fontSize: 10.5,
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                color: 'var(--ink-faint)',
                marginTop: 10,
              }}
            >
              Ap autocorrelation by lag · SDO era
            </p>
          </Reveal>

          <Reveal delay={120}>
            <div
              style={{
                border: '1px solid var(--line)',
                borderRadius: 14,
                background: 'rgba(13,15,20,0.4)',
                padding: 'clamp(20px, 2.6vw, 30px)',
              }}
            >
              <h3
                className="display-condensed"
                style={{
                  fontSize: 'clamp(18px, 2vw, 26px)',
                  color: 'var(--ink)',
                  margin: '0 0 14px',
                  lineHeight: 1.1,
                }}
              >
                The classical fit never touches the 27-day recurrence
              </h3>
              <p style={{ fontSize: 14, lineHeight: 1.6, color: 'var(--ink-dim)', margin: '0 0 22px' }}>
                Coronal holes rotate back into view roughly every 27 days, stamping a periodic ridge
                onto geomagnetic activity. The classical model&apos;s residual retains{' '}
                <span style={{ color: 'var(--gold-text)' }}>more</span> of that band than the raw
                signal — it removes almost none of it. That leftover periodicity is exactly what the
                Sun&apos;s imagery might explain.
              </p>
              <div style={{ display: 'flex', gap: 14 }}>
                <PowerMeter label="Raw Ap" pct={3.0} tone="var(--ink-faint)" />
                <PowerMeter label="Classical residual" pct={3.68} tone="var(--gold-text)" />
              </div>
              <p
                className="font-mono"
                style={{
                  fontSize: 10.5,
                  letterSpacing: '0.06em',
                  textTransform: 'uppercase',
                  color: 'var(--ink-faint)',
                  marginTop: 16,
                  marginBottom: 0,
                }}
              >
                Share of spectral power in the 25–29 day band
              </p>
            </div>
          </Reveal>
        </div>

        {/* classical correlation table */}
        <Reveal style={{ marginTop: 40 }}>
          <div
            style={{
              border: '1px solid var(--line)',
              borderRadius: 12,
              overflow: 'hidden',
              background: 'rgba(13,15,20,0.4)',
            }}
          >
            <div
              className="font-mono"
              style={{
                display: 'grid',
                gridTemplateColumns: '0.8fr repeat(4, 1fr)',
                fontSize: 10,
                letterSpacing: '0.12em',
                textTransform: 'uppercase',
                color: 'var(--ink-faint)',
                borderBottom: '1px solid var(--line)',
              }}
            >
              <span style={{ padding: '13px 18px' }}>Lag</span>
              <span style={{ padding: '13px 18px' }}>SN · full</span>
              <span style={{ padding: '13px 18px' }}>SN · SDO</span>
              <span style={{ padding: '13px 18px' }}>F10.7 · full</span>
              <span style={{ padding: '13px 18px' }}>F10.7 · SDO</span>
            </div>
            {CLASSICAL_CORR.slice(0, showAllCorr ? undefined : 4).map((r) => (
              <div
                key={r.lag}
                className="font-mono"
                style={{
                  display: 'grid',
                  gridTemplateColumns: '0.8fr repeat(4, 1fr)',
                  fontSize: 13,
                  color: 'var(--ink-dim)',
                  borderTop: '1px solid var(--line)',
                }}
              >
                <span style={{ padding: '12px 18px', color: 'var(--ink)' }}>{r.lag}</span>
                <span style={{ padding: '12px 18px' }}>{r.snFull.toFixed(3)}</span>
                <span style={{ padding: '12px 18px' }}>{r.snSdo.toFixed(3)}</span>
                <span style={{ padding: '12px 18px' }}>{r.f107Full.toFixed(3)}</span>
                <span style={{ padding: '12px 18px', color: 'var(--gold-text)' }}>
                  {r.f107Sdo.toFixed(3)}
                </span>
              </div>
            ))}
            <button
              onClick={() => setShowAllCorr((v) => !v)}
              className="font-mono"
              style={{
                width: '100%',
                background: 'transparent',
                border: 'none',
                borderTop: '1px solid var(--line)',
                color: 'var(--accent-text)',
                fontSize: 11,
                letterSpacing: '0.12em',
                textTransform: 'uppercase',
                padding: '13px 18px',
                cursor: 'pointer',
                textAlign: 'left',
              }}
            >
              {showAllCorr ? '− Show fewer lags' : `+ Show all ${CLASSICAL_CORR.length} lags`}
            </button>
          </div>
          <p
            className="font-mono"
            style={{
              fontSize: 10.5,
              letterSpacing: '0.06em',
              textTransform: 'uppercase',
              color: 'var(--ink-faint)',
              marginTop: 12,
            }}
          >
            Pearson r vs. Ap · n = 28,366–34,544 (full) · 5,922–5,934 (SDO era)
          </p>
        </Reveal>
      </div>

      {/* ---------- 06 · forecast skill ---------- */}
      <div id="verdict" style={{ marginTop: 96, scrollMarginTop: 24 }}>
        <Eyebrow index="06 / 08">Forecast skill by horizon</Eyebrow>

        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 24,
            marginBottom: 30,
            flexWrap: 'wrap',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 22, flexWrap: 'wrap' }}>
            <LegendDot color="#3a4150" label="Persistence" />
            <LegendDot color="var(--ink-faint)" label="Classical baseline" />
            <LegendDot color="var(--accent)" label="Surya embedding probe" />
          </div>
          <TargetToggle target={target} setTarget={setTarget} />
        </div>

        <GroupedBars
          categories={LEADS}
          unit="RMSE"
          series={[
            { label: 'Persistence', color: '#3a4150', values: skill.persistence },
            { label: 'Classical', color: '#6b7383', values: skill.classical },
            { label: 'Embedding', color: 'var(--accent)', values: skill.embedding },
          ]}
        />

        <p
          className="font-mono"
          style={{
            fontSize: 11,
            letterSpacing: '0.08em',
            color: 'var(--ink-faint)',
            marginTop: 20,
            textTransform: 'uppercase',
          }}
        >
          Held-out RMSE on target {target} · lower is better · hover a bar for its value ·
          outlined bar = best at that lead
        </p>
      </div>

      {/* ---------- 07 · every variant ---------- */}
      <div id="residual" style={{ marginTop: 96, scrollMarginTop: 24 }}>
        <Eyebrow index="07 / 08">Every variant tried</Eyebrow>
        <p
          style={{
            fontSize: 15,
            lineHeight: 1.6,
            color: 'var(--ink-dim)',
            maxWidth: 660,
            margin: '-16px 0 28px',
          }}
        >
          Five model families against the same split. Embeddings compete head-on, combined with the
          classical features, through a nonlinear MLP, and finally as a residual correction that only
          has to explain what the classical fit gets wrong. The best score in each row is highlighted.
        </p>

        <Reveal>
          <div style={{ overflowX: 'auto' }}>
            <div style={{ minWidth: 640 }}>
              <div
                className="font-mono"
                style={{
                  display: 'grid',
                  gridTemplateColumns: '0.7fr repeat(5, 1fr)',
                  fontSize: 10,
                  letterSpacing: '0.1em',
                  textTransform: 'uppercase',
                  color: 'var(--ink-faint)',
                  borderBottom: '1px solid var(--line)',
                }}
              >
                <span style={{ padding: '13px 14px' }}>Lead</span>
                <span style={{ padding: '13px 14px' }}>Classical</span>
                <span style={{ padding: '13px 14px' }}>Embed</span>
                <span style={{ padding: '13px 14px' }}>Embed + cls</span>
                <span style={{ padding: '13px 14px' }}>MLP</span>
                <span style={{ padding: '13px 14px' }}>Residual</span>
              </div>
              {variants.map((r) => {
                const vals = [r.classical, r.embed, r.embedCls, r.mlp, r.residual]
                const best = Math.min(...vals)
                return (
                  <div
                    key={r.lead}
                    className="font-mono"
                    style={{
                      display: 'grid',
                      gridTemplateColumns: '0.7fr repeat(5, 1fr)',
                      fontSize: 13.5,
                      color: 'var(--ink-dim)',
                      borderBottom: '1px solid var(--line)',
                    }}
                  >
                    <span style={{ padding: '13px 14px', color: 'var(--ink)' }}>{r.lead}</span>
                    {vals.map((v, i) => (
                      <span
                        key={i}
                        style={{
                          padding: '13px 14px',
                          color: v === best ? '#0a0a0a' : 'var(--ink-dim)',
                          background: v === best ? 'var(--gold)' : 'transparent',
                          fontWeight: v === best ? 700 : 400,
                        }}
                      >
                        {v.toFixed(2)}
                      </span>
                    ))}
                  </div>
                )
              })}
            </div>
          </div>
        </Reveal>

        <p
          className="font-mono"
          style={{
            fontSize: 11,
            letterSpacing: '0.08em',
            color: 'var(--ink-faint)',
            marginTop: 16,
            textTransform: 'uppercase',
          }}
        >
          Target {target} · densified training set · classical takes all 5 rows
        </p>
      </div>

      {/* ---------- 08 · where this leaves it ---------- */}
      <div id="where-next" style={{ marginTop: 96, scrollMarginTop: 24 }}>
        <Eyebrow index="08 / 08">Where this leaves it</Eyebrow>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'minmax(0, 1.35fr) minmax(0, 1fr)',
            gap: 'clamp(24px, 4vw, 56px)',
            alignItems: 'start',
          }}
        >
          <Reveal>
            <h3
              className="display-condensed"
              style={{
                fontSize: 'clamp(22px, 2.4vw, 34px)',
                color: 'var(--ink)',
                margin: '0 0 16px',
                lineHeight: 1.05,
              }}
            >
              Sample size mattered — just not enough to flip the result
            </h3>
            <p style={{ fontSize: 15, lineHeight: 1.65, color: 'var(--ink-dim)', margin: '0 0 16px' }}>
              Tripling the training set (598 → 1,785 rows) didn&apos;t change the headline finding.
              But it wasn&apos;t a wash: at lead 5 the gap between classical and the embedding probe
              collapsed, real evidence that sample size was a genuine, partial constraint rather than
              a convenient excuse. Other lead times barely moved.
            </p>
            <p style={{ fontSize: 15, lineHeight: 1.65, color: 'var(--ink-dim)', margin: 0 }}>
              Full daily-density extraction would test that properly, but at roughly 38 hours of
              additional compute for an uncertain, likely-partial payoff it isn&apos;t the
              highest-value next step. If this is picked up again, revisiting embedding pooling and
              resolution — not more of the same weekly-derived signal — is the more promising lever.
            </p>
          </Reveal>

          <Reveal delay={120}>
            <div
              style={{
                border: '1px solid var(--line)',
                borderRadius: 14,
                background: 'rgba(13,15,20,0.4)',
                padding: '22px 22px 26px',
              }}
            >
              <p
                className="font-mono"
                style={{
                  fontSize: 10,
                  letterSpacing: '0.14em',
                  textTransform: 'uppercase',
                  color: 'var(--ink-faint)',
                  margin: '0 0 18px',
                }}
              >
                Gap to classical, RMSE · weekly → densified
              </p>
              {DENSIFY.map((d, i) => {
                const improved = d.dense < d.weekly
                const scale = 2.6
                return (
                  <div key={d.lead} style={{ marginBottom: i === DENSIFY.length - 1 ? 0 : 14 }}>
                    <div
                      className="font-mono"
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        fontSize: 11.5,
                        color: 'var(--ink-dim)',
                        marginBottom: 6,
                      }}
                    >
                      <span style={{ color: 'var(--ink)' }}>{d.lead}</span>
                      <span>
                        {d.weekly.toFixed(2)} →{' '}
                        <span style={{ color: improved ? 'var(--gold-text)' : 'var(--rust, #ff5c5c)' }}>
                          {d.dense.toFixed(2)}
                        </span>
                      </span>
                    </div>
                    <div style={{ position: 'relative', height: 6 }}>
                      <div
                        style={{
                          position: 'absolute',
                          inset: 0,
                          background: 'rgba(255,255,255,0.06)',
                          borderRadius: 3,
                        }}
                      />
                      <div
                        style={{
                          position: 'absolute',
                          left: 0,
                          top: 0,
                          bottom: 0,
                          width: `${Math.min(100, Math.abs(d.weekly) * scale * 10)}%`,
                          background: 'var(--ink-faint)',
                          borderRadius: 3,
                          opacity: 0.5,
                        }}
                      />
                      <div
                        style={{
                          position: 'absolute',
                          left: 0,
                          top: 0,
                          bottom: 0,
                          width: `${Math.min(100, Math.abs(d.dense) * scale * 10)}%`,
                          background: improved ? 'var(--gold)' : 'var(--accent)',
                          borderRadius: 3,
                        }}
                      />
                    </div>
                  </div>
                )
              })}
              <p
                style={{
                  fontSize: 12.5,
                  lineHeight: 1.55,
                  color: 'var(--ink-faint)',
                  margin: '20px 0 0',
                }}
              >
                Lead 5 closed from 1.89 to 0.27. Lead 10 reversed — the one apparent embedding win in
                the weekly run did not survive more data, and is retracted.
              </p>
            </div>
          </Reveal>
        </div>

        {/* coverage caveat */}
        <Reveal style={{ marginTop: 34 }}>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
              gap: 14,
            }}
          >
            {[
              { h: '754 / 764', d: 'weekly slots present (98.7%). The 10 gaps are genuine SDO source-data holes, confirmed by a dedicated re-run.' },
              { h: '30 / 1,416', d: 'densified dates failed (2.1%), clustered around the same known instrument outages.' },
              { h: '2025', d: 'blocked upstream — the NASA Surya benchmark bucket does not yet mirror 2025 data.' },
            ].map((c) => (
              <div
                key={c.h}
                style={{
                  border: '1px solid var(--line)',
                  borderRadius: 12,
                  padding: '18px 18px 20px',
                  background: 'rgba(13,15,20,0.3)',
                }}
              >
                <p
                  className="font-mono"
                  style={{ fontSize: 17, color: 'var(--ink)', margin: '0 0 10px' }}
                >
                  {c.h}
                </p>
                <p style={{ fontSize: 12.5, lineHeight: 1.55, color: 'var(--ink-faint)', margin: 0 }}>
                  {c.d}
                </p>
              </div>
            ))}
          </div>
        </Reveal>
      </div>

      {/* closing footer */}
      <div
        style={{
          marginTop: 110,
          borderTop: '1px solid var(--line)',
          paddingTop: 22,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 16,
          flexWrap: 'wrap',
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
          style={{ fontSize: 11, letterSpacing: '0.16em', color: 'var(--ink-faint)' }}
        >
          08 / 08
        </span>
      </div>
    </div>
  )
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <span style={{ width: 12, height: 12, borderRadius: 3, background: color }} />
      <span
        className="font-mono"
        style={{
          fontSize: 12,
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          color: 'var(--ink-dim)',
        }}
      >
        {label}
      </span>
    </div>
  )
}

function PowerMeter({ label, pct, tone }: { label: string; pct: number; tone: string }) {
  const fill = Math.min(pct / 4.2, 1)
  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}>
      <div
        style={{
          position: 'relative',
          width: '100%',
          minWidth: 54,
          height: 140,
          borderRadius: 8,
          border: '1px solid var(--line)',
          overflow: 'hidden',
          display: 'flex',
          alignItems: 'flex-end',
        }}
      >
        <div style={{ width: '100%', height: `${fill * 100}%`, background: tone, opacity: 0.55 }} />
        <span
          className="font-mono"
          style={{
            position: 'absolute',
            top: 10,
            left: 0,
            right: 0,
            textAlign: 'center',
            fontSize: 15,
            fontWeight: 600,
            color: 'var(--ink)',
          }}
        >
          {pct.toFixed(2)}%
        </span>
      </div>
      <span
        className="font-mono"
        style={{
          fontSize: 10,
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          color: 'var(--ink-faint)',
          textAlign: 'center',
          lineHeight: 1.3,
        }}
      >
        {label}
      </span>
    </div>
  )
}
