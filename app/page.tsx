import { HeroScroll } from '@/components/hero-scroll'
import { ReadoutSection } from '@/components/readout-section'
import { BelowFold } from '@/components/below-fold'
import { Constellations } from '@/components/constellations'

export default function Page() {
  return (
    <>
      <Constellations />
      <main style={{ position: 'relative', zIndex: 1 }}>
        <HeroScroll />
        <ReadoutSection />
        <BelowFold />
      </main>
    </>
  )
}
