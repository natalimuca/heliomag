import { Analytics } from '@vercel/analytics/next'
import type { Metadata, Viewport } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'heliomag · research log',
  description:
    'heliomag compares an AI solar foundation model against classical geomagnetic-activity indices — reading the Sun for warnings Earth\'s own numbers miss.',
}

export const viewport: Viewport = {
  colorScheme: 'dark',
  themeColor: '#000000',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body className="antialiased bg-surface text-ink">
        {children}
        {/* Vercel's analytics script is served by Vercel's edge, so on a static
            GitHub Pages export the beacon has nothing to fetch and every visit
            logs a 404 for /_vercel/insights/script.js. Gated at build time, so a
            Vercel deploy still gets analytics. */}
        {process.env.NODE_ENV === 'production' &&
          process.env.GITHUB_PAGES !== 'true' && <Analytics />}
      </body>
    </html>
  )
}
