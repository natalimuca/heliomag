/** @type {import('next').NextConfig} */

// GitHub Pages serves a project site from a subpath (/heliomag) and can only
// serve static files. Both settings are gated behind this flag so a normal
// `npm run build` — or a future Vercel deploy — is unaffected. The Pages
// workflow sets GITHUB_PAGES=true.
const isPages = process.env.GITHUB_PAGES === 'true'
const repo = 'heliomag'

const nextConfig = {
  ...(isPages
    ? {
        output: 'export',
        basePath: `/${repo}`,
        assetPrefix: `/${repo}/`,
        trailingSlash: true,
      }
    : {}),
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
}

export default nextConfig
