/** @type {import('next').NextConfig} */

const useBasePath = process.env.BASE_PATH ?? "/feed"
const basePath = process.env.NODE_ENV === "production" ? useBasePath : undefined

const nextConfig = {
  reactStrictMode: true,
  basePath: basePath,
  images: {
    domains: ["localhost", "purarue.xyz"],
    loader: "custom",
  },
  i18n: {
    locales: ["en"],
    defaultLocale: "en",
  },
}

module.exports = nextConfig
