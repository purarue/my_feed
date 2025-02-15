import path from "node:path"
import { fileURLToPath } from "node:url"

import { FlatCompat } from "@eslint/eslintrc"
import js from "@eslint/js"
import simpleImportSort from "eslint-plugin-simple-import-sort"

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const compat = new FlatCompat({
  baseDirectory: __dirname,
  recommendedConfig: js.configs.recommended,
  allConfig: js.configs.all,
})

const plugins = [
  ...compat.extends("next/core-web-vitals", "plugin:prettier/recommended"),
  {
    plugins: {
      "simple-import-sort": simpleImportSort,
    },

    rules: {
      "simple-import-sort/imports": "error",
    },
  },
]

export default plugins
