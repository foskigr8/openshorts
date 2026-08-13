import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'

/**
 * Flat config, written for the ESLint that is actually installed here (8.57).
 *
 * The previous version imported `eslint/config` and used `defineConfig` /
 * `globalIgnores`, which are ESLint 9 APIs — so `npm run lint` crashed on
 * startup and nobody could run it. A linter that cannot be run catches
 * nothing: an undefined `setRightTab` shipped and broke the generate button,
 * and two missing icon imports crashed the workspace on render, with a green
 * build both times. Vite does not check for undefined identifiers; this does.
 */
export default [
  { ignores: ['dist/**', 'node_modules/**'] },
  {
    files: ['**/*.{js,jsx}'],
    ...js.configs.recommended,
    plugins: { 'react-hooks': reactHooks },
    languageOptions: {
      ecmaVersion: 2022,
      globals: { ...globals.browser, ...globals.es2021 },
      parserOptions: {
        ecmaVersion: 'latest',
        ecmaFeatures: { jsx: true },
        sourceType: 'module',
      },
    },
    rules: {
      ...js.configs.recommended.rules,
      // The two that matter: a name that does not exist, and an import that
      // was removed while its use stayed behind.
      'no-undef': 'error',
      'no-unused-vars': ['warn', {
        varsIgnorePattern: '^[A-Z_]',
        // A capitalised parameter is a component rendered as <Icon/>, which
        // this rule cannot see through JSX.
        argsIgnorePattern: '^(_|[A-Z])',
        caughtErrors: 'none',
      }],
      'react-hooks/rules-of-hooks': 'error',
    },
  },
  {
    // Build/config files run in Node, not the browser.
    files: ['*.config.js', 'vite.config.js', 'postcss.config.js', 'tailwind.config.js'],
    languageOptions: { globals: { ...globals.node } },
  },
]
