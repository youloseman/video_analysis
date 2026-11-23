// The rules that catch what the 2026-11-23 audit found by hand: a second
// `function copyText` silently replaced the first for the whole app. One
// classic script, so every top-level name is a global and no-undef sees them.
import globals from "globals";

export default [
  {
    files: ["build/app.js"],
    languageOptions: {
      ecmaVersion: 2023,
      sourceType: "script",
      globals: { ...globals.browser },
    },
    rules: {
      "no-redeclare": "error",
      "no-undef": "error",
      "no-dupe-keys": "error",
      "no-dupe-else-if": "error",
      "no-unreachable": "error",
      "no-self-assign": "error",
      "no-cond-assign": "error",
      "no-const-assign": "error",
      "no-func-assign": "error",
      "use-isnan": "error",
      "valid-typeof": "error",
      // Five today, all harmless; a warning so they are seen, not a gate.
      "no-unused-vars": ["warn", { args: "none", caughtErrors: "none", ignoreRestSiblings: true }],
    },
  },
];
