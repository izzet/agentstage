# AgentStage project site

The static research page for AgentStage. It is kept separate from the Python
package so the website toolchain does not affect package development.

## Local preview

```bash
cd site
npm install
npm run dev
```

Use `npm run build` to type-check and create the static production site in
`site/dist/`.

## Release policy

This branch deliberately contains no GitHub Pages deployment workflow and no
paper PDF. The page states that the accepted manuscript is pending public
release. Add a final, approved paper only when its public-release status is
clear, then link it from `src/pages/index.astro`.
