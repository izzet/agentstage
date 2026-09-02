import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';

export default defineConfig({
  site: 'https://izzet.github.io',
  base: '/agentstage',
  integrations: [mdx()],
});
