import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';

export default defineConfig({
  site: 'https://izzet.github.io',
  base: '/agentstage',
  integrations: [mdx(), sitemap()],
});
