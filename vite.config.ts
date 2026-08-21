import { defineConfig } from 'vite';

export default defineConfig({
  // GitHub Pages serves this repo at https://<user>.github.io/stable-fluids/
  // so every asset URL needs that prefix in production. Dev server uses '/'.
  base: process.env.NODE_ENV === 'production' ? '/stable-fluids/' : '/',
});
