import { defineConfig } from 'vite';

export default defineConfig({
  // GitHub Pages serves this repo at https://<user>.github.io/flowlab/
  // so every asset URL needs that prefix in production. Dev server uses '/'.
  base: process.env.NODE_ENV === 'production' ? '/flowlab/' : '/',
});
