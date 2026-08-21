import { viridis } from './viz/colormaps.ts';
import { Heatmap } from './viz/heatmap.ts';

const canvas = document.querySelector<HTMLCanvasElement>('#view')!;
const ctx = canvas.getContext('2d')!;
const readout = document.querySelector<HTMLPreElement>('#readout')!;

const N = 64;
const heatmap = new Heatmap(N, N);

const grid = new Float64Array(N * N);
for (let j = 0; j < N; j++) {
  for (let i = 0; i < N; i++) {
    grid[i + j * N] = Math.sin(j) * Math.cos(i);
  }
}

heatmap.draw(grid, ctx, {
  normalization: { kind: 'auto' },
  colormap: viridis,
});

// lastMin/lastMax exist specifically because the picture alone can't tell
// you whether a field converged — only the numbers can.
readout.textContent = `range: [${heatmap.lastMin.toFixed(4)}, ${heatmap.lastMax.toFixed(4)}]`;
