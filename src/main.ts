import { computeDivergence } from './core/divergence.ts';
import { createFields, createGrid, idxU } from './core/grid.ts';
import { viridis, coolwarm } from './viz/colormaps.ts';
import { Heatmap } from './viz/heatmap.ts';

const canvas = document.querySelector<HTMLCanvasElement>('#view')!;
const ctx = canvas.getContext('2d')!;
const readout = document.querySelector<HTMLPreElement>('#readout')!;

const N = 64; // grid dimensions.
const H = 0.5; // grid spacing.
const heatmap = new Heatmap(N, N);

const g = createGrid(N, N, H);
const f = createFields(g, Float64Array);

for (let j = 0; j < g.ny; j++) {
  for (let i = 0; i <= g.nx; i++) {
    f.u[idxU(g, i, j)] = Math.sin(j / N) * Math.cos(i / N);
  }
}

const div = new Float64Array(f.p.length);
computeDivergence(g, f.u, f.v, div);

heatmap.draw(div, ctx, {
  normalization: { kind: 'symmetric' },
  colormap: coolwarm,
});

// lastMin/lastMax exist specifically because the picture alone can't tell
// you whether a field converged — only the numbers can.
readout.textContent = `range: [${heatmap.lastMin.toFixed(4)}, ${heatmap.lastMax.toFixed(4)}]`;
