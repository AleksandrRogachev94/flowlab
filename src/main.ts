import { computeDivergence } from './core/divergence.ts';
import { createFields, createGrid } from './core/grid.ts';
import { solvePressure } from './core/pressure.ts';
import { subtractGradient } from './core/subtractGradient.ts';
import { addGradient, addRotational } from './scenes/testFields.ts';
import { viridis, coolwarm } from './viz/colormaps.ts';
import { Heatmap } from './viz/heatmap.ts';
import { defaultVectorOptions, drawVectors } from './viz/vectors.ts';

const canvas = document.querySelector<HTMLCanvasElement>('#view')!;
const ctx = canvas.getContext('2d')!;
const readout = document.querySelector<HTMLPreElement>('#readout')!;

const N = 64; // grid dimensions.
const H = 1 / N; // unit square — testFields.ts assumes h = 1/nx.
const RHO = 1.0;
const DT = 1.0;
const heatmap = new Heatmap(N, N);

const g = createGrid(N, N, H);
const f = createFields(g, Float64Array);

addRotational(g, f.u, f.v);
addGradient(g, f.u, f.v);

const div = new Float64Array(f.p.length);
computeDivergence(g, f.u, f.v, div);

const scale = (RHO * H * H) / DT;
const gradScale = DT / (RHO * H);
solvePressure(g, f.p, div, f.label, scale, 1000, 1.9);
subtractGradient(g, f.p, f.u, f.v, f.label, gradScale);

// re-compute divergence
computeDivergence(g, f.u, f.v, div);

heatmap.draw(div, ctx, {
  normalization: { kind: 'symmetric' },
  colormap: coolwarm,
});
drawVectors(ctx, g, f.u, f.v, { ...defaultVectorOptions, mode: 'cell' });

// lastMin/lastMax exist specifically because the picture alone can't tell
// you whether a field converged — only the numbers can.
readout.textContent = `Divergence range: [${heatmap.lastMin.toFixed(4)}, ${heatmap.lastMax.toFixed(4)}]`;
