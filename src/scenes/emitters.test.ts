import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createFields, createGrid, idxP, idxU } from '../core/grid.ts';
import { computeDivergence } from '../core/divergence.ts';
import { Simulation } from '../core/simulation.ts';
import { defaultWallJet, wallJet } from './emitters.ts';

const N = 32;
const setup = () => {
  const g = createGrid(N, N, 1 / N);
  const f = createFields(g, Float64Array);
  return { g, f, div: new Float64Array(f.p.length) };
};

test('the jet is genuinely an inflow (otherwise the rest is vacuous)', () => {
  const { g, f } = setup();
  wallJet().seed(g, f.u, f.v);

  let peak = 0;
  for (let j = 0; j < g.ny; j++) peak = Math.max(peak, f.u[idxU(g, 0, j)]);
  assert.ok(peak > 0.9 * defaultWallJet.speed, `peak inflow ${peak}, expected ~speed`);
  assert.ok(
    f.v.every((x) => x === 0),
    'v must stay untouched — those are interior faces the projection owns',
  );
});

test('inflow flux is the same scene at every resolution', () => {
  // The taper is in WORLD units precisely so this holds. A cell-width ramp
  // drifts this 20% between N=64 and N=512, and a top-hat quantizes the band
  // to whole cells — either way two resolutions stop being comparable, which
  // is exactly what a CPU-vs-GPU or 128-vs-256 study depends on.
  const flux = (n: number) => {
    const g = createGrid(n, n, 1 / n);
    const f = createFields(g, Float64Array);
    wallJet().seed(g, f.u, f.v);
    let s = 0;
    for (let j = 0; j < g.ny; j++) s += f.u[idxU(g, 0, j)] * g.h;
    return s;
  };

  const coarse = flux(64);
  const fine = flux(256);
  assert.ok(
    Math.abs(fine - coarse) / coarse < 2e-3,
    `inflow flux ${coarse} at N=64 vs ${fine} at N=256 — the scene changed with the grid`,
  );
});

test('inflow and outflow balance, so the pressure system stays compatible', () => {
  const { g, f, div } = setup();
  wallJet().seed(g, f.u, f.v);
  computeDivergence(g, f.u, f.v, div);

  let sum = 0;
  for (const d of div) sum += d;
  // Machine precision, not O(h): an inconsistent RHS makes SOR burn every
  // sweep and drift p by a growing constant.
  assert.ok(Math.abs(sum) < 1e-9, `total divergence = ${sum}, expected ~0`);
});

test('prescribed wall faces survive advection and projection unchanged', () => {
  const sim = new Simulation(N, { pressureIters: 200 });
  const jet = wallJet();
  sim.reset(jet.seed, undefined, jet.source);
  const wall = sim.f.u.slice();

  for (let n = 0; n < 60; n++) sim.step();

  for (let j = 0; j < sim.g.ny; j++) {
    for (const i of [0, sim.g.nx]) {
      const k = idxU(sim.g, i, j);
      assert.equal(sim.f.u[k], wall[k], `wall face u[${i},${j}] was clobbered`);
    }
  }
  assert.ok(sim.f.u.every(Number.isFinite), 'u went non-finite');
  assert.ok(sim.iters < 200, `solve pinned at the cap (${sim.iters}) — RHS likely inconsistent`);
});

test('the dye source holds its band and carries downstream', () => {
  const sim = new Simulation(N, { pressureIters: 200 });
  const jet = wallJet({ depthCells: 2 });
  sim.reset(jet.seed, undefined, jet.source);

  const mid = Math.floor(N / 2);
  const [r, g, b] = sim.f.dye;
  const at = (i: number) => [r, g, b].map((c) => c[idxP(sim.g, i, mid)]);

  // Seeded once by reset(), before any step.
  assert.deepEqual(at(0), [...defaultWallJet.colour]);

  for (let n = 0; n < 60; n++) sim.step();

  assert.deepEqual(at(0), [...defaultWallJet.colour], 'source did not hold its value');
  const far = at(mid)[2];
  assert.ok(far > 0.05, `dye never reached mid-domain (blue = ${far})`);
  assert.ok(
    sim.f.dye.every((c) => c.every((x) => x >= -1e-12 && x <= 1 + 1e-12)),
    'overwrite semantics should keep dye inside [0, 1] with no clamping',
  );
});

test('uniform through-flow is an exact steady solution', () => {
  // u = U everywhere (walls included), v = 0: already divergence-free, so
  // advect + project must return it bit-for-bit. Catches sign errors, the
  // wall-face bounds in subtractGradient, and half-cell offsets in sampling.
  const U = 0.5;
  const sim = new Simulation(N, { pressureIters: 200 });
  sim.reset((g, u, v) => {
    u.fill(U);
    v.fill(0);
  });

  for (let n = 0; n < 20; n++) sim.step();

  assert.ok(
    sim.f.u.every((x) => x === U),
    'a uniform through-flow was perturbed',
  );
  assert.ok(
    sim.f.v.every((x) => x === 0),
    'projection invented a transverse velocity',
  );
});
