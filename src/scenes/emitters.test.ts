import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Cell, createFields, createGrid, idxP, idxU } from '../core/grid.ts';
import { computeDivergence } from '../core/divergence.ts';
import { Simulation, type LabelSeed } from '../core/simulation.ts';
import { defaultWallJet, wallJet } from './emitters.ts';
import { openRight } from './obstacles.ts';

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

test('the jet is pure inflow, so the books do NOT balance on their own', () => {
  // Guards the contract change: wallJet used to push the same flux back out
  // across the far wall, which balanced the budget but PRESCRIBED the exit
  // profile and made the boundary reflect. Settling the budget is now the
  // scene's job, and this asserts the emitter really has stopped doing it.
  const { g, f, div } = setup();
  wallJet().seed(g, f.u, f.v);
  computeDivergence(g, f.u, f.v, div);

  // By the divergence theorem, sum(div) * h^2 is the net OUTWARD flux, and the
  // interior faces telescope away — so it must come out as exactly minus the
  // inlet flux. Negative, because mass is entering. Machine precision, not
  // O(h): this is a discrete identity, not an approximation.
  let sum = 0;
  for (const d of div) sum += d;

  let inlet = 0;
  for (let j = 0; j < g.ny; j++) inlet += f.u[idxU(g, 0, j)] * g.h;

  assert.ok(inlet > 0.1, `inlet flux ${inlet} — the jet is not actually flowing`);
  assert.ok(
    Math.abs(sum * g.h * g.h + inlet) < 1e-12,
    `net outward flux ${sum * g.h * g.h}, expected ${-inlet} — the emitter is ` +
      `balancing its own books again, which prescribes the exit profile`,
  );
});

test('a net inflow needs an open outlet, and one Air column provides it', () => {
  // A closed box is all-Neumann, so a net inflow makes the system INCONSISTENT:
  // no p satisfies it, SOR burns every sweep against a residual floor it can
  // never cross, and the null-space component drives p away without bound.
  // One Air column pins p = 0 and the same problem becomes solvable.
  const CAP = 120;
  const run = (labels?: LabelSeed) => {
    const sim = new Simulation(N, N, { pressureIters: CAP });
    sim.reset({ labels, seed: wallJet().seed });
    for (let n = 0; n < 60; n++) sim.step();

    let pMax = 0;
    for (const x of sim.f.p) pMax = Math.max(pMax, Math.abs(x));
    let sumSq = 0;
    let count = 0;
    for (let k = 0; k < sim.div.length; k++) {
      if (sim.f.label[k] !== Cell.Fluid) continue;
      sumSq += sim.div[k] * sim.div[k];
      count += 1;
    }
    return { iters: sim.iters, pMax, divRms: Math.sqrt(sumSq / count) };
  };

  const closed = run();
  const open = run(openRight());

  assert.equal(closed.iters, CAP, 'a closed box should never meet tolerance here');
  assert.ok(open.iters < CAP, `open outlet still pinned at the cap (${open.iters})`);
  // Measured at N=32: 430x on the residual, 280x on the drift. 100x is slack.
  assert.ok(
    open.divRms * 100 < closed.divRms,
    `residual ${open.divRms} open vs ${closed.divRms} closed`,
  );
  assert.ok(open.pMax * 100 < closed.pMax, `p drifted to ${open.pMax} despite an outlet`);
});

test('prescribed wall faces survive advection and projection unchanged', () => {
  const sim = new Simulation(N, N, { pressureIters: 200 });
  const jet = wallJet();
  sim.reset({ labels: openRight(), seed: jet.seed, dyeSource: jet.source });
  const wall = sim.f.u.slice();

  for (let n = 0; n < 60; n++) sim.step();

  for (let j = 0; j < sim.g.ny; j++) {
    // The INLET only. u[nx,j] is no longer prescribed: it fronts an Air cell,
    // so applyOutflow extrapolates it every step — that is the open boundary.
    const k = idxU(sim.g, 0, j);
    assert.equal(sim.f.u[k], wall[k], `inlet face u[0,${j}] was clobbered`);
    assert.equal(
      sim.f.u[idxU(sim.g, sim.g.nx, j)],
      sim.f.u[idxU(sim.g, sim.g.nx - 1, j)],
      `outlet face u[nx,${j}] is not a zero-gradient copy of its neighbour`,
    );
  }
  assert.ok(sim.f.u.every(Number.isFinite), 'u went non-finite');
  assert.ok(sim.iters < 200, `solve pinned at the cap (${sim.iters}) — RHS likely inconsistent`);
});

test('the dye source holds its band and carries downstream', () => {
  const sim = new Simulation(N, N, { pressureIters: 200 });
  const jet = wallJet({ depthCells: 2 });
  sim.reset({ labels: openRight(), seed: jet.seed, dyeSource: jet.source });

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
  const sim = new Simulation(N, N, { pressureIters: 200 });
  sim.reset({
    seed: (g, u, v) => {
      u.fill(U);
      v.fill(0);
    },
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
