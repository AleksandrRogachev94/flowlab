import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createGrid, createFields, idxP, idxU, idxV, type FieldArray } from './grid.ts';
import {
  advectScalar,
  advectScalarMacCormack,
  advectVelocity,
  advectVelocityMacCormack,
} from './advect.ts';
import { computeDivergence } from './divergence.ts';
import { solvePressure } from './pressure.ts';
import { subtractGradient } from './subtractGradient.ts';
import { addVortexPair } from '../scenes/testFields.ts';

const H = 0.5; // deliberately not 1.0, so a missing "/h" can't hide

/** Fresh output buffers, NaN-filled so an unwritten face is loud, not silent. */
function outputs(g: ReturnType<typeof createGrid>) {
  return {
    uOut: new Float64Array((g.nx + 1) * g.ny).fill(NaN),
    vOut: new Float64Array(g.nx * (g.ny + 1)).fill(NaN),
  };
}

test('every output face gets written — no stale values left behind', () => {
  const g = createGrid(4, 4, H);
  const f = createFields(g, Float64Array);
  f.u.fill(0.3);
  f.v.fill(-0.2);
  const { uOut, vOut } = outputs(g);

  advectVelocity(g, f.u, f.v, uOut, vOut, f.label, 0.25 * H);

  // Catches both wrong loop bounds (i < nx instead of i <= nx) and a wrong
  // stride (idxU where idxV belongs): either leaves NaN holes behind.
  for (let j = 0; j < g.ny; j++) {
    for (let i = 0; i <= g.nx; i++) {
      assert.ok(Number.isFinite(uOut[idxU(g, i, j)]), `uOut face (${i},${j}) never written`);
    }
  }
  for (let j = 0; j <= g.ny; j++) {
    for (let i = 0; i < g.nx; i++) {
      assert.ok(Number.isFinite(vOut[idxV(g, i, j)]), `vOut face (${i},${j}) never written`);
    }
  }
});

test('a uniform velocity field advects to itself', () => {
  const g = createGrid(4, 4, H);
  const f = createFields(g, Float64Array);
  const U = 0.75;
  const V = -0.25;
  f.u.fill(U);
  f.v.fill(V);
  const { uOut, vOut } = outputs(g);

  advectVelocity(g, f.u, f.v, uOut, vOut, f.label, 0.3 * H);

  // Constant velocity => the backtrace is exact regardless of RK2 vs Euler,
  // so this isolates indexing bugs from integration bugs. Interior faces
  // only; wall faces are the boundary condition's business, not advection's.
  for (let j = 0; j < g.ny; j++) {
    for (let i = 1; i < g.nx; i++) {
      assert.ok(
        Math.abs(uOut[idxU(g, i, j)] - U) < 1e-12,
        `uOut(${i},${j}) = ${uOut[idxU(g, i, j)]}`,
      );
    }
  }
  for (let j = 1; j < g.ny; j++) {
    for (let i = 0; i < g.nx; i++) {
      assert.ok(
        Math.abs(vOut[idxV(g, i, j)] - V) < 1e-12,
        `vOut(${i},${j}) = ${vOut[idxV(g, i, j)]}`,
      );
    }
  }
});

/**
 * On a field linear in x, bilinear interpolation is exact, so RK2 has a
 * closed form. With u(x) = x/h and v = 0, backtracing from face i gives
 *   mid  = i*h - 0.5*dt*i        -> u(mid) = i*(1 - r/2)
 *   prev = i*h - dt*i*(1 - r/2)  -> u(prev) = i*(1 - r + r^2/2),  r = dt/h
 * That r^2/2 term is the whole point: forward Euler would give i*(1 - r),
 * so this test fails if the midpoint stage is dropped or mis-scaled.
 */
test('RK2 on a ramp matches the closed form (and rejects forward Euler)', () => {
  const g = createGrid(6, 4, H);
  const f = createFields(g, Float64Array);
  for (let j = 0; j < g.ny; j++) {
    for (let i = 0; i <= g.nx; i++) f.u[idxU(g, i, j)] = i; // u(x) = x/h
  }
  const dt = 0.5 * H;
  const r = dt / g.h;
  const factor = 1 - r + 0.5 * r * r;
  const { uOut, vOut } = outputs(g);

  advectVelocity(g, f.u, f.v, uOut, vOut, f.label, dt);

  for (let j = 0; j < g.ny; j++) {
    for (let i = 1; i < g.nx; i++) {
      const got = uOut[idxU(g, i, j)];
      assert.ok(
        Math.abs(got - i * factor) < 1e-12,
        `uOut(${i},${j}) = ${got}, expected ${i * factor}`,
      );
      assert.ok(Math.abs(got - i * (1 - r)) > 1e-9, `uOut(${i},${j}) looks like forward Euler`);
    }
  }
});

test('the same closed form holds for v on a ramp in y', () => {
  const g = createGrid(4, 6, H);
  const f = createFields(g, Float64Array);
  for (let j = 0; j <= g.ny; j++) {
    for (let i = 0; i < g.nx; i++) f.v[idxV(g, i, j)] = j; // v(y) = y/h
  }
  const dt = 0.5 * H;
  const r = dt / g.h;
  const factor = 1 - r + 0.5 * r * r;
  const { uOut, vOut } = outputs(g);

  advectVelocity(g, f.u, f.v, uOut, vOut, f.label, dt);

  // Exercises idxV's stride independently of idxU's — a v-loop that writes
  // through idxU passes the u test above and fails here.
  for (let j = 1; j < g.ny; j++) {
    for (let i = 0; i < g.nx; i++) {
      const got = vOut[idxV(g, i, j)];
      assert.ok(
        Math.abs(got - j * factor) < 1e-12,
        `vOut(${i},${j}) = ${got}, expected ${j * factor}`,
      );
    }
  }
});

test('normal velocity stays zero on all four walls', () => {
  const g = createGrid(6, 6, H);
  const f = createFields(g, Float64Array);
  // Nonzero inside, zero on the walls it is normal to — the u.n = 0 state
  // that projection produces and that advection must not destroy.
  for (let j = 0; j < g.ny; j++) {
    for (let i = 0; i <= g.nx; i++) f.u[idxU(g, i, j)] = Math.sin((Math.PI * i) / g.nx);
  }
  for (let j = 0; j <= g.ny; j++) {
    for (let i = 0; i < g.nx; i++) f.v[idxV(g, i, j)] = Math.sin((Math.PI * j) / g.ny);
  }
  const { uOut, vOut } = outputs(g);

  advectVelocity(g, f.u, f.v, uOut, vOut, f.label, 0.4 * H);

  for (let j = 0; j < g.ny; j++) {
    assert.ok(
      Math.abs(uOut[idxU(g, 0, j)]) < 1e-12,
      `left wall u(0,${j}) = ${uOut[idxU(g, 0, j)]}`,
    );
    assert.ok(Math.abs(uOut[idxU(g, g.nx, j)]) < 1e-12, `right wall u(${g.nx},${j})`);
  }
  for (let i = 0; i < g.nx; i++) {
    assert.ok(
      Math.abs(vOut[idxV(g, i, 0)]) < 1e-12,
      `bottom wall v(${i},0) = ${vOut[idxV(g, i, 0)]}`,
    );
    assert.ok(Math.abs(vOut[idxV(g, i, g.ny)]) < 1e-12, `top wall v(${i},${g.ny})`);
  }
});

test('a huge timestep stays bounded instead of blowing up', () => {
  const g = createGrid(8, 8, H);
  const f = createFields(g, Float64Array);
  for (let j = 0; j < g.ny; j++) {
    for (let i = 0; i <= g.nx; i++) f.u[idxU(g, i, j)] = Math.sin(i * 1.7) * Math.cos(j * 2.3);
  }
  for (let j = 0; j <= g.ny; j++) {
    for (let i = 0; i < g.nx; i++) f.v[idxV(g, i, j)] = Math.cos(i * 1.1) * Math.sin(j * 0.9);
  }
  const uMax = Math.max(...f.u.map(Math.abs));
  const vMax = Math.max(...f.v.map(Math.abs));
  const { uOut, vOut } = outputs(g);

  // 50x the CFL-stable step. This is semi-Lagrangian's whole selling point
  // over explicit advection: every output is a bilinear blend of inputs, so
  // it cannot exceed the input range no matter how far the backtrace runs.
  advectVelocity(g, f.u, f.v, uOut, vOut, f.label, 50 * H);

  for (let k = 0; k < uOut.length; k++) {
    assert.ok(
      Number.isFinite(uOut[k]) && Math.abs(uOut[k]) <= uMax + 1e-12,
      `uOut[${k}] = ${uOut[k]}`,
    );
  }
  for (let k = 0; k < vOut.length; k++) {
    assert.ok(
      Number.isFinite(vOut[k]) && Math.abs(vOut[k]) <= vMax + 1e-12,
      `vOut[${k}] = ${vOut[k]}`,
    );
  }
});

/**
 * The one test a steady field cannot give you. addRotational is an exact
 * Euler solution, so it stays put whether advection works or not; a vortex
 * dipole must physically TRAVEL, and only a correct backtrace moves it the
 * right way. Runs the full advect -> project loop, so it doubles as the
 * integration test for the per-frame sequence in main.ts.
 */
test('a vortex dipole self-propels across the box', () => {
  const N = 48;
  const h = 1 / N;
  const dt = 0.05;
  const g = createGrid(N, N, h);
  const f = createFields(g, Float64Array);
  addVortexPair(g, f.u, f.v);

  let uN: FieldArray = new Float64Array(f.u.length);
  let vN: FieldArray = new Float64Array(f.v.length);
  const div = new Float64Array(f.p.length);
  const scale = (h * h) / dt;
  const gradScale = dt / h;

  // Speed-squared-weighted centroid height: where the flow's energy sits.
  const centroidY = (): number => {
    let num = 0;
    let den = 0;
    for (let j = 0; j < g.ny; j++) {
      for (let i = 0; i < g.nx; i++) {
        const cu = 0.5 * (f.u[idxU(g, i, j)] + f.u[idxU(g, i + 1, j)]);
        const cv = 0.5 * (f.v[idxV(g, i, j)] + f.v[idxV(g, i, j + 1)]);
        const e = cu * cu + cv * cv;
        num += e * (j + 0.5) * h;
        den += e;
      }
    }
    return num / den;
  };

  const startY = centroidY();
  for (let n = 0; n < 150; n++) {
    advectVelocity(g, f.u, f.v, uN, vN, f.label, dt);
    [f.u, uN] = [uN, f.u];
    [f.v, vN] = [vN, f.v];
    computeDivergence(g, f.u, f.v, div);
    solvePressure(g, f.p, div, f.label, scale, 40, 1.8);
    subtractGradient(g, f.p, f.u, f.v, f.label, gradScale);
  }
  const endY = centroidY();

  assert.ok(Number.isFinite(endY), `centroid went non-finite: ${endY}`);
  // Seeded at y=0.30 with the +blob left of the -blob, which induces upward
  // motion. A kernel that does nothing leaves this unchanged; one that
  // backtraces with the wrong sign sends it down.
  assert.ok(
    endY - startY > 0.04,
    `dipole rose only ${(endY - startY).toFixed(4)} (from ${startY.toFixed(4)} to ${endY.toFixed(4)}); expected clear upward travel`,
  );
});

/**
 * A constant velocity makes the backtrace exact, so with dt chosen to move the
 * fluid a WHOLE cell the answer is a pure index shift with no interpolation at
 * all. That turns advectScalar into an equality test rather than a tolerance
 * test, and it is the check that pins the half-cell offset: sampling a cell
 * center half a cell off replaces the shift with a 50/50 blur of two columns.
 */
test('advectScalar translates a stripe by exactly one cell', () => {
  const g = createGrid(8, 8, H);
  const f = createFields(g, Float64Array);
  const U = 0.75;
  f.u.fill(U);
  f.v.fill(0); // no y motion, so the stripe must stay a stripe
  for (let j = 0; j < g.ny; j++) f.dye[0][idxP(g, 3, j)] = 1;
  const out = new Float64Array(g.nx * g.ny).fill(NaN);

  advectScalar(g, f.u, f.v, f.dye[0], out, f.label, H / U);

  for (let j = 0; j < g.ny; j++) {
    for (let i = 0; i < g.nx; i++) {
      const want = i === 4 ? 1 : 0;
      assert.ok(
        Math.abs(out[idxP(g, i, j)] - want) < 1e-12,
        `cell (${i},${j}) = ${out[idxP(g, i, j)]}, want ${want}`,
      );
    }
  }
});

test('advectScalar carries dye along +y when only v is set', () => {
  const g = createGrid(8, 8, H);
  const f = createFields(g, Float64Array);
  const V = 0.5;
  f.v.fill(V);
  for (let i = 0; i < g.nx; i++) f.dye[0][idxP(g, i, 3)] = 1;
  const out = new Float64Array(g.nx * g.ny).fill(NaN);

  advectScalar(g, f.u, f.v, f.dye[0], out, f.label, H / V);

  // The mirror of the test above, exercising the y half of the backtrace.
  // Neither of these can see a bug in the MIDPOINT stage, though: under a
  // constant velocity the midpoint sample equals the endpoint sample, so RK2
  // and forward Euler agree. The ramp tests below are what cover that.
  for (let j = 0; j < g.ny; j++) {
    for (let i = 0; i < g.nx; i++) {
      const want = j === 4 ? 1 : 0;
      assert.ok(
        Math.abs(out[idxP(g, i, j)] - want) < 1e-12,
        `cell (${i},${j}) = ${out[idxP(g, i, j)]}, want ${want}`,
      );
    }
  }
});

test('advectScalar invents no new extrema, even at an absurd dt', () => {
  const g = createGrid(16, 16, H);
  const f = createFields(g, Float64Array);
  addVortexPair(g, f.u, f.v);
  for (let j = 0; j < g.ny; j++) {
    for (let i = 0; i < g.nx; i++) f.dye[0][idxP(g, i, j)] = i < 8 ? 1 : 0;
  }
  const out = new Float64Array(g.nx * g.ny).fill(NaN);

  // Bilinear blending bounds the output by the input range, which is what
  // makes dye safe to render on a FIXED [0,1] ramp: it can fade but never
  // clip. A scheme that overshoots here (MacCormack unclamped, say) would
  // need that guarantee restored by explicit clamping.
  advectScalar(g, f.u, f.v, f.dye[0], out, f.label, 50 * H);

  for (let k = 0; k < out.length; k++) {
    assert.ok(Number.isFinite(out[k]), `cell ${k} never written`);
    assert.ok(out[k] >= -1e-12 && out[k] <= 1 + 1e-12, `cell ${k} = ${out[k]} escaped [0,1]`);
  }
});

test('advectScalar leaves a uniform dye field untouched', () => {
  const g = createGrid(16, 16, H);
  const f = createFields(g, Float64Array);
  addVortexPair(g, f.u, f.v);
  f.dye[0].fill(0.7);
  const out = new Float64Array(g.nx * g.ny).fill(NaN);

  advectScalar(g, f.u, f.v, f.dye[0], out, f.label, 0.3 * H);

  // Any backtrace whatsoever lands in a region of constant 0.7, so this is
  // independent of the velocity being right — it isolates the sampler's edge
  // clamping. An out-of-range read shows up as NaN, not as a small error.
  for (let k = 0; k < out.length; k++) {
    assert.ok(Math.abs(out[k] - 0.7) < 1e-12, `cell ${k} = ${out[k]}`);
  }
});

/**
 * The scalar counterpart of the RK2 ramp tests above, and the only kind that
 * can see a bug in the midpoint stage: under a CONSTANT velocity the midpoint
 * sample equals the endpoint sample, so every exact-translation test passes
 * whichever velocity component the midpoint reads.
 *
 * Both the dye and the velocity are linear, so bilinear interpolation is exact
 * and the whole backtrace has a closed form. With u(x) = x/h, v = 0 and dye
 * q(x) = x/h, backtracing from the center of column i (x = (i+0.5)h) gives
 *   mid  = x(1 - r/2)
 *   prev = x(1 - r + r^2/2),   r = dt/h
 * and since q just reports the x it was sampled at, the answer is that same
 * factor times (i+0.5). Forward Euler would give (i+0.5)(1 - r) instead.
 */
test('advectScalar RK2 on a velocity ramp matches the closed form', () => {
  const g = createGrid(8, 4, H);
  const f = createFields(g, Float64Array);
  for (let j = 0; j < g.ny; j++) {
    for (let i = 0; i <= g.nx; i++) f.u[idxU(g, i, j)] = i; // u(x) = x/h
  }
  for (let j = 0; j < g.ny; j++) {
    for (let i = 0; i < g.nx; i++) f.dye[0][idxP(g, i, j)] = i + 0.5; // q(x) = x/h
  }
  const dt = 0.5 * H;
  const r = dt / g.h;
  const factor = 1 - r + 0.5 * r * r;
  const out = new Float64Array(g.nx * g.ny).fill(NaN);

  advectScalar(g, f.u, f.v, f.dye[0], out, f.label, dt);

  // From i = 1: column 0 backtraces past the first cell center, where sampleP
  // clamps and the closed form stops applying.
  for (let j = 0; j < g.ny; j++) {
    for (let i = 1; i < g.nx; i++) {
      const got = out[idxP(g, i, j)];
      const want = (i + 0.5) * factor;
      assert.ok(Math.abs(got - want) < 1e-12, `cell (${i},${j}) = ${got}, expected ${want}`);
      assert.ok(
        Math.abs(got - (i + 0.5) * (1 - r)) > 1e-9,
        `cell (${i},${j}) looks like forward Euler`,
      );
    }
  }
});

test('advectScalar RK2 holds on a ramp in y too', () => {
  const g = createGrid(4, 8, H);
  const f = createFields(g, Float64Array);
  for (let j = 0; j <= g.ny; j++) {
    for (let i = 0; i < g.nx; i++) f.v[idxV(g, i, j)] = j; // v(y) = y/h
  }
  for (let j = 0; j < g.ny; j++) {
    for (let i = 0; i < g.nx; i++) f.dye[0][idxP(g, i, j)] = j + 0.5; // q(y) = y/h
  }
  const dt = 0.5 * H;
  const r = dt / g.h;
  const factor = 1 - r + 0.5 * r * r;
  const out = new Float64Array(g.nx * g.ny).fill(NaN);

  advectScalar(g, f.u, f.v, f.dye[0], out, f.label, dt);

  // Run as a pair, these two pin each midpoint stage to its OWN velocity
  // component: reading v for midX passes the x test (where v is 0, so the
  // midpoint collapses back to Euler and the r^2/2 term vanishes) and fails
  // here, and the reverse for reading u in midY.
  for (let j = 1; j < g.ny; j++) {
    for (let i = 0; i < g.nx; i++) {
      const got = out[idxP(g, i, j)];
      const want = (j + 0.5) * factor;
      assert.ok(Math.abs(got - want) < 1e-12, `cell (${i},${j}) = ${got}, expected ${want}`);
      assert.ok(
        Math.abs(got - (j + 0.5) * (1 - r)) > 1e-9,
        `cell (${i},${j}) looks like forward Euler`,
      );
    }
  }
});

// --- MacCormack ------------------------------------------------------------

test('MacCormack leaves a uniform velocity field alone', () => {
  const g = createGrid(6, 6, H);
  const f = createFields(g, Float64Array);
  const U = 0.75;
  const V = -0.25;
  f.u.fill(U);
  f.v.fill(V);
  const hat = outputs(g);
  const { uOut, vOut } = outputs(g);

  advectVelocityMacCormack(g, f.u, f.v, hat.uOut, hat.vOut, uOut, vOut, f.label, 0.3 * H);

  // The round trip is exact on a constant field, so the correction term is
  // identically zero — anything else means the reverse pass has the wrong
  // sign or is carried by the wrong velocity.
  for (let j = 0; j < g.ny; j++) {
    for (let i = 1; i < g.nx; i++) {
      assert.ok(Math.abs(uOut[idxU(g, i, j)] - U) < 1e-12, `uOut(${i},${j})`);
    }
  }
  for (let j = 1; j < g.ny; j++) {
    for (let i = 0; i < g.nx; i++) {
      assert.ok(Math.abs(vOut[idxV(g, i, j)] - V) < 1e-12, `vOut(${i},${j})`);
    }
  }
});

test('the limiter keeps MacCormack inside the input range at a huge dt', () => {
  const g = createGrid(8, 8, H);
  const f = createFields(g, Float64Array);
  for (let j = 0; j < g.ny; j++) {
    for (let i = 0; i <= g.nx; i++) f.u[idxU(g, i, j)] = Math.sin(i * 1.7) * Math.cos(j * 2.3);
  }
  for (let j = 0; j <= g.ny; j++) {
    for (let i = 0; i < g.nx; i++) f.v[idxV(g, i, j)] = Math.cos(i * 1.1) * Math.sin(j * 0.9);
  }
  const uMax = Math.max(...f.u.map(Math.abs));
  const vMax = Math.max(...f.v.map(Math.abs));
  const hat = outputs(g);
  const { uOut, vOut } = outputs(g);

  // Plain semi-Lagrangian gets this for free (its output IS a blend of
  // inputs). MacCormack's correction is what could overshoot, so this test
  // fails the moment clampToStencil* is dropped.
  advectVelocityMacCormack(g, f.u, f.v, hat.uOut, hat.vOut, uOut, vOut, f.label, 50 * H);

  for (let k = 0; k < uOut.length; k++) {
    assert.ok(Number.isFinite(uOut[k]) && Math.abs(uOut[k]) <= uMax + 1e-12, `uOut[${k}]`);
  }
  for (let k = 0; k < vOut.length; k++) {
    assert.ok(Number.isFinite(vOut[k]) && Math.abs(vOut[k]) <= vMax + 1e-12, `vOut[${k}]`);
  }
});

/**
 * The claim the whole scheme exists for, as a number. A Gaussian blob carried
 * by a uniform velocity should translate with its shape intact; every step of
 * bilinear interpolation shaves the peak instead. Same field, same steps, same
 * dt — only the scheme differs.
 */
test('MacCormack keeps a scalar peak that semi-Lagrangian smears away', () => {
  const g = createGrid(64, 16, H);
  const f = createFields(g, Float64Array);
  f.u.fill(1.0);
  const sigma = 4 * H;
  const seed = (q: FieldArray): void => {
    for (let j = 0; j < g.ny; j++) {
      for (let i = 0; i < g.nx; i++) {
        const dx = (i + 0.5) * H - 8 * H;
        const dy = (j + 0.5) * H - 8 * H;
        q[idxP(g, i, j)] = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
      }
    }
  };

  const dt = 0.5 * H;
  const run = (mac: boolean): number => {
    let a: FieldArray = new Float64Array(g.nx * g.ny);
    let b: FieldArray = new Float64Array(g.nx * g.ny);
    const hat = new Float64Array(g.nx * g.ny);
    seed(a);
    for (let n = 0; n < 40; n++) {
      if (mac) advectScalarMacCormack(g, f.u, f.v, a, hat, b, f.label, dt);
      else advectScalar(g, f.u, f.v, a, b, f.label, dt);
      [a, b] = [b, a];
    }
    return Math.max(...a);
  };

  const sl = run(false);
  const mc = run(true);
  assert.ok(mc > sl + 0.05, `MacCormack peak ${mc} not clearly above semi-Lagrangian ${sl}`);
  assert.ok(mc <= 1 + 1e-12, `MacCormack overshot the initial maximum: ${mc}`);
});
