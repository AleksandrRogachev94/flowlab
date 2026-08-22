import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createGrid, createFields, idxU, idxV, type FieldArray } from './grid.ts';
import { advectVelocity } from './advect.ts';
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
