/**
 * Test fields for verifying projection: u* = curl(psi) + grad(phi).
 * Projection must kill the gradient part and leave the curl part alone.
 *
 * Both are built from DISCRETE differences of a potential rather than
 * sampled analytic derivatives, so div(curl(psi)) cancels exactly (1e-16)
 * instead of to O(h^2). Tolerances stay tight enough to blame the solver.
 *
 * Assumes the unit square (h = 1/nx, nx === ny). Both potentials give zero
 * normal velocity at the walls, so total divergence sums to 0 as the
 * pressure solve requires. Frequencies differ on purpose: same frequency
 * makes curl and grad exact negatives that cancel to nothing.
 *
 * Both functions ADD into u/v — zero the arrays first for a fresh field.
 */

import { idxU, idxV, type FieldArray, type Grid } from '../core/grid.ts';

const PI = Math.PI;

/**
 * Discrete curl of a stream function: u = dpsi/dy, v = -dpsi/dx, on exactly
 * the stencil computeDivergence inverts. Building both components from ONE
 * potential buys two things: div(u) cancels to machine precision rather than
 * O(h^2), and total divergence telescopes to exactly 0 for ANY psi, so the
 * all-Neumann pressure solve is always compatible.
 *
 * `psi` is sampled at cell CORNERS: psi(i, j) sits at (i*h, j*h).
 */
export function addCurlOfStream(
  g: Grid,
  u: FieldArray,
  v: FieldArray,
  psi: (i: number, j: number) => number,
  amp: number,
): void {
  const inv = amp / g.h;

  for (let j = 0; j < g.ny; j++)
    for (let i = 0; i <= g.nx; i++) u[idxU(g, i, j)] += inv * (psi(i, j + 1) - psi(i, j));

  for (let j = 0; j <= g.ny; j++)
    for (let i = 0; i < g.nx; i++) v[idxV(g, i, j)] += -inv * (psi(i + 1, j) - psi(i, j));
}

/** Divergence-free part: u = curl(psi), psi = sin(pi x) sin(pi y) at corners. */
export function addRotational(g: Grid, u: FieldArray, v: FieldArray, amp = 1): void {
  addCurlOfStream(g, u, v, (i, j) => Math.sin(PI * i * g.h) * Math.sin(PI * j * g.h), amp);
}

/**
 * Counter-rotating vortex pair: two opposite-signed Gaussian blobs side by
 * side. Each sits in the other's flow, so the PAIR self-propels — upward as
 * placed — and splits apart against the far wall.
 *
 * That motion is the point. addRotational is an exact steady Euler solution,
 * so a correct advection kernel and a broken one both render a still picture;
 * it tests stability, not correctness. A dipole that refuses to move, drifts
 * sideways, or crawls is visibly wrong.
 *
 * `amp` reads as peak speed: |grad| of a unit Gaussian peaks at
 * sqrt(2)/sigma * exp(-1/2), which the prefactor divides out. Default sigma
 * is ~5 cells at N=64 — wide enough to survive smearing, narrow enough that
 * the blobs reach ~1e-6 at the walls.
 */
export function addVortexPair(g: Grid, u: FieldArray, v: FieldArray, amp = 1, sigma = 0.08): void {
  const k = sigma / (Math.SQRT2 * Math.exp(-0.5));
  const blob = (x: number, y: number, cx: number, cy: number) =>
    Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (sigma * sigma));

  addCurlOfStream(
    g,
    u,
    v,
    (i, j) => {
      const x = i * g.h;
      const y = j * g.h;
      return k * (blob(x, y, 0.35, 0.3) - blob(x, y, 0.65, 0.3));
    },
    amp,
  );
}

/**
 * Deterministic LCG. Method comparison is only meaningful against an
 * IDENTICAL initial condition, so this must never use Math.random().
 */
function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/**
 * Alternating-sign Gaussian vortices — the benchmark scene for comparing
 * advection schemes. Neighbouring vortices orbit, stretch each other into
 * filaments a few cells across, and merge; that is exactly the scale a
 * dissipative scheme erases first, so bilinear vs cubic, or confinement on
 * vs off, differ visibly rather than only numerically.
 *
 * Not the textbook double shear layer, which needs PERIODIC sides — in a
 * closed box that becomes a wall-interaction test instead. Compact blobs
 * decay to nothing near the walls, keeping the boundaries out of it.
 *
 * Signs alternate so net circulation is ~0. `amp` is per-blob peak speed;
 * overlapping blobs sum above it.
 */
export function addVortexCluster(
  g: Grid,
  u: FieldArray,
  v: FieldArray,
  amp = 1,
  count = 14,
  sigma = 0.05,
  seed = 12345,
): void {
  const k = sigma / (Math.SQRT2 * Math.exp(-0.5));
  // 3.5 sigma of margin puts the blobs' wall value near 2e-6 — small enough
  // that wall-normal velocity stays negligible. See addVortexPair.
  const margin = 3.5 * sigma;
  const span = 1 - 2 * margin;
  const rand = lcg(seed);

  const cx = new Float64Array(count);
  const cy = new Float64Array(count);
  const sign = new Float64Array(count);
  for (let n = 0; n < count; n++) {
    cx[n] = margin + rand() * span;
    cy[n] = margin + rand() * span;
    sign[n] = n % 2 === 0 ? 1 : -1;
  }

  addCurlOfStream(
    g,
    u,
    v,
    (i, j) => {
      const x = i * g.h;
      const y = j * g.h;
      let psi = 0;
      for (let n = 0; n < count; n++) {
        const dx = x - cx[n];
        const dy = y - cy[n];
        psi += sign[n] * Math.exp(-(dx * dx + dy * dy) / (sigma * sigma));
      }
      return k * psi;
    },
    amp,
  );
}

/**
 * Pure gradient part: u = grad(phi), phi = cos(2pi x) cos(2pi y) at centers.
 * Boundary faces are skipped, which is the Neumann condition dphi/dn = 0.
 */
export function addGradient(g: Grid, u: FieldArray, v: FieldArray, amp = 1): void {
  const inv = amp / g.h;
  const phi = (i: number, j: number) =>
    Math.cos(2 * PI * (i + 0.5) * g.h) * Math.cos(2 * PI * (j + 0.5) * g.h);

  for (let j = 0; j < g.ny; j++)
    for (let i = 1; i < g.nx; i++) u[idxU(g, i, j)] += inv * (phi(i, j) - phi(i - 1, j));

  for (let j = 1; j < g.ny; j++)
    for (let i = 0; i < g.nx; i++) v[idxV(g, i, j)] += inv * (phi(i, j) - phi(i, j - 1));
}
