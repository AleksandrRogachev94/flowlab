/**
 * Test fields for verifying projection: u* = curl(psi) + grad(phi).
 * Projection must kill the gradient part and leave the curl part alone.
 *
 * Both are built from DISCRETE differences of a potential rather than
 * sampled analytic derivatives, so div(curl(psi)) cancels exactly (1e-16)
 * instead of to O(h^2). Tolerances stay tight enough to blame the solver.
 *
 * The domain is one unit TALL and W = nx*h units wide (Simulation sets
 * h = 1/ny). Everything below is written against W rather than against 1, so a
 * seed places itself the same way in a square box and in a wide one; on a
 * square grid W = 1 and every formula reduces to what it was. Both potentials
 * give zero normal velocity at the walls, so total divergence sums to 0 as the
 * pressure solve requires. Frequencies differ on purpose: same frequency
 * makes curl and grad exact negatives that cancel to nothing.
 *
 * Every function here ADDs into its output — zero the arrays first for a fresh
 * field. That is what lets a scene be composed from several seeds.
 *
 * addDyeDisk is the odd one out: it seeds a passive tracer, not velocity, and
 * so verifies nothing on its own. It lives here because it is scene setup and
 * shares the same world-coordinate conventions.
 */

import { idxP, idxU, idxV, type FieldArray, type Grid } from '../core/grid.ts';
import type { DyeSource } from '../core/simulation.ts';

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

/** Domain width in world units; the height is always 1. */
const widthOf = (g: Grid): number => g.nx * g.h;

/** Divergence-free part: u = curl(psi), one half-period of sine per axis, so
 *  psi vanishes on all four walls whatever the aspect ratio. */
export function addRotational(g: Grid, u: FieldArray, v: FieldArray, amp = 1): void {
  const w = widthOf(g);
  addCurlOfStream(g, u, v, (i, j) => Math.sin((PI * i * g.h) / w) * Math.sin(PI * j * g.h), amp);
}

/**
 * A disk of dye at concentration 1, centred in the unit square by default.
 * Passive tracer: it shows where the fluid GOES, which no view of the velocity
 * field does directly, and its edge is the readout for advection's dissipation.
 *
 * Radius and centre are in WORLD units, like every other seed here, rather
 * than in cells — a cell-quantized radius changes the shape as N changes,
 * which makes two resolutions non-comparable.
 *
 * The edge ramps over one cell instead of stepping 1 -> 0. A hard step is
 * jagged at the grid scale, and its corners are precisely the sub-cell
 * features semi-Lagrangian advection erases in the first few steps — so the
 * scheme gets blamed for smoothing what was aliasing in the seed. One cell of
 * ramp is narrow enough that the blur you then watch grow is genuinely the
 * scheme's.
 *
 * ADDs, like the velocity seeds: overlapping disks sum past 1.
 */
export function addDyeDisk(
  g: Grid,
  dye: FieldArray,
  r = 0.15,
  cx = 0.5 * widthOf(g),
  cy = 0.5,
): void {
  for (let j = 0; j < g.ny; j++) {
    const y = (j + 0.5) * g.h;
    for (let i = 0; i < g.nx; i++) {
      const x = (i + 0.5) * g.h;
      // t crosses 0.5 exactly at the radius, so the ramp straddles it.
      const t = Math.min(Math.max((r - Math.hypot(x - cx, y - cy)) / g.h + 0.5, 0), 1);
      dye[idxP(g, i, j)] += t * t * (3 - 2 * t); // smoothstep
    }
  }
}

/**
 * One centred disk, written identically to every channel. Identical data
 * carried by identical velocity stays identical, so this renders as pure
 * greyscale: the classic single-dye picture, and the control for judging what
 * the triad's colour mixing is actually worth.
 *
 * It does spend three advections carrying one scalar's worth of information.
 * That is the deliberate trade — the step loop and the RGB compositor stay
 * branch-free, and dye advection is negligible beside the pressure solve.
 */
export function addDyeMono(g: Grid, dye: FieldArray[], r = 0.15): void {
  for (const c of dye) addDyeDisk(g, c, r);
}

/**
 * Horizontal bands across the whole domain — the strain map.
 *
 * Where addDyeTriad's three fat blobs answer "where did this fluid come from",
 * many thin bands answer "how much has it been STRETCHED": every band edge is a
 * material line, so the local spacing IS the strain field. Bands crowd where
 * the flow accelerates and spread where it slows, since the flux between two
 * material lines is conserved.
 *
 * The period is in CELLS because its useful lower bound is a resolution limit,
 * not a physical length: below ~8 cells the strain thins bands past the grid
 * within a few diameters and they alias into moire. Worth having only since
 * MacCormack advection landed — semi-Lagrangian blurred a 12-cell period to
 * flat grey almost immediately, which is why this was not a useful tracer
 * before.
 */
export function addDyeStripes(g: Grid, dye: FieldArray[], periodCells = 12): void {
  for (let j = 0; j < g.ny; j++) {
    for (let c = 0; c < dye.length; c++) {
      const v = stripeAt(g, j, periodCells, c, dye.length);
      for (let i = 0; i < g.nx; i++) dye[c][idxP(g, i, j)] = v;
    }
  }
}

/**
 * One channel's value at row j: a bright band times a hue.
 *
 * Two independent periods, and keeping them apart is the whole design.
 *
 *   BRIGHTNESS cycles every `periodCells` — many thin bands, squared so they
 *   are narrow bright lines with dark gaps rather than a continuous wash. The
 *   gaps are what let the eye follow ONE line through a vortex.
 *
 *   HUE cycles ONCE over the domain height, via the three channels offset by a
 *   third of a cycle: red at the bottom, through green, to blue at the top.
 *   Composited as RGB that is an ordinary cyclic colour ramp, built in the DATA
 *   rather than in the renderer — which is how three scalars show far more than
 *   three bands.
 *
 * Cycling hue per BAND instead (the obvious first try) looks striking and
 * analyses nothing: every band is then the same rainbow, so colour reports
 * position within a period and cannot say which band you are looking at. Tied
 * to the domain instead, colour is a label a band keeps no matter how far the
 * wake folds it — which is exactly the question a strain map has to answer.
 *
 * Sub-grid mixing still reads as desaturation, as in addDyeTriad: once bands
 * interleave below the cell scale the channels average to their common mean.
 */
function stripeAt(g: Grid, j: number, periodCells: number, c: number, channels: number): number {
  const band = 0.5 + 0.5 * Math.cos((2 * Math.PI * j) / periodCells);
  const hue = 0.5 + 0.5 * Math.cos(2 * Math.PI * (j / g.ny - c / channels));
  return band * band * hue;
}

/**
 * addDyeStripes as a SOURCE — the same pattern re-stamped in the inlet columns
 * every step, so bands keep arriving instead of washing out once.
 *
 * A seeded tracer is a transient in any scene with an outlet: every dyed
 * particle leaves within one domain transit and nothing replaces it. That is
 * fine in a closed box, where the fluid recirculates forever, and useless in a
 * channel — which is exactly the scene where a strain map is most worth having.
 *
 * Assumes the WHOLE left edge is inflow, so it belongs to karman and not to a
 * scene like wallJet where part of that edge is wall: stamping a wall cell
 * pins dye against it with no flow to carry it away. main.ts gates this on
 * SceneDef.inflow rather than guessing.
 */
export function stripeInflow(periodCells = 12, depthCells = 2): DyeSource {
  return (g, dye) => {
    const depth = Math.min(depthCells, g.nx);
    for (let j = 0; j < g.ny; j++) {
      for (let c = 0; c < dye.length; c++) {
        const v = stripeAt(g, j, periodCells, c, dye.length);
        for (let i = 0; i < depth; i++) dye[c][idxP(g, i, j)] = v;
      }
    }
  };
}

/**
 * One disk per dye channel, on a triangle about the centre and sized to
 * OVERLAP their neighbours. Displayed as RGB, the three pairwise overlaps read
 * as yellow, cyan and magenta from the first frame.
 *
 * The overlaps are the instrument. A single channel only ever shows an edge
 * softening, which is hard to judge; interleaved channels mix to a colour that
 * was never seeded, so filaments thinner than a cell announce themselves as
 * the picture desaturating toward grey. That is numerical diffusion made
 * legible — and the thing to watch when a higher-order scheme lands.
 */
export function addDyeTriad(g: Grid, dye: FieldArray[], r = 0.13, d = 0.12): void {
  const cx = 0.5 * widthOf(g);
  for (let c = 0; c < dye.length; c++) {
    // Start at 90 deg so the red disk sits on top, then step by a full turn
    // divided between the channels.
    const a = PI / 2 + (2 * PI * c) / dye.length;
    addDyeDisk(g, dye[c], r, cx + d * Math.cos(a), 0.5 + d * Math.sin(a));
  }
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
  const cx = 0.5 * widthOf(g);
  const blob = (x: number, y: number, bx: number, by: number) =>
    Math.exp(-((x - bx) ** 2 + (y - by) ** 2) / (sigma * sigma));

  addCurlOfStream(
    g,
    u,
    v,
    (i, j) => {
      const x = i * g.h;
      const y = j * g.h;
      return k * (blob(x, y, cx - 0.15, 0.3) - blob(x, y, cx + 0.15, 0.3));
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
  const spanX = widthOf(g) - 2 * margin;
  const spanY = 1 - 2 * margin;
  const rand = lcg(seed);

  const cx = new Float64Array(count);
  const cy = new Float64Array(count);
  const sign = new Float64Array(count);
  for (let n = 0; n < count; n++) {
    cx[n] = margin + rand() * spanX;
    cy[n] = margin + rand() * spanY;
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
  const w = widthOf(g);
  const phi = (i: number, j: number) =>
    Math.cos((2 * PI * (i + 0.5) * g.h) / w) * Math.cos(2 * PI * (j + 0.5) * g.h);

  for (let j = 0; j < g.ny; j++)
    for (let i = 1; i < g.nx; i++) u[idxU(g, i, j)] += inv * (phi(i, j) - phi(i - 1, j));

  for (let j = 1; j < g.ny; j++)
    for (let i = 0; i < g.nx; i++) v[idxV(g, i, j)] += inv * (phi(i, j) - phi(i, j - 1));
}
