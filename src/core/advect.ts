import {
  clampToStencilP,
  clampToStencilU,
  clampToStencilV,
  idxP,
  idxU,
  idxV,
  isSolidOrOutside,
  sampleP,
  sampleU,
  sampleV,
  type FieldArray,
  type Grid,
} from './grid.ts';

/**
 * Which advection scheme `Simulation` runs. Both are semi-Lagrangian; the
 * second wraps the first in an error-correction pass. See advectVelocity and
 * advectVelocityMacCormack.
 */
export type AdvectionScheme = 'semiLagrangian' | 'macCormack';

/**
 * Where the fluid now at (x, y) came from, one dt ago, by RK2 (midpoint):
 *   mid  = x - (dt/2) * vel(x)
 *   prev = x - dt * vel(mid)
 * `u0`/`v0` are the carrier velocity AT (x, y), passed in because the caller
 * often knows one of them exactly: on a u face the field stores u there, on a
 * v face it stores v. A cell centre stores neither and samples both.
 *
 * dt < 0 traces FORWARD in time instead — that is the only difference between
 * the two halves of a MacCormack step.
 *
 * Module-level result object: allocated once (Rule 3), never in the loop, and
 * safe to share because this is single-threaded and it is read back
 * immediately. Keeping the RK2 in one place is worth more than the purity —
 * MacCormack would otherwise repeat it in six spots.
 */
const back = { x: 0, y: 0 };

function backtrace(
  g: Grid,
  uCar: FieldArray,
  vCar: FieldArray,
  x: number,
  y: number,
  u0: number,
  v0: number,
  dt: number,
): void {
  const midX = x - 0.5 * dt * u0;
  const midY = y - 0.5 * dt * v0;
  back.x = x - dt * sampleU(g, uCar, midX, midY);
  back.y = y - dt * sampleV(g, vCar, midX, midY);
}

/**
 * The face-advection workhorse: backtrace every face along the CARRIER
 * velocity (uCar, vCar) and sample the SOURCE field (uSrc, vSrc) there.
 *
 * The two are the same arrays for ordinary self-advection (advectVelocity).
 * They differ in MacCormack's reverse pass, which pushes the already-advected
 * field back along the original velocity — the carrier is always u^n, only
 * what rides on it changes.
 *
 * @param label a face touching a solid keeps its prescribed velocity, COPIED
 *              through rather than skipped — uOut is a separate buffer and
 *              would otherwise keep stale ping-pong data. Bounds match
 *              subtractGradient's, so a face advection writes is always one
 *              projection can correct.
 */
function advectFaces(
  g: Grid,
  uCar: FieldArray,
  vCar: FieldArray,
  uSrc: FieldArray,
  vSrc: FieldArray,
  uOut: FieldArray,
  vOut: FieldArray,
  label: Uint8Array,
  dt: number,
): void {
  const h = g.h;

  for (let j = 0; j < g.ny; j++) {
    const y = (j + 0.5) * h;
    for (let i = 0; i <= g.nx; i++) {
      const k = idxU(g, i, j);
      if (isSolidOrOutside(g, label, i - 1, j) || isSolidOrOutside(g, label, i, j)) {
        uOut[k] = uSrc[k];
        continue;
      }
      const x = i * h;
      backtrace(g, uCar, vCar, x, y, uCar[k], sampleV(g, vCar, x, y), dt);
      uOut[k] = sampleU(g, uSrc, back.x, back.y);
    }
  }

  for (let j = 0; j <= g.ny; j++) {
    const y = j * h;
    for (let i = 0; i < g.nx; i++) {
      const k = idxV(g, i, j);
      if (isSolidOrOutside(g, label, i, j - 1) || isSolidOrOutside(g, label, i, j)) {
        vOut[k] = vSrc[k];
        continue;
      }
      const x = (i + 0.5) * h;
      backtrace(g, uCar, vCar, x, y, sampleU(g, uCar, x, y), vCar[k], dt);
      vOut[k] = sampleV(g, vSrc, back.x, back.y);
    }
  }
}

/**
 * Semi-Lagrangian advection of velocity: u^{n+1}(x) = u^n(x - dt * u).
 * Backtrace each face to where its fluid came from and interpolate there.
 * Unconditionally stable — every output is a bilinear blend of inputs, so it
 * cannot exceed the input range at any dt. The price is dissipation.
 *
 * RK2 (midpoint), not forward Euler — see backtrace(). Both components are
 * needed at each stage even when advecting u alone, since the backtrace moves
 * diagonally; only the one living at the face is exact.
 *
 * NOT in-place (Rule 2): u advects itself, so uIn must survive the pass.
 */
export function advectVelocity(
  g: Grid,
  uIn: FieldArray,
  vIn: FieldArray,
  uOut: FieldArray,
  vOut: FieldArray,
  label: Uint8Array,
  dt: number,
): void {
  advectFaces(g, uIn, vIn, uIn, vIn, uOut, vOut, label, dt);
}

/**
 * MacCormack (a.k.a. BFECC-style) velocity advection — second-order accurate
 * in space, at three backtraces instead of one.
 *
 * The idea: bilinear interpolation is a low-pass filter, so one semi-Lagrangian
 * pass returns a SMOOTHED field. Run the same pass backwards and the smoothing
 * happens a second time; if advection were exact, the round trip would land
 * back on u^n exactly, so whatever it misses by is (twice) the error of one
 * pass. Subtract half of it:
 *
 *   uHat = A(u^n)                     forward, dt
 *   uBar = A_rev(uHat)                backward, -dt, SAME carrier u^n
 *   u^{n+1} = uHat + (u^n - uBar) / 2
 *
 * The leading (dissipative, second-derivative) error term is identical in both
 * passes and cancels; what survives is a dispersive third-derivative term.
 * That is the trade — much less smearing, some ringing.
 *
 * The correction is unbounded on its own, so each result is clamped to the
 * range of the four values the FORWARD pass interpolated (Selle et al. 2008).
 * Without that clamp the scheme is not unconditionally stable and sharp
 * fronts grow new extrema.
 *
 * @param uHat scratch, size of uIn; holds the forward pass.
 * @param uOut receives the reverse pass first, then the corrected result
 *             in place — each face reads and writes only its own index, so
 *             this stays a legal one-thread-per-cell kernel (Rule 2).
 */
export function advectVelocityMacCormack(
  g: Grid,
  uIn: FieldArray,
  vIn: FieldArray,
  uHat: FieldArray,
  vHat: FieldArray,
  uOut: FieldArray,
  vOut: FieldArray,
  label: Uint8Array,
  dt: number,
): void {
  advectFaces(g, uIn, vIn, uIn, vIn, uHat, vHat, label, dt);
  advectFaces(g, uIn, vIn, uHat, vHat, uOut, vOut, label, -dt);

  const h = g.h;

  for (let j = 0; j < g.ny; j++) {
    const y = (j + 0.5) * h;
    for (let i = 0; i <= g.nx; i++) {
      const k = idxU(g, i, j);
      if (isSolidOrOutside(g, label, i - 1, j) || isSolidOrOutside(g, label, i, j)) {
        uOut[k] = uIn[k];
        continue;
      }
      const x = i * h;
      const corrected = uHat[k] + 0.5 * (uIn[k] - uOut[k]);
      backtrace(g, uIn, vIn, x, y, uIn[k], sampleV(g, vIn, x, y), dt);
      uOut[k] = clampToStencilU(g, uIn, back.x, back.y, corrected);
    }
  }

  for (let j = 0; j <= g.ny; j++) {
    const y = j * h;
    for (let i = 0; i < g.nx; i++) {
      const k = idxV(g, i, j);
      if (isSolidOrOutside(g, label, i, j - 1) || isSolidOrOutside(g, label, i, j)) {
        vOut[k] = vIn[k];
        continue;
      }
      const x = (i + 0.5) * h;
      const corrected = vHat[k] + 0.5 * (vIn[k] - vOut[k]);
      backtrace(g, uIn, vIn, x, y, sampleU(g, uIn, x, y), vIn[k], dt);
      vOut[k] = clampToStencilV(g, vIn, back.x, back.y, corrected);
    }
  }
}

/**
 * Semi-Lagrangian advection of a PASSIVE cell-centered scalar — dye here,
 * later smoke or temperature: q^{n+1}(x) = q^n(x - dt * u). Same RK2 backtrace
 * as advectVelocity above, but the carried field is not the carrying field, so
 * u/v are read-only and any number of scalars can ride one velocity.
 *
 * Pass the POST-projection velocity: a divergence-free carrier neither
 * concentrates nor thins the tracer, while u* invents sinks and sources
 * wherever it compresses.
 *
 * NOT in-place (Rule 2). Solid cells copy through as in advectVelocity, since
 * qOut is a separate ping-pong buffer and skipping them leaves stale data.
 *
 * @param dg the DYE grid — the one q lives on, which need not be the velocity
 *           grid `g`. Everything here is already in WORLD coordinates, so the
 *           split is small and mechanical: `dg` decides where this thread sits
 *           and how q is indexed and interpolated, `g` decides the carrier
 *           velocity and the labels. Defaults to `g`, which is the identity
 *           and what every test and the CPU-only path use.
 *
 *           Refining the dye alone buys the tracer resolution to HOLD a thin
 *           filament that the velocity field is stretching; it does not invent
 *           structure, because bilinear interpolation makes the carrier
 *           effectively band-limited at `g`. Note also that the CFL is set on
 *           `g.h` (see Simulation.step), so at dyeScale s the backtrace covers
 *           s times as many dye cells — stable either way, but MacCormack's
 *           clamp falls back to first order more often in the fast regions.
 */
export function advectScalar(
  g: Grid,
  u: FieldArray,
  v: FieldArray,
  qIn: FieldArray,
  qOut: FieldArray,
  label: Uint8Array,
  dt: number,
  dg: Grid = g,
): void {
  const h = dg.h;
  const ratio = dg.h / g.h;

  for (let j = 0; j < dg.ny; j++) {
    const y = (j + 0.5) * h;
    const lj = Math.floor((j + 0.5) * ratio);
    for (let i = 0; i < dg.nx; i++) {
      const k = idxP(dg, i, j);
      if (isSolidOrOutside(g, label, Math.floor((i + 0.5) * ratio), lj)) {
        qOut[k] = qIn[k];
        continue;
      }
      const x = (i + 0.5) * h;
      // Both stages interpolate both components: a cell center stores neither
      // u nor v, where advectVelocity gets one of them exact for free.
      backtrace(g, u, v, x, y, sampleU(g, u, x, y), sampleV(g, v, x, y), dt);
      qOut[k] = sampleP(dg, qIn, back.x, back.y);
    }
  }
}

/**
 * MacCormack for a passive scalar — same three-pass correction as
 * advectVelocityMacCormack, see there for the derivation. This is where it
 * shows up most: dye is the field whose sharp edges the eye actually tracks.
 *
 * @param qHat scratch, size of qIn. One buffer serves every dye channel, since
 *             the channels are advected one after another.
 */
export function advectScalarMacCormack(
  g: Grid,
  u: FieldArray,
  v: FieldArray,
  qIn: FieldArray,
  qHat: FieldArray,
  qOut: FieldArray,
  label: Uint8Array,
  dt: number,
  dg: Grid = g,
): void {
  advectScalar(g, u, v, qIn, qHat, label, dt, dg);
  advectScalar(g, u, v, qHat, qOut, label, -dt, dg);

  const h = dg.h;
  const ratio = dg.h / g.h;

  for (let j = 0; j < dg.ny; j++) {
    const y = (j + 0.5) * h;
    const lj = Math.floor((j + 0.5) * ratio);
    for (let i = 0; i < dg.nx; i++) {
      const k = idxP(dg, i, j);
      if (isSolidOrOutside(g, label, Math.floor((i + 0.5) * ratio), lj)) {
        qOut[k] = qIn[k];
        continue;
      }
      const x = (i + 0.5) * h;
      const corrected = qHat[k] + 0.5 * (qIn[k] - qOut[k]);
      backtrace(g, u, v, x, y, sampleU(g, u, x, y), sampleV(g, v, x, y), dt);
      qOut[k] = clampToStencilP(dg, qIn, back.x, back.y, corrected);
    }
  }
}
