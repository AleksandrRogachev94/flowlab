import { idxU, idxV, isSolid, sampleU, sampleV, type FieldArray, type Grid } from './grid.ts';

/**
 * Semi-Lagrangian advection of velocity: u^{n+1}(x) = u^n(x - dt * u).
 * Backtrace each face to where its fluid came from and interpolate there.
 * Unconditionally stable — every output is a bilinear blend of inputs, so it
 * cannot exceed the input range at any dt. The price is dissipation.
 *
 * RK2 (midpoint), not forward Euler:
 *   mid  = x - (dt/2) * vel(x)
 *   prev = x - dt * vel(mid)
 * Both components are needed at each stage even when advecting u alone, since
 * the backtrace moves diagonally; only the one living at the face is exact.
 *
 * NOT in-place (Rule 2): u advects itself, so uIn must survive the pass.
 *
 * @param label a face touching a solid keeps its prescribed velocity, COPIED
 *              through rather than skipped — uOut is a separate buffer and
 *              would otherwise keep stale ping-pong data. Bounds match
 *              subtractGradient's, so a face advection writes is always one
 *              projection can correct.
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
  const h = g.h;
  const halfDt = 0.5 * dt;

  for (let j = 0; j < g.ny; j++) {
    const y = (j + 0.5) * h;
    for (let i = 0; i <= g.nx; i++) {
      const k = idxU(g, i, j);
      if (isSolid(g, label, i - 1, j) || isSolid(g, label, i, j)) {
        uOut[k] = uIn[k];
        continue;
      }
      const x = i * h;
      const midX = x - halfDt * uIn[k];
      const midY = y - halfDt * sampleV(g, vIn, x, y);
      const prevX = x - dt * sampleU(g, uIn, midX, midY);
      const prevY = y - dt * sampleV(g, vIn, midX, midY);
      uOut[k] = sampleU(g, uIn, prevX, prevY);
    }
  }

  for (let j = 0; j <= g.ny; j++) {
    const y = j * h;
    for (let i = 0; i < g.nx; i++) {
      const k = idxV(g, i, j);
      if (isSolid(g, label, i, j - 1) || isSolid(g, label, i, j)) {
        vOut[k] = vIn[k];
        continue;
      }
      const x = (i + 0.5) * h;
      const midX = x - halfDt * sampleU(g, uIn, x, y);
      const midY = y - halfDt * vIn[k];
      const prevX = x - dt * sampleU(g, uIn, midX, midY);
      const prevY = y - dt * sampleV(g, vIn, midX, midY);
      vOut[k] = sampleV(g, vIn, prevX, prevY);
    }
  }
}
