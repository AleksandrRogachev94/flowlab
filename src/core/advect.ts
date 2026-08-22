import { idxU, idxV, isSolid, sampleU, sampleV, type FieldArray, type Grid } from './grid.ts';

/**
 * Semi-Lagrangian advection of velocity: u^{n+1}(x) = u^n(x - dt * u).
 *
 * For each face, backtrace to where the fluid arriving there came from and
 * interpolate the old field at that point. Unconditionally stable: the answer
 * is a bilinear blend of existing values, so it cannot exceed the input range
 * however large dt is. The price is numerical dissipation.
 *
 * The backtrace is RK2 (midpoint), not forward Euler:
 *   mid  = x - (dt/2) * vel(x)
 *   prev = x - dt * vel(mid)
 * Both velocity components are needed at each stage even when advecting u
 * alone, since the backtrace moves diagonally. Only the component living at
 * the face itself is read directly; the other is interpolated.
 *
 * NOT in-place (Rule 2): u advects itself, so uIn must survive the pass.
 *
 * @param label per-cell Fluid/Air/Solid. A face touching a solid keeps its
 *              prescribed velocity, COPIED through rather than skipped — uOut
 *              is a separate buffer and would otherwise keep stale ping-pong
 *              data. Copying also preserves a moving wall's velocity, which
 *              zeroing would destroy. These bounds deliberately match
 *              subtractGradient's, so advection and projection own exactly the
 *              same set of faces; a face advection wrote but projection cannot
 *              correct would leak mass through the wall forever.
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
