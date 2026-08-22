/**
 * MAC (staggered) grid storage.
 *
 * For an nx x ny grid of cells with spacing h:
 *
 *   p   at cell CENTERS          -> nx * ny
 *   u   at VERTICAL faces (x-vel) -> (nx + 1) * ny
 *   v   at HORIZONTAL faces (y-vel) -> nx * (ny + 1)
 *
 *        v[i, j+1]
 *      +-----^-----+
 *      |           |
 *  u[i,j]  p[i,j]  u[i+1,j]
 *      >           >
 *      |           |
 *      +-----^-----+
 *        v[i, j]
 *
 * There are two equivalent ways to read the same convention, and both
 * come up constantly — they are NOT in conflict:
 *
 *   Per cell:  cell (i,j) is bounded by u[i,j] LEFT / u[i+1,j] RIGHT,
 *              and v[i,j] BELOW / v[i,j+1] ABOVE.
 *   Per face:  u[i,j] separates cell (i-1,j) from cell (i,j);
 *              v[i,j] separates cell (i,j-1) from cell (i,j).
 *
 * i.e. a face's index matches the cell on its HIGH side. Reading the
 * per-face rule with i+1 substituted recovers the per-cell rule:
 * u[i+1,j] separates cell (i,j) from (i+1,j) — cell (i,j)'s right face.
 *
 *      cell (i-1,j)   cell (i,j)   cell (i+1,j)
 *    |             |             |             |
 *   u[i-1,j]     u[i,j]       u[i+1,j]      u[i+2,j]
 *                   \             /
 *                    left of (i,j)  right of (i,j)
 */

/**
 * Drives every boundary decision — see Bridson ch. 5.
 */
export const Cell = {
  Fluid: 0,
  Air: 1,
  Solid: 2,
} as const;
export type Cell = (typeof Cell)[keyof typeof Cell];

/**
 * Float64Array for the CPU reference; the union with Float32Array exists
 * so `createFields` can hand back Float32-backed fields to isolate the
 * precision variable in the Phase 1 vs Phase 2 comparison (PLAN.md §8).
 */
export type FieldArray = Float64Array | Float32Array;
export type FieldCtor = Float64ArrayConstructor | Float32ArrayConstructor;

/** Pure geometry — no arrays. Shared by every field (p, u, v, dye, ...). */
export interface Grid {
  readonly nx: number;
  readonly ny: number;
  /** cell size; keep dx === dy, the stencils below assume it */
  readonly h: number;
}

/** The MAC grid's storage. See the header comment for the layout/strides. */
export interface Fields {
  p: FieldArray;
  u: FieldArray;
  v: FieldArray;
  label: Uint8Array;
}

export function createGrid(nx: number, ny: number, h: number): Grid {
  return { nx, ny, h };
}

/**
 * Allocates the four MAC-grid buffers. Zero-initialized, which means every
 * cell starts labeled Cell.Fluid (0) — solid walls and boundaries are set
 * up later by scenario code (Step 3), not here.
 */
export function createFields(g: Grid, ctor: FieldCtor): Fields {
  return {
    p: new ctor(g.nx * g.ny),
    u: new ctor((g.nx + 1) * g.ny),
    v: new ctor(g.nx * (g.ny + 1)),
    label: new Uint8Array(g.nx * g.ny),
  };
}

/** Cell-center index: p[i,j] and label[i,j]. Row-major, stride nx. */
export function idxP(g: Grid, i: number, j: number): number {
  return i + j * g.nx;
}

/**
 * Vertical-face index: u[i,j], the x-velocity on the face between cell
 * (i-1,j) and cell (i,j). A row of nx cells has nx+1 such faces (one on
 * each side, like fence posts around fence gaps), so the stride is nx+1,
 * not nx — that's the one detail that differs from idxP/idxV.
 */
export function idxU(g: Grid, i: number, j: number): number {
  return i + j * (g.nx + 1);
}

/**
 * Horizontal-face index: v[i,j], the y-velocity on the face between cell
 * (i,j-1) and cell (i,j). The stride here is nx (same as idxP) — the +1
 * shows up in the number of rows (ny+1), not in how many faces fit in one
 * row, which is why this is NOT the same formula as idxU with x/y swapped.
 */
export function idxV(g: Grid, i: number, j: number): number {
  return i + j * g.nx;
}

/**
 * Is cell (i,j) unavailable to couple with — either a labeled Solid, or
 * outside the domain entirely? Outside-as-solid is how the closed box gets
 * its walls without a ghost ring around the array.
 *
 * The bounds check MUST run before the label lookup: idxP wraps at row
 * boundaries (it's just i + j*nx), so an out-of-range (i,j) can silently
 * alias a different, valid cell instead of failing loudly. Checking bounds
 * first is what prevents that.
 */
export function isSolid(g: Grid, label: Uint8Array, i: number, j: number): boolean {
  return i < 0 || j < 0 || i >= g.nx || j >= g.ny || label[idxP(g, i, j)] === Cell.Solid;
}

/**
 * Splits a continuous local-index coordinate into a base index i0 and a
 * fraction f, clamped so i0 and i0+1 are both valid.
 *
 * The two bounds differ on purpose: the POSITION clamps to count-1, so an
 * out-of-domain point interpolates at the edge instead of extrapolating off
 * it; the INDEX clamps to count-2, so i0+1 stays in range at the far edge,
 * where the position clamp lands exactly on count-1.
 *
 * Clamping happens here, after the caller applies the half-cell offset —
 * clamping raw x/y misses sampleU at y=0, which needs floor(-0.5) = -1.
 */
function clampedAxis(pos: number, count: number): { i0: number; f: number } {
  const clamped = Math.min(Math.max(pos, 0), count - 1);
  const i0 = Math.min(Math.floor(clamped), count - 2);
  return { i0, f: clamped - i0 };
}

/** Bilinear blend of four corners, as nested lerps: 3 multiplies, not 8. */
function bilerp(
  v00: number,
  v10: number,
  v01: number,
  v11: number,
  fx: number,
  fy: number,
): number {
  const lo = v00 + fx * (v10 - v00);
  const hi = v01 + fx * (v11 - v01);
  return lo + fy * (hi - lo);
}

/**
 * Bilinear sample of u at an arbitrary world-space point. u[i,j] sits at
 * (i*h, (j+0.5)*h) — hence the -0.5 on y only — and has nx+1 by ny values.
 */
export function sampleU(g: Grid, u: FieldArray, x: number, y: number): number {
  const { i0, f: fx } = clampedAxis(x / g.h, g.nx + 1);
  const { i0: j0, f: fy } = clampedAxis(y / g.h - 0.5, g.ny);
  return bilerp(
    u[idxU(g, i0, j0)],
    u[idxU(g, i0 + 1, j0)],
    u[idxU(g, i0, j0 + 1)],
    u[idxU(g, i0 + 1, j0 + 1)],
    fx,
    fy,
  );
}

/**
 * Bilinear sample of v at an arbitrary world-space point. v[i,j] sits at
 * ((i+0.5)*h, j*h) — the offset is on x here — and has nx by ny+1 values.
 */
export function sampleV(g: Grid, v: FieldArray, x: number, y: number): number {
  const { i0, f: fx } = clampedAxis(x / g.h - 0.5, g.nx);
  const { i0: j0, f: fy } = clampedAxis(y / g.h, g.ny + 1);
  return bilerp(
    v[idxV(g, i0, j0)],
    v[idxV(g, i0 + 1, j0)],
    v[idxV(g, i0, j0 + 1)],
    v[idxV(g, i0 + 1, j0 + 1)],
    fx,
    fy,
  );
}
