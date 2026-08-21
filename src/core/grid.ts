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

// TODO(you), when advection arrives (Step 2):
//   Bilinear interpolation `sampleU(g, u, x, y)` / `sampleV(...)` — needed
//   to look up velocity at an arbitrary point, not just a grid index. This
//   is where the staggered grid's half-cell face offsets actually bite:
//   u[i,j] sits at position (i*h, (j+0.5)*h), not (i*h, j*h). Deferred
//   until advection needs it — projection (Step 1) never samples off-grid.
