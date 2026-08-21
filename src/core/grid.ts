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
 * Note the index convention this implies: cell (i,j) is bounded by
 * u[i,j] on its LEFT and u[i+1,j] on its right; v[i,j] below and
 * v[i,j+1] above. Getting this consistent once, here, is what makes the
 * divergence and gradient stencils fall out without sign hunting.
 */

/** Drives every boundary decision — see Bridson ch. 5. */
export const enum Cell {
  Fluid = 0,
  Air = 1,
  Solid = 2,
}

/**
 * Float64Array for the CPU reference; swap to Float32Array to isolate the
 * precision variable in the Phase 1 vs Phase 2 comparison (PLAN.md section 8).
 */
export type FieldArray = Float64Array;
export type FieldCtor = Float64ArrayConstructor | Float32ArrayConstructor;

export interface Grid {
  readonly nx: number;
  readonly ny: number;
  /** cell size; keep dx === dy, the stencils below assume it */
  readonly h: number;
}

// TODO(you) — start here, in roughly this order:
//
//   1. `createGrid(nx, ny, h, ctor)` allocating p / u / v / labels.
//      Allocate ONCE at startup; kernels reuse these buffers forever.
//
//   2. Index helpers. Write them as functions first (idxP, idxU, idxV);
//      inline them later only if profiling says to.
//         idxP(g, i, j) = i + j * nx
//         idxU(g, i, j) = ?     // stride is nx + 1
//         idxV(g, i, j) = ?     // stride is nx
//
//   3. A test that pins the conventions down before anything depends on
//      them: `node --test`. Assert the array lengths, and assert that the
//      four faces of an interior cell are the indices you expect. This
//      test is boring and it will save you an evening.
//
//   4. Bilinear interpolation `sampleU(g, u, x, y)` — needed by advection,
//      and the place where the half-cell offsets of the staggered grid
//      bite. Not needed for Step 1 (projection), so defer it.
