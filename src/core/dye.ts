import { DYE_CHANNELS, idxP, type FieldArray, type Grid } from './grid.ts';

/**
 * How dye gets INTO the domain, as data rather than as a closure over the
 * host arrays.
 *
 * A source used to be a `(g, dye, dt) => void` callback that wrote straight
 * into `Fields.dye`. That signature is why the fused GPU step had to ship the
 * whole dye field up the bus every frame — 25 MB at 1920x1080 — since the only
 * place a closure could run was the host. Describing the source explicitly
 * instead lets the SAME description drive a host loop and a device kernel, and
 * that is what lets the dye stay resident (see gpu/stepGpu.ts).
 *
 * A DyePatch is a Dirichlet condition on the tracer — a fixed rectangle of
 * prescribed values re-imposed every step, which is what every inlet emitter
 * in scenes/ actually is. Constant in time, so it is built once at reset and
 * uploaded once.
 */

/**
 * A rectangle of prescribed dye, in DYE-grid cells.
 *
 * `data` is DYE_CHANNELS value planes followed by ONE coverage plane, each
 * nx*ny and row-major within the rect. Coverage is what lets a source write
 * only PART of its rectangle: the apply is a lerp, so 1 overwrites, 0 leaves
 * the cell alone, and the fractional values in between are free. Without it a
 * band emitter would have to zero every row outside its band — which erases
 * dye that legitimately drifted there — or carry a separate mask.
 */
export interface DyePatch {
  i0: number;
  j0: number;
  nx: number;
  ny: number;
  data: Float32Array;
}

/** Plane count in DyePatch.data: the channels, plus coverage. */
export const PATCH_PLANES = DYE_CHANNELS + 1;

/**
 * Builds a patch cell by cell. `fill` receives GLOBAL dye-grid indices (so a
 * source can compute world coordinates the same way it always did), writes the
 * channel values into `rgb`, and returns the coverage for that cell.
 */
export function makeDyePatch(
  i0: number,
  j0: number,
  nx: number,
  ny: number,
  fill: (i: number, j: number, rgb: Float32Array) => number,
): DyePatch {
  const cells = nx * ny;
  const data = new Float32Array(PATCH_PLANES * cells);
  const rgb = new Float32Array(DYE_CHANNELS);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      rgb.fill(0);
      const coverage = fill(i0 + i, j0 + j, rgb);
      const k = i + j * nx;
      for (let c = 0; c < DYE_CHANNELS; c++) data[c * cells + k] = rgb[c];
      data[DYE_CHANNELS * cells + k] = coverage;
    }
  }
  return { i0, j0, nx, ny, data };
}

/** The host half of the patch — gpu/project.wgsl's `dye_patch` is the other,
 *  and the lerp has to stay identical or the two engines drift at the inlet. */
export function applyDyePatch(dg: Grid, dye: FieldArray[], patch: DyePatch): void {
  const cells = patch.nx * patch.ny;
  for (let j = 0; j < patch.ny; j++) {
    const dj = patch.j0 + j;
    if (dj < 0 || dj >= dg.ny) continue;
    for (let i = 0; i < patch.nx; i++) {
      const di = patch.i0 + i;
      if (di < 0 || di >= dg.nx) continue;
      const s = i + j * patch.nx;
      const cov = patch.data[DYE_CHANNELS * cells + s];
      if (cov === 0) continue;
      const k = idxP(dg, di, dj);
      for (let c = 0; c < dye.length; c++) {
        dye[c][k] += cov * (patch.data[c * cells + s] - dye[c][k]);
      }
    }
  }
}

/**
 * A length quoted in VELOCITY cells, in DYE cells.
 *
 * Every emitter in scenes/ specifies itself against the solver's grid — "three
 * cells inward from the wall", "a line every twelve cells" — and those are
 * statements about physical size, not about array indices. Refining the dye
 * must not silently halve them: an inlet thinner than the backtrace reaches
 * stops acting like a boundary condition at all.
 */
export function inDyeCells(cells: number, g: Grid, dg: Grid): number {
  return Math.max(1, Math.round((cells * g.h) / dg.h));
}
