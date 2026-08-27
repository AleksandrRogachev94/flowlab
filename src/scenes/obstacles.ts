/**
 * Geometry seeds: they write `label`, nothing else. Coordinates are in WORLD
 * units like every other seed in this directory, so the same scene rasterizes
 * to the same physical shape at any resolution.
 *
 * These do NOT restore the invariants a label change implies (zeroed faces,
 * pinned pressure, cleared dye) — Simulation.reset() calls commitLabels() once
 * after all of them, so a shape here stays a shape.
 */

import { Cell, idxP, type Grid } from '../core/grid.ts';
import type { LabelSeed } from '../core/simulation.ts';

/** Applies several label seeds in order; later ones win where they overlap. */
export function allLabels(...seeds: LabelSeed[]): LabelSeed {
  return (g, label) => {
    for (const s of seeds) s(g, label);
  };
}

/**
 * A solid disk. Rasterized by CELL CENTRE, so the surface is a staircase.
 *
 * That staircase is not only a cosmetic approximation — its corners pin the
 * separation points, which is a real part of why an inviscid solver sheds
 * vortices at all, and equally why the separation angle here will not move
 * with Reynolds number the way a physical one does. Cut cells would fix that
 * and cost a fractional-area rewrite of every kernel; deliberately not done.
 */
export function solidDisk(cx: number, cy: number, r: number): LabelSeed {
  return (g, label) => {
    const r2 = r * r;
    for (let j = 0; j < g.ny; j++) {
      const y = (j + 0.5) * g.h;
      for (let i = 0; i < g.nx; i++) {
        const x = (i + 0.5) * g.h;
        if ((x - cx) ** 2 + (y - cy) ** 2 <= r2) label[idxP(g, i, j)] = Cell.Solid;
      }
    }
  };
}

/**
 * An axis-aligned solid slab, corners in WORLD units. Rasterized by cell
 * centre like solidDisk, so a slab thinner than a cell can vanish entirely —
 * quote thin plates as a multiple of g.h rather than as a fixed length, or a
 * low resolution silently deletes them.
 */
export function solidRect(x0: number, y0: number, x1: number, y1: number): LabelSeed {
  return (g, label) => {
    for (let j = 0; j < g.ny; j++) {
      const y = (j + 0.5) * g.h;
      if (y < y0 || y > y1) continue;
      for (let i = 0; i < g.nx; i++) {
        const x = (i + 0.5) * g.h;
        if (x >= x0 && x <= x1) label[idxP(g, i, j)] = Cell.Solid;
      }
    }
  };
}

/**
 * Paints a SOLID capsule — a segment of radius r — into an existing label
 * field, immediately, rather than returning a seed to be applied at reset.
 * The interactive brush. Returns whether anything actually changed, so the
 * caller can skip the label upload and the solids-mask rebuild on a stroke
 * that landed entirely on cells that were already solid.
 *
 * A CAPSULE and not a disk because a pointer is sampled once per frame and
 * moves a long way between samples. Stamping a disk per sample leaves a dotted
 * line at any speed a person actually drags at; sweeping the disk along the
 * segment since the last sample is the same cost and draws a continuous
 * stroke.
 *
 * ONLY Fluid becomes Solid, which is a safety rule and not an optimisation.
 * The outlet is a column of Air cells, and painting over it would turn a
 * channel back into a closed box — an all-Neumann system with a large net
 * inflow, which has no solution at all and which the solver would express as a
 * pressure field diverging without bound. Restricting the brush to Fluid also
 * makes a second stroke over the same place a no-op rather than a rewrite.
 */
export function paintSolid(
  g: Grid,
  label: Uint8Array,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  r: number,
): boolean {
  const dx = x1 - x0;
  const dy = y1 - y0;
  const lenSq = dx * dx + dy * dy;
  const r2 = r * r;
  let changed = false;

  // Only the cells the capsule's bounding box can reach.
  const lo = (w: number): number => Math.max(0, Math.floor(w / g.h - 0.5));
  const i0 = lo(Math.min(x0, x1) - r);
  const i1 = Math.min(g.nx - 1, Math.ceil((Math.max(x0, x1) + r) / g.h - 0.5));
  const j0 = lo(Math.min(y0, y1) - r);
  const j1 = Math.min(g.ny - 1, Math.ceil((Math.max(y0, y1) + r) / g.h - 0.5));

  for (let j = j0; j <= j1; j++) {
    const y = (j + 0.5) * g.h;
    for (let i = i0; i <= i1; i++) {
      const k = idxP(g, i, j);
      if (label[k] !== Cell.Fluid) continue;
      const x = (i + 0.5) * g.h;
      // Distance to the SEGMENT: project onto it, clamp to its ends. lenSq 0
      // is a single click, where t collapses to 0 and this is a plain disk.
      const t = lenSq > 0 ? Math.min(Math.max(((x - x0) * dx + (y - y0) * dy) / lenSq, 0), 1) : 0;
      const px = x0 + t * dx - x;
      const py = y0 + t * dy - y;
      if (px * px + py * py <= r2) {
        label[k] = Cell.Solid;
        changed = true;
      }
    }
  }
  return changed;
}

/**
 * The rightmost column becomes Air — an open outflow.
 *
 * Two things happen at once, and both matter. The pressure system stops being
 * all-Neumann, so it is nonsingular and consistent for ANY inflow: no flux
 * budget to balance, and p no longer drifts by an arbitrary constant. And the
 * exit stops being prescribed: a fixed outflow profile REFLECTS, so a vortex
 * arriving at the boundary bounces back into the wake instead of leaving. A
 * vortex street cannot be measured without this.
 *
 * One column is all it takes: p = 0 is imposed on the fluid cells adjacent to
 * it, which is where the condition acts. The column itself is outside the
 * fluid domain and is never solved.
 */
export function openRight(): LabelSeed {
  return (g, label) => {
    for (let j = 0; j < g.ny; j++) label[idxP(g, g.nx - 1, j)] = Cell.Air;
  };
}
