/**
 * Geometry seeds: they write `label`, nothing else. Coordinates are in WORLD
 * units like every other seed in this directory, so the same scene rasterizes
 * to the same physical shape at any resolution.
 *
 * These do NOT restore the invariants a label change implies (zeroed faces,
 * pinned pressure, cleared dye) — Simulation.reset() calls commitLabels() once
 * after all of them, so a shape here stays a shape.
 */

import { Cell, idxP } from '../core/grid.ts';
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
