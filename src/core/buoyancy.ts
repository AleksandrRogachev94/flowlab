import { isSolidCell } from './boundaries.ts';
import { idxV, sampleP, type FieldArray, type Grid } from './grid.ts';

/**
 * Buoyancy: the one place the dye stops being passive.
 *
 * THE MODEL. Boussinesq, in its usual graphics form (Fedkiw, Stam & Jensen
 * 2001): treat the fluid as incompressible everywhere, and put the whole
 * effect of a density difference into a single vertical body force
 *
 *     f_y = beta * (T - T_ambient)
 *
 * Hot gas is less dense than the air around it and rises. With T_ambient taken
 * as 0 — the whole field is a departure from ambient, which is what a source
 * seeding into a still box already gives — that is one weight on one scalar,
 * and `w` here is beta.
 *
 * WHY NO NEW FIELD. Temperature is an advected scalar that rides the velocity,
 * does not diffuse (there is no viscosity here to diffuse it with), and enters
 * through a fixed rectangle at a wall. That is precisely what a channel of
 * `Fields.dye` already is — advected by both engines on a grid that may be
 * finer than the velocity's, sourced by a DyePatch, faded, drawn. A fourth
 * array would have duplicated the advection kernel, the upload path, the
 * readback and the ping-pong buffers to get a field that behaves identically.
 * So CHANNEL 0 IS THE TEMPERATURE in a scene that sets a weight, and `w` is
 * that declaration. What it costs is that a buoyant scene's channels have to
 * be read as state rather than as tracers; scenes/catalog.ts says which.
 *
 * WHAT IS DELIBERATELY NOT HERE. Fedkiw's force has a second term, -alpha * s,
 * for the weight of the suspended soot, and this had one. It was measured out:
 * with the heat and the smoke both entering at 1 and fading at rates within a
 * factor of two of each other, the soot term is a fixed ~14% of the force at
 * the vent and only outweighs the heat after 14 seconds of sim time — by which
 * point BOTH fields are down at 1e-4 and nothing is visible either way. It was
 * a second weight, a second concept and two extra bilinear gathers per face
 * doing the work of turning 4.0 into 3.5. Soot would earn its place with a
 * fade rate an order of magnitude below the heat's, which is not this scene.
 *
 * WHY dt IS HERE, unlike the stir brush. This is a genuine sustained force, so
 * the momentum it delivers over a stretch of sim time must not depend on how
 * many steps that stretch was cut into: `v += dt * f`. core/stir.ts is the
 * opposite case and its comment explains why.
 */

/** The dye channel a buoyant scene's temperature lives in. gpu/project.wgsl's
 *  `buoyancy` hardcodes the same plane. */
export const HEAT = 0;

/**
 * Adds the buoyancy force to v, BEFORE the projection — the same seam, and for
 * the same reason, as applyStir: a body force is divergent by construction and
 * the pressure solve is what turns it into motion the fluid can actually make.
 * Rising gas has to push something else down, and the solve is where that
 * happens. Applied after subtractGradient it would be a leak the next step
 * inherits.
 *
 * Only v, because gravity is the only direction here.
 *
 * The bounds are subtractGradient's and applyStir's, exactly: interior faces
 * (j in [1, ny-1]) and nothing touching a solid. Both matter for the same
 * reasons they do there — an edge face is prescribed boundary data, and a
 * solid's faces store the solid's velocity, which every other kernel assumes.
 *
 * `dg` is the DYE grid, which may be a refinement of `g`. The face at
 * ((i+0.5)h, jh) is sampled bilinearly out of it, so the two grids meet in
 * world coordinates and nothing here has to know the refinement factor —
 * gpu/project.wgsl's `buoyancy` does the same, and must keep doing it the same
 * way or the engines drift.
 */
export function applyBuoyancy(
  g: Grid,
  dg: Grid,
  v: FieldArray,
  heat: FieldArray,
  label: Uint8Array,
  w: number,
  dt: number,
): void {
  if (w === 0) return;
  const gain = w * dt;
  for (let j = 1; j < g.ny; j++) {
    const y = j * g.h;
    for (let i = 0; i < g.nx; i++) {
      if (isSolidCell(g, label, i, j - 1) || isSolidCell(g, label, i, j)) continue;
      v[idxV(g, i, j)] += gain * sampleP(dg, heat, (i + 0.5) * g.h, y);
    }
  }
}
