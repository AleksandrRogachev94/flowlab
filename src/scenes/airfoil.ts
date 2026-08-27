/**
 * A NACA four-digit wing section at angle of attack, in the same channel as
 * the vortex street.
 *
 * WHAT THIS SHOWS HONESTLY. The stagnation streamline splitting at the nose;
 * the acceleration over the suction side, which the smoke rake reports
 * directly as line CROWDING (the flux between two material lines is conserved,
 * so where they squeeze together the fluid is moving faster); the pressure
 * side running slow and the lines spreading; and at a high enough angle, the
 * separation that ends all of it — the suction-side flow letting go of the
 * surface and the wake opening up into a broad unsteady mess. Winding the
 * angle up until that happens is the whole reason this scene is here.
 *
 * WHAT IT DOES NOT SHOW. There is no viscosity term and no Kutta condition, so
 * the circulation this settles on is not the physical one — it is set by where
 * the rasterized staircase happens to force the flow to separate, which near a
 * sharp trailing edge is roughly the right place and near a rounded leading
 * edge is not. Read the picture, not a lift coefficient. The same caveat
 * karman.ts gives about Strouhal number, one step stronger: a cylinder's
 * separation is genuinely corner-pinned in the real flow too, and an
 * aerofoil's is not.
 *
 * The section is the standard NACA four-digit family, e.g. 2412: 2% camber, at
 * 40% chord, 12% thick. It is defined by two closed-form curves, which is why
 * it costs a dozen lines here rather than a mesh.
 */

import { Cell, idxP } from '../core/grid.ts';
import { type Grid } from '../core/grid.ts';
import type { LabelSeed, SceneSpec, Seed } from '../core/simulation.ts';
import { allLabels, openRight } from './obstacles.ts';

export interface AirfoilOptions {
  /** Free-stream speed. */
  speed: number;
  /** Chord length, in world units (the channel is 1 tall). */
  chord: number;
  /** QUARTER-CHORD position in world units — the point the section pivots
   *  about, so changing the angle rotates the wing rather than translating it
   *  across the channel. It is also roughly the aerodynamic centre, which is
   *  the conventional reference for exactly that reason. */
  cx: number;
  cy: number;
  /** Positive is nose UP: the flow arrives from below the chord line and the
   *  upper surface becomes the suction side. */
  aoaDeg: number;
  /** NACA four-digit code. */
  code: number;
}

export const defaultAirfoil: AirfoilOptions = {
  speed: 1,
  chord: 0.7,
  cx: 0.5,
  cy: 0.5,
  /**
   * EIGHTEEN degrees — well past stall, and chosen for how much of the frame
   * it puts in motion.
   *
   * Ten degrees is the prettier single picture: attached flow, a clean
   * deflection, the suction side crowding the smoke lines together. It is also
   * nearly a still image, and it occupies almost none of the channel. Measured
   * as the fraction of the frame HEIGHT carrying appreciable vorticity
   * downstream of the wing, at 16:9:
   *
   *     chord 0.56, 10 deg    26%    wake unsteadiness 0.35
   *     chord 0.56, 16 deg    35%                      0.55
   *     chord 0.70, 14 deg    31%                      0.62
   *     chord 0.70, 18 deg    45%                      0.70
   *     chord 0.85, 16 deg    43%                      0.64
   *
   * The angle matters roughly twice as much as the chord does, which makes
   * sense — a bigger wing at a small angle is still a thin obstacle, while a
   * stalled one sheds a wake several times its own thickness. 0.85 of chord
   * buys nothing over 0.70 and starts blocking the channel.
   *
   * So the scene shows the wing DOING the interesting thing rather than the
   * safe one: the flow separates near the leading edge and the whole upper
   * surface sheds, which is both the widest picture available and the most
   * interesting thing a wing does. Drop this to 8-10 for the textbook attached
   * shot.
   */
  aoaDeg: 18,
  code: 2412,
};

/** m (max camber), p (its chord position), t (max thickness), all as fractions
 *  of chord, unpacked from the four-digit code. */
function section(code: number): { m: number; p: number; t: number } {
  return {
    m: Math.floor(code / 1000) / 100,
    p: (Math.floor(code / 100) % 10) / 10,
    t: (code % 100) / 100,
  };
}

/**
 * Half thickness at chord fraction x, the standard four-digit polynomial. The
 * last coefficient is -0.1036 rather than the original -0.1015: that variant
 * closes the trailing edge exactly instead of leaving it open by ~0.2% chord.
 * An open trailing edge would rasterize into a slot the flow leaks through,
 * which is the one place on an aerofoil where a leak changes everything.
 */
function halfThickness(t: number, x: number): number {
  return (
    5 * t * (0.2969 * Math.sqrt(x) - 0.126 * x - 0.3516 * x * x + 0.2843 * x ** 3 - 0.1036 * x ** 4)
  );
}

/** Mean camber line at chord fraction x. Two parabolas meeting at p with
 *  matching slope; flat for a symmetric section, where p would divide by 0. */
function camber(m: number, p: number, x: number): number {
  if (m === 0) return 0;
  return x < p
    ? (m / (p * p)) * (2 * p * x - x * x)
    : (m / (1 - p) ** 2) * (1 - 2 * p + 2 * p * x - x * x);
}

/**
 * The section, rasterized by cell centre like solidDisk.
 *
 * The inside test is done in CHORD coordinates: rotate the cell centre into
 * the wing's frame, then ask whether it lies within the thickness envelope of
 * the camber line. That is the cheap form of the test — it offsets the surface
 * VERTICALLY from the camber line, where the true section offsets it NORMAL to
 * it. The difference is a factor of cos(atan(dyc/dx)), which for 2% camber is
 * under 0.5% of thickness everywhere except the first few percent of chord.
 * Well under one cell, so the full surface construction would rasterize to the
 * same cells and buy nothing.
 */
export function nacaSection(
  cx: number,
  cy: number,
  chord: number,
  aoaDeg: number,
  code: number,
): LabelSeed {
  const { m, p, t } = section(code);
  // The BODY is rotated by -aoa (nose up, since the trailing edge is at
  // positive chord x and must drop). Mapping a world point INTO the chord
  // frame is therefore the inverse, a rotation by +aoa.
  const a = (aoaDeg * Math.PI) / 180;
  const ca = Math.cos(a);
  const sa = Math.sin(a);

  return (g, label) => {
    for (let j = 0; j < g.ny; j++) {
      const dy = (j + 0.5) * g.h - cy;
      for (let i = 0; i < g.nx; i++) {
        const dx = (i + 0.5) * g.h - cx;
        // Chord fractions, with the pivot at quarter chord.
        const xc = (ca * dx - sa * dy) / chord + 0.25;
        if (xc < 0 || xc > 1) continue;
        const yc = (sa * dx + ca * dy) / chord;
        if (Math.abs(yc - camber(m, p, xc)) <= halfThickness(t, xc)) {
          label[idxP(g, i, j)] = Cell.Solid;
        }
      }
    }
  };
}

/**
 * The height the free stream divides at, for the two-tone rake to split on.
 *
 * The LEADING EDGE, not the chord line and not the channel axis. On a lifting
 * section the true dividing streamline arrives a little BELOW the nose — that
 * offset is the circulation, and it is what the colours would report if this
 * solver had a Kutta condition to set it with. It does not (see the file
 * comment), so the nose is the honest approximation: right to within the
 * amount of lift the scene cannot vouch for anyway.
 */
export function leadingEdgeY(options: Partial<AirfoilOptions> = {}): number {
  const { chord, cy, aoaDeg } = { ...defaultAirfoil, ...options };
  return cy + 0.25 * chord * Math.sin((aoaDeg * Math.PI) / 180);
}

/** Wing, outlet and free stream. No dye — see scenes/catalog.ts. */
export function airfoilChannel(_g: Grid, options: Partial<AirfoilOptions> = {}): SceneSpec {
  const { speed, chord, cx, cy, aoaDeg, code } = { ...defaultAirfoil, ...options };

  // Uniform through-flow, impulsively started — karman.ts's seed and its
  // reasoning, unchanged. Starting a wing this way is its own phenomenon: the
  // starting vortex sheds off the trailing edge in the first few moments and
  // convects away, and the circulation left behind on the wing is its equal
  // and opposite. Watch the first second.
  const seed: Seed = (_gg, u) => u.fill(speed);

  return {
    labels: allLabels(openRight(), nacaSection(cx, cy, chord, aoaDeg, code)),
    seed,
  };
}
