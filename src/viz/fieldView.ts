import { idxP, type Grid } from '../core/grid.ts';
import type { Simulation } from '../core/simulation.ts';
import { computeVorticity } from '../core/vorticity.ts';
import { coolwarm, iceFire, ocean } from './colormaps.ts';
import { Heatmap } from './heatmap.ts';
import { cellVelocity, defaultVectorOptions, drawVectors } from './vectors.ts';

/**
 * 'vorticity'  signed curl — the analysis view. Speed is smooth and hides
 *              structure; vorticity concentrates at cores and shear layers.
 * 'speed'      magnitude — reads well as a picture, weakest for structure.
 * 'dye'        three passive tracers composited as RGB — the only view showing
 *              where fluid GOES rather than how fast it moves. Colours that
 *              were never seeded mark where channels have interleaved below
 *              the grid scale, which is advection's dissipation made visible.
 * 'divergence' projection residual — the debug view.
 */
export const VIEWS = ['dye', 'vorticity', 'speed', 'divergence'] as const;
export type View = (typeof VIEWS)[number];

/** What the frame was normalized by. The picture is rescaled every frame, so
 *  only these numbers show decay and convergence. */
export interface ViewStats {
  maxSpeed: number;
  divMax: number;
}

/** Renders a Simulation as a heatmap plus arrows. Owns its scratch buffers,
 *  allocated once (Rule 3). */
export class FieldView {
  private readonly heatmap: Heatmap;
  private readonly speed: Float64Array;
  private readonly vort: Float64Array;

  constructor(g: Grid) {
    this.heatmap = new Heatmap(g.nx, g.ny);
    this.speed = new Float64Array(g.nx * g.ny);
    this.vort = new Float64Array(g.nx * g.ny);
  }

  draw(ctx: CanvasRenderingContext2D, sim: Simulation, view: View): ViewStats {
    const { g, f, div } = sim;

    let maxSpeed = 0;
    let divMax = 0;
    for (let j = 0; j < g.ny; j++) {
      for (let i = 0; i < g.nx; i++) {
        const k = idxP(g, i, j);
        const [cu, cv] = cellVelocity(g, f.u, f.v, i, j);
        const s = Math.hypot(cu, cv);
        this.speed[k] = s;
        if (s > maxSpeed) maxSpeed = s;
        const d = Math.abs(div[k]);
        if (d > divMax) divMax = d;
      }
    }

    if (view === 'vorticity') {
      computeVorticity(g, f.u, f.v, this.vort);
      // Sign carries the meaning, so a counter-rotating pair must read as
      // opposite colours around zero. iceFire not coolwarm: zero must be DARK
      // or the white arrows vanish over the calm majority of the frame.
      this.heatmap.draw(this.vort, ctx, {
        normalization: { kind: 'symmetric' },
        colormap: iceFire,
        smooth: true,
      });
    } else if (view === 'speed') {
      // fixed [0, max], not 'auto': zero must land at the bottom of the ramp,
      // not at this frame's slowest nonzero cell.
      this.heatmap.draw(this.speed, ctx, {
        normalization: { kind: 'fixed', min: 0, max: maxSpeed || 1 },
        colormap: ocean,
        smooth: true,
      });
    } else if (view === 'dye') {
      // Straight to RGB — see Heatmap.drawRGB. No colormap, and no
      // normalization: dye is seeded at 1 and only dissipates, so the absolute
      // brightness IS the measurement.
      const [r, gr, b] = f.dye;
      this.heatmap.drawRGB(r, gr, b, ctx, { smooth: true });
    } else {
      // coolwarm keeps its near-white zero here: the debug question is "is
      // anything nonzero", and specks of colour read fastest against white.
      this.heatmap.draw(div, ctx, {
        normalization: { kind: 'fixed', min: -divMax, max: divMax },
        colormap: coolwarm,
      });
    }

    // refSpeed matches the heatmap's per-frame normalization, so the picture
    // shows structure while magnitude lives in the readout.
    drawVectors(ctx, g, f.u, f.v, {
      ...defaultVectorOptions,
      spacingPx: 20,
      scale: 13,
      headSize: 3,
      color: 'rgba(255, 255, 255, 0.85)',
      refSpeed: maxSpeed,
    });

    return { maxSpeed, divMax };
  }
}
