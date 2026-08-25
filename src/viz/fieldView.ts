import { Cell, idxP, type Grid } from '../core/grid.ts';
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
  /** Worst single fluid cell. Reported, but NOT the headline: the extremes sit
   *  on the obstacle's staircase corners at ~1000x the median, so this number
   *  tracks the geometry rather than the state of the solve. */
  divMax: number;
  /**
   * RMS |div| over fluid cells, as a fraction of maxSpeed/h — the largest
   * divergence one cell could carry. Dimensionless, so it is comparable across
   * resolutions and flow speeds, where the raw max is not.
   */
  divRms: number;
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
    let divSq = 0;
    let fluidCells = 0;
    for (let j = 0; j < g.ny; j++) {
      for (let i = 0; i < g.nx; i++) {
        const k = idxP(g, i, j);
        const [cu, cv] = cellVelocity(g, f.u, f.v, i, j);
        const s = Math.hypot(cu, cv);
        this.speed[k] = s;
        // FLUID cells only: an Air outlet is supposed to be divergent and a
        // solid's faces are frozen boundary data — either would blow out the
        // stats with values nothing is solving for.
        if (f.label[k] !== Cell.Fluid) continue;
        if (s > maxSpeed) maxSpeed = s;
        const d = Math.abs(div[k]);
        if (d > divMax) divMax = d;
        divSq += d * d;
        fluidCells++;
      }
    }

    const divRms = fluidCells ? Math.sqrt(divSq / fluidCells) / ((maxSpeed || 1) / g.h) : 0;

    if (view === 'vorticity') {
      computeVorticity(g, f.u, f.v, this.vort);
      // Sign carries the meaning, so a counter-rotating pair must read as
      // opposite colours around zero. iceFire not coolwarm: zero must be DARK
      // or the white arrows vanish over the calm majority of the frame.
      //
      // p99 and not the max: the max lives on the cylinder's staircase corners
      // at ~5x anything in the wake, so scaling to it renders the whole vortex
      // street nearly black. Solid cells need no masking here — drawSolids
      // paints over them, and at p99 they do not move the range at all.
      this.heatmap.draw(this.vort, ctx, {
        normalization: { kind: 'percentile', p: 0.99 },
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
      //
      // p99, for the same reason as vorticity above and with the same cause:
      // |div| spans ~1000x between its median and its max, and the max lives on
      // the cylinder's staircase corners. Scaling to it painted every other cell
      // at the ramp's neutral midpoint, i.e. a blank frame with three specks —
      // hiding exactly the interior residual this view exists to show.
      this.heatmap.draw(div, ctx, {
        normalization: { kind: 'percentile', p: 0.99 },
        colormap: coolwarm,
      });
    }

    this.drawSolids(ctx, g, f.label);

    // Device pixels per CSS pixel: the backing store is oversampled on a retina
    // display, so every pixel length below must scale with it.
    const dpr = ctx.canvas.width / (ctx.canvas.clientWidth || ctx.canvas.width);

    // refSpeed matches the heatmap's per-frame normalization, so the picture
    // shows structure while magnitude lives in the readout.
    drawVectors(ctx, g, f.u, f.v, {
      ...defaultVectorOptions,
      spacingPx: 26 * dpr,
      scale: 14 * dpr,
      headSize: 3.5 * dpr,
      lineWidth: dpr,
      color: 'rgba(255, 255, 255, 0.62)',
      refSpeed: maxSpeed,
    });

    return { maxSpeed, divMax, divRms };
  }

  /**
   * Paint solid cells over the field, after the heatmap and before the arrows.
   * Inside a solid no view is showing data, so masking it beats inviting the
   * reader to see structure in it. Drawn as the RASTERIZED cells, staircase and
   * all: the staircase is what the solver actually sees — it pins the
   * separation points — so a smooth circle would misrepresent the result.
   *
   * ONE path filled ONCE, not a fillRect per cell. Cell edges land on
   * fractional pixels, and two separate fills sharing such a pixel composite to
   * 1-(1-a)(1-b) < 1, letting the background bleed through as a thin seam — a
   * visible grid of stripes across the body. Filling the union computes that
   * pixel's coverage a single time, so interior edges disappear.
   */
  private drawSolids(ctx: CanvasRenderingContext2D, g: Grid, label: Uint8Array): void {
    const h = ctx.canvas.height;
    const cw = ctx.canvas.width / g.nx;
    const ch = h / g.ny;

    ctx.beginPath();
    for (let j = 0; j < g.ny; j++) {
      // Same y flip as Heatmap.draw and gridToScreen.
      const y = h - (j + 1) * ch;
      for (let i = 0; i < g.nx; i++) {
        if (label[idxP(g, i, j)] === Cell.Solid) ctx.rect(i * cw, y, cw, ch);
      }
    }
    // Pale, not black: black reads as a hole in the background, and a light
    // body stays legible under both the dark dye view and the dark-at-zero
    // vorticity map. It is also what reference images of this benchmark use.
    ctx.fillStyle = '#b4becc';
    ctx.fill();
  }
}
