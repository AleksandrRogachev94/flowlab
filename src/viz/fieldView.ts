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

/** The solver's residual, summarized. */
export interface DivStats {
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

/**
 * A DIAGNOSTIC, and a full pass over every cell — which is why it is its own
 * function rather than something draw() computes on the way past.
 *
 * It used to run inside draw(), every frame, over a `div` that Simulation only
 * refreshes every `residualEvery` steps, to produce numbers the status line
 * discards unless the readout is on. At 2.12M cells that was a couple of ms a
 * frame spent on nothing. main.ts now calls it only when something reads it.
 */
export function divergenceStats(sim: Simulation, maxSpeed: number): DivStats {
  const { g, f, div } = sim;
  // FLUID cells only: an Air outlet is supposed to be divergent and a solid's
  // faces are frozen boundary data — either would blow out the stats with
  // values nothing is solving for.
  let divMax = 0;
  let divSq = 0;
  let fluidCells = 0;
  for (let k = 0; k < div.length; k++) {
    if (f.label[k] !== Cell.Fluid) continue;
    const d = Math.abs(div[k]);
    if (d > divMax) divMax = d;
    divSq += d * d;
    fluidCells++;
  }
  const divRms = fluidCells ? Math.sqrt(divSq / fluidCells) / ((maxSpeed || 1) / g.h) : 0;
  return { divMax, divRms };
}

/** Renders a Simulation as a heatmap plus arrows. Owns its scratch buffers,
 *  allocated once (Rule 3). */
export class FieldView {
  private readonly heatmap: Heatmap;
  /**
   * A second heatmap at the DYE grid's size, used by the dye view alone.
   *
   * Dye may be stored finer than the velocity field (Simulation.dyeG), and
   * `heatmap` is one pixel per VELOCITY cell — feeding it dye-sized arrays
   * would read the wrong rows. Two objects rather than one resized per view,
   * because at dyeScale 1 they are the same size and this costs nothing.
   */
  private readonly dyeHeatmap: Heatmap;
  private readonly speed: Float64Array;
  private readonly vort: Float64Array;
  /** Cached solid-cell mask — see drawSolids. */
  private solidsPath: Path2D | null = null;
  private solidsW = 0;
  private solidsH = 0;

  constructor(g: Grid, dg: Grid = g) {
    this.heatmap = new Heatmap(g.nx, g.ny);
    this.dyeHeatmap = dg === g ? this.heatmap : new Heatmap(dg.nx, dg.ny);
    this.speed = new Float64Array(g.nx * g.ny);
    this.vort = new Float64Array(g.nx * g.ny);
  }

  /** The labels changed (a restart, a scene switch); the cached solid mask no
   *  longer matches and the next draw rebuilds it. */
  invalidateSolids(): void {
    this.solidsPath = null;
  }

  /**
   * `arrows` is off by default because the picture is better without them:
   * the dye and vorticity views already carry direction in their structure,
   * and a full-screen grid of white arrows sits on top of exactly the
   * filaments the scheme comparison is about. It stays available because
   * direction is the one thing a scalar field genuinely cannot show — a
   * recirculation bubble and a fast through-flow can render identically.
   */
  /**
   * @param dyeOnDevice the dye view is being painted by viz/dyeGpu.ts onto the
   *        WebGPU canvas UNDER this one, so this canvas must be cleared to
   *        transparent rather than blitted over — it still carries the solids
   *        mask and the arrows, which are the two overlays that are cheaper to
   *        keep in 2D than to port.
   */
  draw(
    ctx: CanvasRenderingContext2D,
    sim: Simulation,
    view: View,
    arrows = false,
    dyeOnDevice = false,
  ): number {
    const { g, f } = sim;

    // The cell-centred speed field is filled only for the view that shows it.
    // It used to be built every frame, and at 1920x1080 that one loop — 2M
    // cellVelocity gathers plus a Math.hypot each — cost more than the entire
    // GPU pressure solve it was drawn next to. Every other consumer of
    // maxSpeed (the arrows' refSpeed, divRms's normalization) only needs a
    // scale, and the FACE max is that scale: it brackets the cell-centred max
    // from above by at most one face's worth of averaging.
    let maxSpeed = 0;
    if (view === 'speed') {
      for (let j = 0; j < g.ny; j++) {
        for (let i = 0; i < g.nx; i++) {
          const k = idxP(g, i, j);
          const [cu, cv] = cellVelocity(g, f.u, f.v, i, j);
          const s = Math.sqrt(cu * cu + cv * cv);
          this.speed[k] = s;
          if (f.label[k] === Cell.Fluid && s > maxSpeed) maxSpeed = s;
        }
      }
    } else {
      maxSpeed = sim.maxFaceSpeed();
    }

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
      if (dyeOnDevice) {
        // The picture is already on the canvas behind this one; anything drawn
        // here would hide it. f.dye is stale in this case anyway — see
        // GpuStep.readDye.
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      } else {
        // Straight to RGB — see Heatmap.drawRGB. No colormap, and no
        // normalization: dye is seeded at 1 and only dissipates, so the
        // absolute brightness IS the measurement.
        const [r, gr, b] = f.dye;
        this.dyeHeatmap.drawRGB(r, gr, b, ctx, { smooth: true });
      }
    } else {
      // coolwarm keeps its near-white zero here: the debug question is "is
      // anything nonzero", and specks of colour read fastest against white.
      //
      // p99, for the same reason as vorticity above and with the same cause:
      // |div| spans ~1000x between its median and its max, and the max lives on
      // the cylinder's staircase corners. Scaling to it painted every other cell
      // at the ramp's neutral midpoint, i.e. a blank frame with three specks —
      // hiding exactly the interior residual this view exists to show.
      this.heatmap.draw(sim.div, ctx, {
        normalization: { kind: 'percentile', p: 0.99 },
        colormap: coolwarm,
      });
    }

    this.drawSolids(ctx, g, f.label);

    if (arrows) {
      // Device pixels per CSS pixel: the backing store is oversampled on a
      // retina display, so every pixel length below must scale with it.
      const dpr = ctx.canvas.width / (ctx.canvas.clientWidth || ctx.canvas.width);

      // refSpeed matches the heatmap's per-frame normalization, so the picture
      // shows structure while magnitude lives in the readout.
      drawVectors(ctx, g, f.u, f.v, {
        ...defaultVectorOptions,
        // Wider and longer than they were while they were always-on: an
        // optional overlay should be legible when you ask for it, and it no
        // longer has to stay out of the way of every screenshot.
        spacingPx: 34 * dpr,
        scale: 20 * dpr,
        headSize: 4.5 * dpr,
        lineWidth: dpr,
        color: 'rgba(255, 255, 255, 0.72)',
        refSpeed: maxSpeed,
      });
    }

    return maxSpeed;
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
    const w = ctx.canvas.width;
    const h = ctx.canvas.height;

    // The union is CACHED as a Path2D: labels change only on restart() (which
    // calls invalidateSolids) and the geometry only on resize (caught by the
    // size check), yet the scan is over every cell — 2M label reads a frame at
    // 1920x1080 to rediscover the same cylinder.
    if (!this.solidsPath || w !== this.solidsW || h !== this.solidsH) {
      this.solidsW = w;
      this.solidsH = h;
      this.solidsPath = new Path2D();
      const cw = w / g.nx;
      const ch = h / g.ny;
      for (let j = 0; j < g.ny; j++) {
        // Same y flip as Heatmap.draw and gridToScreen.
        const y = h - (j + 1) * ch;
        for (let i = 0; i < g.nx; i++) {
          if (label[idxP(g, i, j)] === Cell.Solid) this.solidsPath.rect(i * cw, y, cw, ch);
        }
      }
    }
    // A DARK body with a lit rim, and the rim is what makes the dark possible.
    //
    // This used to be a pale grey fill, on the argument that black reads as a
    // hole. True, but the fill was competing: the dye view's background IS
    // black, so a near-white slab was the brightest object in the frame and it
    // sat in the middle of it, pulling the eye off the smoke it exists to
    // interrupt. The rim keeps the silhouette legible on both dark views — the
    // dye view and iceFire's dark-at-zero vorticity map — while the body
    // itself recedes to roughly the background, so the picture is the flow.
    //
    // Drawn as a SHADOW and not a stroke, and that is forced rather than
    // stylistic. solidsPath is a union of one rect per solid cell, so stroking
    // it would outline every interior cell edge as well as the outline, and
    // the body would come back as a bright grid. A shadow is cast from the
    // composited alpha of the whole path, which has no interior edges in it —
    // the same reason the path is filled once rather than per cell, one step
    // further on. Two passes because a single one is barely visible at this
    // radius, and the fill is opaque so the glow only ever shows outside.
    const dpr = ctx.canvas.width / (ctx.canvas.clientWidth || ctx.canvas.width);
    ctx.save();
    ctx.shadowColor = 'rgba(150, 200, 255, 0.75)';
    ctx.shadowBlur = 9 * dpr;
    ctx.fillStyle = '#161c26';
    ctx.fill(this.solidsPath);
    ctx.fill(this.solidsPath);
    ctx.restore();
  }
}
