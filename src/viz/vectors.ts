import { idxU, idxV, type FieldArray, type Grid } from '../core/grid.ts';

/**
 * How arrow LENGTH encodes speed. Direction is always exact; only length
 * is a display choice.
 *
 *  linear  - honest, but useless once one vortex dominates: everything else
 *            collapses to sub-pixel and vanishes.
 *  sqrt    - compresses the range; usually the best default for a flow that
 *            spans an order of magnitude.
 *  unit    - all arrows the same length, direction only. Pair with a speed
 *            heatmap underneath so color carries magnitude. Most readable
 *            for "where is the fluid going", worst for "how fast".
 */
export type ArrowScaling = 'linear' | 'sqrt' | 'unit';

export interface VectorOptions {
  /**
   * Target spacing between arrows, in CSS pixels — not cells. Cell-count
   * spacing looks fine at one resolution and becomes a smear or a void at
   * another; pixel spacing self-scales, since it's what "readable" actually
   * means. Converted to a cell step inside drawVectors from the canvas size
   * and g.nx (assumes a roughly square grid/canvas, true for every scene
   * here).
   */
  spacingPx: number;
  /** pixels per unit speed (after scaling is applied) */
  scale: number;
  /**
   * 'cell' - one arrow per cell at its center, from averaged face values.
   *          The normal view.
   * 'face' - draw each u/v face value AT its face as a half-arrow. Debug
   *          only, small grids only: this renders the staggered storage
   *          layout literally, and is how you confirm the index convention
   *          and see that wall faces really are zero.
   */
  mode: 'cell' | 'face';
  scaling: ArrowScaling;
  color: string;
  /** arrowhead size in pixels; 0 draws bare line segments */
  headSize: number;
  /**
   * Speed mapping to a full-length arrow. Pass the frame's max to make length
   * RELATIVE, keeping the picture's shape as the flow decays; omit for
   * absolute lengths, which visibly shrink. Match this to the heatmap's
   * normalization — mixing the two makes a decaying flow look frozen while
   * its arrows quietly shrink away.
   */
  refSpeed?: number;
}

export const defaultVectorOptions: VectorOptions = {
  spacingPx: 24,
  scale: 12,
  mode: 'cell',
  scaling: 'sqrt',
  color: 'rgba(255,255,255,0.75)',
  headSize: 4,
};

/**
 * Maps a point in GRID coordinates (cell units, origin at bottom-left of
 * cell (0,0)) to canvas pixel coordinates.
 *
 * The y flip here MUST match the one in Heatmap.draw(), or the arrows will
 * disagree with the field rendered underneath them.
 */
export function gridToScreen(
  g: Grid,
  gx: number,
  gy: number,
  w: number,
  h: number,
): [number, number] {
  return [(gx / g.nx) * w, h - (gy / g.ny) * h];
}

/**
 * Cell-centered velocity, averaged from the two opposing faces. This
 * averaging is intrinsic to the MAC grid: no velocity vector is *stored* at
 * a cell center, so producing one always costs an interpolation. The same
 * operation shows up again in vorticity confinement.
 */
export function cellVelocity(
  g: Grid,
  u: FieldArray,
  v: FieldArray,
  i: number,
  j: number,
): [number, number] {
  return [
    0.5 * (u[idxU(g, i, j)] + u[idxU(g, i + 1, j)]),
    0.5 * (v[idxV(g, i, j)] + v[idxV(g, i, j + 1)]),
  ];
}

/**
 * Strokes one arrow from grid-space origin (gx, gy) with grid-space
 * velocity (vx, vy) into an already-open path — caller owns beginPath/stroke.
 */
function pathArrow(
  ctx: CanvasRenderingContext2D,
  g: Grid,
  gx: number,
  gy: number,
  vx: number,
  vy: number,
  w: number,
  h: number,
  opts: VectorOptions,
): void {
  const speed = Math.hypot(vx, vy);
  if (speed < 1e-12) return; // avoids 0/0 in the direction below

  // refSpeed maps speed into [0,1] first, so `scale` becomes the pixel length
  // of the fastest arrow rather than a raw pixels-per-unit-speed factor.
  const t = opts.refSpeed && opts.refSpeed > 0 ? speed / opts.refSpeed : speed;
  const shown = opts.scaling === 'unit' ? 1 : opts.scaling === 'sqrt' ? Math.sqrt(t) : t;
  const k = (shown / speed) * opts.scale;

  const [x1, y1] = gridToScreen(g, gx, gy, w, h);
  // Flip vy: positive vy is "up" in grid space but "down" in canvas space.
  // gridToScreen already flips the position; the vector needs its own flip.
  const dx = vx * k;
  const dy = -vy * k;
  const x2 = x1 + dx;
  const y2 = y1 + dy;

  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);

  if (opts.headSize > 0) {
    const len = Math.hypot(dx, dy);
    const ux = dx / len;
    const uy = dy / len;
    const px = -uy;
    const py = ux;
    const head = opts.headSize;

    ctx.moveTo(x2 - ux * head + px * head * 0.5, y2 - uy * head + py * head * 0.5);
    ctx.lineTo(x2, y2);
    ctx.lineTo(x2 - ux * head - px * head * 0.5, y2 - uy * head - py * head * 0.5);
  }
}

/**
 * Draws the velocity field as arrows onto an already-sized canvas context.
 * Composites on top of whatever's already drawn (e.g. a Heatmap blit) —
 * doesn't clear anything itself.
 */
export function drawVectors(
  ctx: CanvasRenderingContext2D,
  g: Grid,
  u: FieldArray,
  v: FieldArray,
  opts: VectorOptions = defaultVectorOptions,
): void {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;

  ctx.strokeStyle = opts.color;
  ctx.lineWidth = 1;
  ctx.beginPath(); // one path for every arrow — stroke() once at the end

  if (opts.mode === 'cell') {
    // pixels-per-cell -> cells-per-arrow, so spacing reads the same at any N
    const step = Math.max(1, Math.round(opts.spacingPx / (w / g.nx)));
    for (let j = 0; j < g.ny; j += step) {
      for (let i = 0; i < g.nx; i += step) {
        const [vx, vy] = cellVelocity(g, u, v, i, j);
        pathArrow(ctx, g, i + 0.5, j + 0.5, vx, vy, w, h, opts);
      }
    }
  } else {
    // face mode: draw each stored value AT its own face, not averaged.
    // Deliberately not subsampled — this mode exists to inspect small
    // grids in full, not to render a large one legibly.
    for (let j = 0; j < g.ny; j++) {
      for (let i = 0; i <= g.nx; i++) {
        pathArrow(ctx, g, i, j + 0.5, u[idxU(g, i, j)], 0, w, h, opts);
      }
    }
    for (let j = 0; j <= g.ny; j++) {
      for (let i = 0; i < g.nx; i++) {
        pathArrow(ctx, g, i + 0.5, j, 0, v[idxV(g, i, j)], w, h, opts);
      }
    }
  }

  ctx.stroke();
}
