import type { Colormap } from './colormaps.ts';

export type Normalization =
  /** map [min,max] of this frame's data onto [0,1] */
  | { kind: 'auto' }
  /** map [-m,m] onto [0,1] where m = max|value|, so 0 lands exactly at 0.5 */
  | { kind: 'symmetric' }
  /** fixed range, so colors are comparable across frames */
  | { kind: 'fixed'; min: number; max: number };

export interface DrawOptions {
  colormap: Colormap;
  normalization: Normalization;
}

/**
 * Resolve a Normalization choice against actual data to concrete [lo, hi]
 * bounds. Pulled out of Heatmap.draw() as a pure function (no canvas) so it
 * can be unit tested directly with node --test.
 */
export function computeRange(
  field: Float64Array,
  normalization: Normalization,
): { lo: number; hi: number } {
  switch (normalization.kind) {
    case 'auto': {
      let lo = Infinity;
      let hi = -Infinity;
      for (let i = 0; i < field.length; i++) {
        const v = field[i];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      return { lo, hi };
    }
    case 'symmetric': {
      // Track magnitude in its own variable — assigning the signed value
      // into the bound directly (as an earlier version of this did) lets a
      // negative value poison the running max, since abs(v) >= 0 always
      // beats a negative "hi" on the next iteration regardless of true size.
      let m = 0;
      for (let i = 0; i < field.length; i++) {
        const a = Math.abs(field[i]);
        if (a > m) m = a;
      }
      return { lo: -m, hi: m };
    }
    case 'fixed':
      return { lo: normalization.min, hi: normalization.max };
  }
}

/**
 * Renders a scalar field (flat array, row-major, nx*ny) as a color image.
 *
 * The whole trick: the per-pixel loop runs at GRID resolution into a tiny
 * offscreen canvas, and the GPU does the upscale via drawImage. Looping at
 * display resolution would be ~100x more work for identical output.
 */
export class Heatmap {
  readonly nx: number;
  readonly ny: number;

  /** offscreen canvas is EXACTLY nx x ny — one grid cell per pixel */
  private readonly buf: HTMLCanvasElement;
  private readonly bufCtx: CanvasRenderingContext2D;
  private readonly img: ImageData;

  /** min/max of the most recent draw — display these, don't trust your eyes */
  lastMin = 0;
  lastMax = 0;

  constructor(nx: number, ny: number) {
    this.nx = nx;
    this.ny = ny;

    this.buf = document.createElement('canvas');
    this.buf.width = nx;
    this.buf.height = ny;

    const ctx = this.buf.getContext('2d', { willReadFrequently: false });
    if (!ctx) throw new Error('2d context unavailable');
    this.bufCtx = ctx;

    this.img = ctx.createImageData(nx, ny);
    // Alpha never changes, so set it once here instead of every frame.
    const d = this.img.data;
    for (let i = 3; i < d.length; i += 4) d[i] = 255;
  }

  /**
   * Fill `this.img.data` from `field`, then blit scaled into `dest`.
   */
  draw(field: Float64Array, dest: CanvasRenderingContext2D, opts: DrawOptions): void {
    const { lo, hi } = computeRange(field, opts.normalization);
    this.lastMin = lo;
    this.lastMax = hi;

    const isSameHiLo = hi - lo < 1e-9;

    // Paint secondary canvas
    for (let j = 0; j < this.ny; j++) {
      const row = this.ny - 1 - j;
      for (let i = 0; i < this.nx; i++) {
        const t = !isSameHiLo ? (field[i + j * this.nx] - lo) / (hi - lo) : 0.5;
        opts.colormap(t, this.img.data, (i + row * this.nx) * 4);
      }
    }

    // Blit
    this.bufCtx.putImageData(this.img, 0, 0);
    dest.imageSmoothingEnabled = false; // crisp cells for debugging
    dest.drawImage(this.buf, 0, 0, dest.canvas.width, dest.canvas.height);
  }
}
