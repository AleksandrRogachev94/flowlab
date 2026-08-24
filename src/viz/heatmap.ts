import type { FieldArray } from '../core/grid.ts';
import type { Colormap } from './colormaps.ts';

export type Normalization =
  /** map [min,max] of this frame's data onto [0,1] */
  | { kind: 'auto' }
  /** map [-m,m] onto [0,1] where m = max|value|, so 0 lands exactly at 0.5 */
  | { kind: 'symmetric' }
  /**
   * Like 'symmetric', but m is the p-th percentile of |value| rather than the
   * max, and anything past it clamps to the end of the ramp.
   *
   * For a field whose extremes live in a few cells, the max is a terrible
   * scale: on the karman wake, max|curl| sits on the cylinder's staircase
   * corners at ~8x the shed vortices, which would render the whole street in
   * the middle eighth of the ramp. The percentile scales to the wake and
   * clamps only the surface spike — a discretization artifact, not the subject.
   */
  | { kind: 'percentile'; p: number }
  /** fixed range, so colors are comparable across frames */
  | { kind: 'fixed'; min: number; max: number };

export interface DrawOptions {
  colormap: Colormap;
  normalization: Normalization;
  /**
   * Bilinear-upscale the grid-resolution buffer (GPU-side, free). Smooth
   * gradients for presentation; leave off to see exact cell boundaries when
   * debugging.
   */
  smooth?: boolean;
}

/**
 * Resolve a Normalization choice against actual data to concrete [lo, hi]
 * bounds. Pulled out of Heatmap.draw() as a pure function (no canvas) so it
 * can be unit tested directly with node --test.
 */
export function computeRange(
  field: FieldArray,
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
    case 'percentile': {
      // Histogram rather than a sort: one extra O(n) pass instead of
      // O(n log n) plus an allocation, and BINS resolution on the threshold is
      // far finer than the colour ramp can show anyway.
      let max = 0;
      for (let i = 0; i < field.length; i++) {
        const a = Math.abs(field[i]);
        if (a > max) max = a;
      }
      if (max === 0) return { lo: 0, hi: 0 };

      const BINS = 512;
      const counts = new Int32Array(BINS);
      for (let i = 0; i < field.length; i++) {
        const b = Math.min(BINS - 1, Math.floor((Math.abs(field[i]) / max) * BINS));
        counts[b] += 1;
      }

      const target = normalization.p * field.length;
      let seen = 0;
      for (let b = 0; b < BINS; b++) {
        seen += counts[b];
        // Upper edge of the bin the percentile falls in, so m is never 0 when
        // the percentile lands in the first bin.
        if (seen >= target) return { lo: (-(b + 1) / BINS) * max, hi: ((b + 1) / BINS) * max };
      }
      return { lo: -max, hi: max };
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
  draw(field: FieldArray, dest: CanvasRenderingContext2D, opts: DrawOptions): void {
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

    this.blit(dest, opts.smooth);
  }

  /**
   * Composite three fields as the R, G and B channels of one image, with no
   * colormap involved: for independent dyes the MIXING is the signal, and any
   * single-channel ramp would have to discard it to produce one number.
   *
   * Values are read as concentrations on [0,1] against a black background.
   * Deliberately NOT normalized, unlike draw(): dye only dissipates, so
   * rescaling per frame would hide the decay. Out-of-range values are clamped
   * (and rounded) for free by Uint8ClampedArray.
   */
  drawRGB(
    r: FieldArray,
    g: FieldArray,
    b: FieldArray,
    dest: CanvasRenderingContext2D,
    opts: { smooth?: boolean } = {},
  ): void {
    // The scale is fixed, so report it rather than leaving whatever the last
    // draw() happened to measure.
    this.lastMin = 0;
    this.lastMax = 1;

    const d = this.img.data;
    for (let j = 0; j < this.ny; j++) {
      const row = this.ny - 1 - j;
      for (let i = 0; i < this.nx; i++) {
        const k = i + j * this.nx;
        const o = (i + row * this.nx) * 4;
        d[o] = r[k] * 255;
        d[o + 1] = g[k] * 255;
        d[o + 2] = b[k] * 255;
      }
    }

    this.blit(dest, opts.smooth);
  }

  /** Push the grid-resolution buffer to the display canvas, scaled up. */
  private blit(dest: CanvasRenderingContext2D, smooth?: boolean): void {
    this.bufCtx.putImageData(this.img, 0, 0);
    dest.imageSmoothingEnabled = smooth ?? false;
    dest.drawImage(this.buf, 0, 0, dest.canvas.width, dest.canvas.height);
  }
}
