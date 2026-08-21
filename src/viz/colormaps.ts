/**
 * Colormaps write RGB directly into an RGBA byte buffer — no per-pixel
 * allocation, since this runs nx*ny times per frame. Alpha is the caller's job.
 *
 * `t` is expected in [0,1]; values outside are clamped.
 */
export type Colormap = (t: number, out: Uint8ClampedArray, i: number) => void;

type Anchor = readonly [pos: number, r: number, g: number, b: number];

/** Piecewise-linear interpolation over evenly-or-unevenly spaced anchors. */
function rampFromAnchors(anchors: readonly Anchor[]): Colormap {
  return (t, out, i) => {
    if (!(t > 0))
      t = 0; // also catches NaN
    else if (t > 1) t = 1;

    let k = 1;
    while (k < anchors.length - 1 && t > anchors[k][0]) k++;
    const a = anchors[k - 1];
    const b = anchors[k];
    const span = b[0] - a[0];
    const f = span > 0 ? (t - a[0]) / span : 0;

    out[i] = a[1] + f * (b[1] - a[1]);
    out[i + 1] = a[2] + f * (b[2] - a[2]);
    out[i + 2] = a[3] + f * (b[3] - a[3]);
  };
}

/**
 * Perceptually uniform, colorblind-safe. Use for MAGNITUDES: speed, dye,
 * temperature, |vorticity| — anything where zero is one end of the range.
 */
export const viridis = rampFromAnchors([
  [0.0, 68, 1, 84],
  [0.125, 72, 40, 120],
  [0.25, 62, 74, 137],
  [0.375, 49, 104, 142],
  [0.5, 38, 130, 142],
  [0.625, 31, 158, 137],
  [0.75, 53, 183, 121],
  [0.875, 109, 205, 89],
  [1.0, 253, 231, 37],
]);

/**
 * Diverging, neutral grey at t=0.5. Use for SIGNED fields: pressure,
 * divergence, vorticity, a single velocity component — with a symmetric
 * range so that t=0.5 means exactly zero. This is the map that makes a
 * correctly-projected divergence field read as flat grey.
 */
export const coolwarm = rampFromAnchors([
  [0.0, 59, 76, 192],
  [0.25, 122, 157, 238],
  [0.5, 221, 221, 221],
  [0.75, 238, 139, 110],
  [1.0, 180, 4, 38],
]);

/** Greyscale, for when you want to see raw structure without hue bias. */
export const grey = rampFromAnchors([
  [0.0, 0, 0, 0],
  [1.0, 255, 255, 255],
]);
