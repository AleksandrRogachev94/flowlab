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

/**
 * Magnitude map for the live view on a dark page — presentation counterpart
 * of viridis, which stays the choice for reading exact values. Stops short of
 * white on purpose: white arrows draw on top, and a ramp reaching 255 hides
 * them exactly where the flow is most interesting.
 */
export const ocean = rampFromAnchors([
  [0.0, 8, 12, 38],
  [0.3, 24, 55, 130],
  [0.6, 45, 110, 205],
  [0.85, 80, 155, 235],
  [1.0, 125, 190, 245],
]);

/**
 * Diverging with a DARK centre: cool negative, near-black zero, warm positive.
 * For signed fields under a white arrow overlay. coolwarm puts near-white at
 * zero, and in a field like vorticity most of the domain IS near zero, so the
 * arrows vanish over exactly the calm regions where they carry the meaning.
 */
export const iceFire = rampFromAnchors([
  [0.0, 130, 225, 255],
  [0.25, 40, 130, 215],
  [0.5, 16, 18, 24],
  [0.75, 225, 105, 45],
  [1.0, 255, 215, 130],
]);

/** Greyscale, for when you want to see raw structure without hue bias. */
export const grey = rampFromAnchors([
  [0.0, 0, 0, 0],
  [1.0, 255, 255, 255],
]);

/* ------------------------------------------------------------ dye palettes */

/**
 * How the three dye channels become a colour.
 *
 * 'rgb'  channel 0/1/2 straight to R/G/B. The default, and the right one
 *        wherever the channels are independent tracers: no colormap and no
 *        normalization, so absolute brightness IS the measurement, and two
 *        dyes interleaved below the cell scale mix to a colour neither was
 *        seeded with — which is numerical diffusion made visible.
 * 'fire' channel 0 is a TEMPERATURE and channel 1 is soot (see
 *        core/buoyancy.ts). Straight RGB would render those as a red flame in
 *        green smoke, which is not a mistake about the colours so much as
 *        about what the field is: once a channel stops being a tracer and
 *        starts being a state variable, the picture has to say what the state
 *        MEANS.
 */
export const DYE_PALETTES = ['rgb', 'fire'] as const;
export type DyePalette = (typeof DYE_PALETTES)[number];

/** The index viz/dye.wgsl switches on. One table, so the shader and the 2D
 *  path cannot disagree about which number is which palette. */
export function paletteCode(p: DyePalette): number {
  return DYE_PALETTES.indexOf(p);
}

/**
 * Blackbody-ish emission plus grey absorption — the 'fire' palette, and the
 * TWIN of the fs() branch in viz/dye.wgsl. The two must stay identical or the
 * same scene looks different on the two engines, which is the same contract
 * core/dye.ts's applyDyePatch has with its kernel.
 *
 * The flame is EMISSIVE and added; the smoke is absorptive and only shows
 * where the flame is not. The rising powers on the three channels are what
 * make one scalar read as a temperature: red appears first and saturates
 * early, green follows, blue only near the top of the range, so the ramp runs
 * dark red -> orange -> yellow -> white in the same order a real flame does.
 * Nothing here is a fit to Planck's law; it is the cheapest curve with that
 * ORDERING, which is the part the eye reads.
 *
 * The coefficients above 1 are deliberate over-range: the core clips to white,
 * which is what stops a plume with a 1.0 source from looking like a flat
 * orange blob.
 *
 * EVEN RED IS SUPERLINEAR, and that is a fix rather than taste. A linear red
 * term means the long cool tail of the temperature field — which is most of
 * the frame, since the heat outlives the flame by design — adds a few percent
 * of red to every smoke cell, and the whole picture comes out dusty pink. At
 * t^2 the tail contributes nothing visible and the flame keeps its edge.
 */
const SMOKE_RGB = [0.9, 0.36, 0.15] as const;

export function fireTone(t: number, s: number, out: Uint8ClampedArray, i: number): void {
  const a = t > 1 ? 1 : t > 0 ? t : 0;
  const t2 = t * t;
  // Uint8ClampedArray does the clamping, so nothing here has to.
  out[i] = 255 * (1.4 * t2 + SMOKE_RGB[0] * s * (1 - a));
  out[i + 1] = 255 * (1.15 * t2 * t + SMOKE_RGB[1] * s * (1 - a));
  out[i + 2] = 255 * (1.25 * t2 * t2 * t + SMOKE_RGB[2] * s * (1 - a));
}
