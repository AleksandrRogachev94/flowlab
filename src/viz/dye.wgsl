// The dye view, drawn from the device buffer the fused step leaves it in.
//
// The 2D path this replaces read the dye back to the host every frame, looped
// over every cell in JavaScript to fill an ImageData, and let drawImage upscale
// it. At 1.3M cells that is ~9 MB down the bus and ~5M byte writes on the main
// thread, per frame, to show a picture the device was already holding.
//
// One oversized triangle, one fragment shader, one read-only binding on
// stepGpu's resident dyeIn. No colormap and no normalization: dye is seeded at
// 1 and only dissipates, so the absolute brightness IS the measurement (the
// same argument Heatmap.drawRGB makes).

struct Params {
  /** The DYE grid — this shader knows nothing about the velocity grid. */
  nx: u32,
  ny: u32,
  /** The canvas backing store, in device pixels. */
  w: u32,
  h: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Three planes of nx*ny, back to back — core/grid.ts's DYE_CHANNELS laid out
// exactly as advect.wgsl writes them.
@group(0) @binding(1) var<storage, read> dye: array<f32>;

/**
 * The clip-space triangle trick: three vertices, no vertex buffer, no index
 * buffer. It covers [-1,1]^2 with a single primitive, so there is no diagonal
 * seam for the rasterizer to double-shade.
 */
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(3.0, 1.0),
  );
  return vec4<f32>(p[i], 0.0, 1.0);
}

/** Bilinear over one channel plane, at cell CENTRES — the same convention
 *  grid.ts's sampleP uses, and the same smoothing the 2D blit used to get for
 *  free from imageSmoothingEnabled. At a dye grid near display resolution it
 *  is close to a no-op; below that it is what keeps the picture from turning
 *  into visible cells. */
fn sample(gx: f32, gy: f32, base: u32) -> f32 {
  let i0 = min(u32(gx), params.nx - 2u);
  let j0 = min(u32(gy), params.ny - 2u);
  let tx = gx - f32(i0);
  let ty = gy - f32(j0);
  let k = base + i0 + j0 * params.nx;
  let lo = mix(dye[k], dye[k + 1u], tx);
  let hi = mix(dye[k + params.nx], dye[k + params.nx + 1u], tx);
  return mix(lo, hi, ty);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  // @builtin(position) is in framebuffer pixels with the origin at the TOP
  // left; grid row 0 is the BOTTOM. Same flip as Heatmap.draw.
  let fx = pos.x / f32(params.w);
  let fy = 1.0 - pos.y / f32(params.h);
  // -0.5 puts the sample on cell centres; the clamp stops the edge half-cell
  // extrapolating, which is grid.ts's clampedAxis rule.
  let gx = clamp(fx * f32(params.nx) - 0.5, 0.0, f32(params.nx - 1u));
  let gy = clamp(fy * f32(params.ny) - 0.5, 0.0, f32(params.ny - 1u));
  let cells = params.nx * params.ny;
  return vec4<f32>(
    sample(gx, gy, 0u),
    sample(gx, gy, cells),
    sample(gx, gy, 2u * cells),
    1.0,
  );
}
