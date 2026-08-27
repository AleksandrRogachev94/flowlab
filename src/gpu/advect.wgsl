// Semi-Lagrangian advection on the GPU — the counterpart of src/core/advect.ts.
// Same RK2 backtrace, same MacCormack correction, same boundary rules; see
// docs/WEBGPU.md §8 for the design and the numbers.
//
// One shader, six entry points, ONE binding layout. Every pass is the same
// shape — read a carrier velocity, read a source field, write one output —
// so the bindings never change and only the entry point and the bind group
// do. The four roles:
//
//   carU/carV  the CARRIER: the velocity every backtrace follows. Always u^n
//              for velocity advection, the projected u for dye.
//   src        the field being SAMPLED at the backtraced point.
//   orig       the field before this step. Only the correct_* passes read it
//              (for u^n[k] and for the limiter's stencil); the forward passes
//              bind it to the same buffer as `src` and ignore it.
//   dst        the output. Never aliases any of the above — that is what
//              keeps this a legal one-thread-per-cell gather (Rule 2).
//
// MacCormack is TWO dispatches here, not three. The CPU does forward, reverse,
// then correction, but the reverse pass's output is read only at its own index
// by the correction — so the two fuse into one kernel with no intermediate
// buffer. Same arithmetic, one less pass over memory, and this kernel is
// memory-bound.

// TWO grids, and only the dye kernels can tell the difference.
//
// nx/ny/h describe the VELOCITY grid — the carrier, the labels, and the u/v
// fields themselves. dnx/dny/dh describe the DYE grid, which is the same grid
// unless Simulation was built with a dyeScale above 1 (see its `dyeG`).
//
// The split is this cheap because every kernel below already works in WORLD
// coordinates: a thread turns its index into an (x, y) and every sampler turns
// an (x, y) back into indices on ITS OWN field. So carrierU/carrierV need no
// change at all — they were never told which grid the caller sits on.
struct Params {
  nx: u32,
  ny: u32,
  h: f32,
  // Signed: the forward pass traces back by +dt, the fused correction traces
  // both ways from it.
  dt: f32,
  dnx: u32,
  dny: u32,
  dh: f32,
  // std140 rounds a uniform struct up to 16 bytes; naming the slot beats
  // letting the layout do it silently.
  pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Uint8Array has no WGSL equivalent; the host widens `label` to u32.
@group(0) @binding(1) var<storage, read> label: array<u32>;
@group(0) @binding(2) var<storage, read> carU: array<f32>;
@group(0) @binding(3) var<storage, read> carV: array<f32>;
@group(0) @binding(4) var<storage, read> src: array<f32>;
@group(0) @binding(5) var<storage, read> orig: array<f32>;
@group(0) @binding(6) var<storage, read_write> dst: array<f32>;

const SOLID: u32 = 2u;
/** Must match DYE_CHANNELS in core/grid.ts. */
const CHANNELS: u32 = 3u;

// ---------------------------------------------------------------- indexing

fn idxP(i: i32, j: i32) -> u32 { return u32(i) + u32(j) * params.nx; }
/** idxP on the DYE grid. */
fn idxD(i: i32, j: i32) -> u32 { return u32(i) + u32(j) * params.dnx; }
fn idxU(i: i32, j: i32) -> u32 { return u32(i) + u32(j) * (params.nx + 1u); }
fn idxV(i: i32, j: i32) -> u32 { return u32(i) + u32(j) * params.nx; }

/** grid.ts's isSolidOrOutside(): off-grid is wall. Bounds test FIRST — the
 *  flat index wraps at row ends and would alias a real cell. */
fn solidOrOutside(i: i32, j: i32) -> bool {
  if (i < 0 || j < 0 || i >= i32(params.nx) || j >= i32(params.ny)) {
    return true;
  }
  return label[idxP(i, j)] == SOLID;
}

/**
 * The same question asked from a DYE cell. Labels are the solver's geometry
 * and stay on the velocity grid — a finer copy would only be the same
 * staircase drawn with more steps — so the dye cell's CENTRE is mapped across
 * and the coarse cell answers. At dyeScale 1 the ratio is 1 and this reduces
 * to solidOrOutside(i, j) exactly.
 */
fn dyeSolidOrOutside(i: i32, j: i32) -> bool {
  let r = params.dh / params.h;
  return solidOrOutside(
    i32(floor((f32(i) + 0.5) * r)),
    i32(floor((f32(j) + 0.5) * r)),
  );
}

// ------------------------------------------------------------ interpolation

struct Axis {
  i0: i32,
  f: f32,
};

/** grid.ts's clampedAxis(). The two clamps differ on purpose: the POSITION to
 *  count-1 so an outside point interpolates at the edge instead of
 *  extrapolating, the INDEX to count-2 so i0+1 stays in range. */
fn axis(pos: f32, count: i32) -> Axis {
  let c = clamp(pos, 0.0, f32(count - 1));
  // c >= 0 after the clamp, so truncation IS floor.
  let i0 = min(i32(c), count - 2);
  return Axis(i0, c - f32(i0));
}

fn bilerp(v00: f32, v10: f32, v01: f32, v11: f32, fx: f32, fy: f32) -> f32 {
  let lo = v00 + fx * (v10 - v00);
  let hi = v01 + fx * (v11 - v01);
  return lo + fy * (hi - lo);
}

// WGSL cannot take a storage array as a function parameter without an
// extension, so the sampler is written once per (array, staggering) pair the
// kernels actually need. Six short functions beats one clever one that only
// compiles on some drivers.

/** sampleU on carU — u[i,j] sits at (i*h, (j+0.5)*h), hence the -0.5 on y. */
fn carrierU(x: f32, y: f32) -> f32 {
  let a = axis(x / params.h, i32(params.nx) + 1);
  let b = axis(y / params.h - 0.5, i32(params.ny));
  let k = idxU(a.i0, b.i0);
  let s = params.nx + 1u;
  return bilerp(carU[k], carU[k + 1u], carU[k + s], carU[k + s + 1u], a.f, b.f);
}

/** sampleV on carV — v[i,j] sits at ((i+0.5)*h, j*h), so the -0.5 is on x. */
fn carrierV(x: f32, y: f32) -> f32 {
  let a = axis(x / params.h - 0.5, i32(params.nx));
  let b = axis(y / params.h, i32(params.ny) + 1);
  let k = idxV(a.i0, b.i0);
  let s = params.nx;
  return bilerp(carV[k], carV[k + 1u], carV[k + s], carV[k + s + 1u], a.f, b.f);
}

fn srcU(x: f32, y: f32) -> f32 {
  let a = axis(x / params.h, i32(params.nx) + 1);
  let b = axis(y / params.h - 0.5, i32(params.ny));
  let k = idxU(a.i0, b.i0);
  let s = params.nx + 1u;
  return bilerp(src[k], src[k + 1u], src[k + s], src[k + s + 1u], a.f, b.f);
}

fn srcV(x: f32, y: f32) -> f32 {
  let a = axis(x / params.h - 0.5, i32(params.nx));
  let b = axis(y / params.h, i32(params.ny) + 1);
  let k = idxV(a.i0, b.i0);
  let s = params.nx;
  return bilerp(src[k], src[k + 1u], src[k + s], src[k + s + 1u], a.f, b.f);
}

/** sampleP on one dye channel of src. `base` selects the channel: the three
 *  channels share one buffer, laid out end to end. */
fn srcQ(x: f32, y: f32, base: u32) -> f32 {
  let a = axis(x / params.dh - 0.5, i32(params.dnx));
  let b = axis(y / params.dh - 0.5, i32(params.dny));
  let k = base + idxD(a.i0, b.i0);
  let s = params.dnx;
  return bilerp(src[k], src[k + 1u], src[k + s], src[k + s + 1u], a.f, b.f);
}

// ---------------------------------------------------------------- limiter

/** grid.ts's clampToRange: hold q inside the four values the forward pass
 *  blended. Without it the MacCormack correction can invent a new extremum. */
fn clampToRange(q: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
  return clamp(q, min(min(a, b), min(c, d)), max(max(a, b), max(c, d)));
}

fn clampOrigU(x: f32, y: f32, q: f32) -> f32 {
  let a = axis(x / params.h, i32(params.nx) + 1);
  let b = axis(y / params.h - 0.5, i32(params.ny));
  let k = idxU(a.i0, b.i0);
  let s = params.nx + 1u;
  return clampToRange(q, orig[k], orig[k + 1u], orig[k + s], orig[k + s + 1u]);
}

fn clampOrigV(x: f32, y: f32, q: f32) -> f32 {
  let a = axis(x / params.h - 0.5, i32(params.nx));
  let b = axis(y / params.h, i32(params.ny) + 1);
  let k = idxV(a.i0, b.i0);
  let s = params.nx;
  return clampToRange(q, orig[k], orig[k + 1u], orig[k + s], orig[k + s + 1u]);
}

fn clampOrigQ(x: f32, y: f32, q: f32, base: u32) -> f32 {
  let a = axis(x / params.dh - 0.5, i32(params.dnx));
  let b = axis(y / params.dh - 0.5, i32(params.dny));
  let k = base + idxD(a.i0, b.i0);
  let s = params.dnx;
  return clampToRange(q, orig[k], orig[k + 1u], orig[k + s], orig[k + s + 1u]);
}

// --------------------------------------------------------------- backtrace

/** RK2 midpoint: where the fluid now at (x,y) was one dt ago. (u0,v0) is the
 *  carrier AT (x,y) — passed in because a face knows one component exactly.
 *  dt < 0 traces FORWARD, which is all the reverse pass is. */
fn backtrace(x: f32, y: f32, u0: f32, v0: f32, dt: f32) -> vec2<f32> {
  let midX = x - 0.5 * dt * u0;
  let midY = y - 0.5 * dt * v0;
  return vec2<f32>(x - dt * carrierU(midX, midY), y - dt * carrierV(midX, midY));
}

// ------------------------------------------------------------- entry points
//
// Every kernel guards its bounds first: the dispatch is rounded up to whole
// 8x8 workgroups, so the last row and column run threads that are off the
// field. Note the field EXTENTS differ — u is (nx+1) by ny, v is nx by
// (ny+1), dye is nx by ny — which is the one place the MAC staggering leaks
// into the plumbing.
//
// A face or cell touching a solid copies its source through rather than
// skipping: `dst` is a separate buffer and would otherwise keep stale
// ping-pong data. The dye entries are the ones on the dye grid — dnx by dny —
// so they use idxD, dh and dyeSolidOrOutside where the velocity entries use
// the plain ones; everything else about them is unchanged. Bounds match subtractGradient's, so a face advection writes
// is always one projection can correct.

@compute @workgroup_size(8, 8)
fn advect_u(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x > params.nx || gid.y >= params.ny) { return; }
  let i = i32(gid.x);
  let j = i32(gid.y);
  let k = idxU(i, j);
  if (solidOrOutside(i - 1, j) || solidOrOutside(i, j)) {
    dst[k] = src[k];
    return;
  }
  let x = f32(i) * params.h;
  let y = (f32(j) + 0.5) * params.h;
  // carU[k] is exact here: this thread sits ON a u face.
  let b = backtrace(x, y, carU[k], carrierV(x, y), params.dt);
  dst[k] = srcU(b.x, b.y);
}

@compute @workgroup_size(8, 8)
fn advect_v(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.nx || gid.y > params.ny) { return; }
  let i = i32(gid.x);
  let j = i32(gid.y);
  let k = idxV(i, j);
  if (solidOrOutside(i, j - 1) || solidOrOutside(i, j)) {
    dst[k] = src[k];
    return;
  }
  let x = (f32(i) + 0.5) * params.h;
  let y = f32(j) * params.h;
  let b = backtrace(x, y, carrierU(x, y), carV[k], params.dt);
  dst[k] = srcV(b.x, b.y);
}

/** All three dye channels in ONE dispatch. The backtrace is the expensive
 *  part and it does not depend on the channel, so doing it once and gathering
 *  three values is a third of the CPU's work, not the same work three times. */
@compute @workgroup_size(8, 8)
fn advect_dye(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.dnx || gid.y >= params.dny) { return; }
  let i = i32(gid.x);
  let j = i32(gid.y);
  let k = idxD(i, j);
  let cells = params.dnx * params.dny;
  if (dyeSolidOrOutside(i, j)) {
    for (var c = 0u; c < CHANNELS; c++) {
      dst[k + c * cells] = src[k + c * cells];
    }
    return;
  }
  let x = (f32(i) + 0.5) * params.dh;
  let y = (f32(j) + 0.5) * params.dh;
  // A cell centre stores NEITHER component, so both are interpolated.
  let b = backtrace(x, y, carrierU(x, y), carrierV(x, y), params.dt);
  for (var c = 0u; c < CHANNELS; c++) {
    dst[k + c * cells] = srcQ(b.x, b.y, c * cells);
  }
}

// The MacCormack correction, with the reverse pass fused in (see the header):
//
//   qBar      = A_rev(qHat)             traced FORWARD from this cell
//   corrected = qHat + (q^n - qBar)/2   the error estimate, halved
//   dst       = clamp(corrected)        to the forward pass's donor stencil
//
// `src` is qHat, `orig` is q^n. Both backtraces start from the same (u0,v0),
// so the carrier is sampled once per direction and nothing else is repeated.

@compute @workgroup_size(8, 8)
fn correct_u(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x > params.nx || gid.y >= params.ny) { return; }
  let i = i32(gid.x);
  let j = i32(gid.y);
  let k = idxU(i, j);
  if (solidOrOutside(i - 1, j) || solidOrOutside(i, j)) {
    dst[k] = orig[k];
    return;
  }
  let x = f32(i) * params.h;
  let y = (f32(j) + 0.5) * params.h;
  let u0 = carU[k];
  let v0 = carrierV(x, y);
  let rev = backtrace(x, y, u0, v0, -params.dt);
  let corrected = src[k] + 0.5 * (orig[k] - srcU(rev.x, rev.y));
  let fwd = backtrace(x, y, u0, v0, params.dt);
  dst[k] = clampOrigU(fwd.x, fwd.y, corrected);
}

@compute @workgroup_size(8, 8)
fn correct_v(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.nx || gid.y > params.ny) { return; }
  let i = i32(gid.x);
  let j = i32(gid.y);
  let k = idxV(i, j);
  if (solidOrOutside(i, j - 1) || solidOrOutside(i, j)) {
    dst[k] = orig[k];
    return;
  }
  let x = (f32(i) + 0.5) * params.h;
  let y = f32(j) * params.h;
  let u0 = carrierU(x, y);
  let v0 = carV[k];
  let rev = backtrace(x, y, u0, v0, -params.dt);
  let corrected = src[k] + 0.5 * (orig[k] - srcV(rev.x, rev.y));
  let fwd = backtrace(x, y, u0, v0, params.dt);
  dst[k] = clampOrigV(fwd.x, fwd.y, corrected);
}

@compute @workgroup_size(8, 8)
fn correct_dye(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.dnx || gid.y >= params.dny) { return; }
  let i = i32(gid.x);
  let j = i32(gid.y);
  let k = idxD(i, j);
  let cells = params.dnx * params.dny;
  if (dyeSolidOrOutside(i, j)) {
    for (var c = 0u; c < CHANNELS; c++) {
      dst[k + c * cells] = orig[k + c * cells];
    }
    return;
  }
  let x = (f32(i) + 0.5) * params.dh;
  let y = (f32(j) + 0.5) * params.dh;
  let u0 = carrierU(x, y);
  let v0 = carrierV(x, y);
  let rev = backtrace(x, y, u0, v0, -params.dt);
  let fwd = backtrace(x, y, u0, v0, params.dt);
  for (var c = 0u; c < CHANNELS; c++) {
    let base = c * cells;
    let corrected = src[k + base] + 0.5 * (orig[k + base] - srcQ(rev.x, rev.y, base));
    dst[k + base] = clampOrigQ(fwd.x, fwd.y, corrected, base);
  }
}
