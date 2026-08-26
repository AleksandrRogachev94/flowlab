// The last three host kernels, ported so the fields can stay resident on the
// device: computeDivergence, subtractGradient and applyOutflow from src/core/.
// Each is a direct transcription — same bounds, same label rules, same
// arithmetic — because docs/WEBGPU.md §8 already established the cost of these
// staying on the host: u and v uploaded twice per step and three stalls per
// frame. The kernels themselves are trivial; being ON the device is the point.
//
// One binding layout for all five entry points, multigrid.wgsl-style fixed
// slots: each pipeline's auto layout contains only the slots its kernel uses.

struct Params {
  nx: u32,
  ny: u32,
  // -(rho * h^2 / dt) / h. divergence() writes the multigrid's rhs DIRECTLY:
  // b = -scale * div, with div = (du + dv) / h, folded into one constant so
  // the fine level needs no separate "scale the rhs" pass and the multigrid's
  // own uniforms stay constant (see multigridGpu.ts).
  divCoef: f32,
  // dt / (rho * h) — subtractGradient's conversion of a pressure difference
  // into a velocity change.
  gradScale: f32,
  // exp(-dyeDecay * dt): the fade the host used to apply in its own loop over
  // every dye cell. 1 when the scene has no decay, and then the host skips the
  // dispatch entirely rather than multiplying 6M floats by one.
  dyeKeep: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> label: array<u32>;
@group(0) @binding(2) var<storage, read> p: array<f32>;
// read_write: the projection is in place, which is legal here for the same
// reason it is on the CPU — each face reads only its OWN u value; neighbours
// come from `p`, a different array (see subtractGradient.ts).
@group(0) @binding(3) var<storage, read_write> u: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;
// The multigrid's fine-level rhs, written by divergence().
@group(0) @binding(5) var<storage, read_write> b: array<f32>;
// The advector's dye INPUT, faded in place by decay() before it is advected.
@group(0) @binding(6) var<storage, read_write> dye: array<f32>;

const AIR: u32 = 1u;
const SOLID: u32 = 2u;
/** Must match DYE_CHANNELS in core/grid.ts, as in advect.wgsl. */
const CHANNELS: u32 = 3u;

fn idxP(i: u32, j: u32) -> u32 { return i + j * params.nx; }
fn idxU(i: u32, j: u32) -> u32 { return i + j * (params.nx + 1u); }
fn idxV(i: u32, j: u32) -> u32 { return i + j * params.nx; }

// grid.ts's isSolidOrOutside(): off-grid is wall; bounds test FIRST because
// the flat index wraps at row ends and would alias a real cell.
fn solidOrOutside(i: i32, j: i32) -> bool {
  if (i < 0 || j < 0 || i >= i32(params.nx) || j >= i32(params.ny)) {
    return true;
  }
  return label[idxP(u32(i), u32(j))] == SOLID;
}

// computeDivergence, fused with the rhs fold (see divCoef above). Reads the
// four faces of its own cell and nothing else, so it needs no labels — the
// invariant it rests on (a face bordering a solid stores the solid's velocity)
// is exactly the CPU version's, kept true by the same advection rules.
@compute @workgroup_size(8, 8)
fn divergence(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.nx || gid.y >= params.ny) { return; }
  let i = gid.x;
  let j = gid.y;
  b[idxP(i, j)] = params.divCoef *
    (u[idxU(i + 1u, j)] - u[idxU(i, j)] + v[idxV(i, j + 1u)] - v[idxV(i, j)]);
}

// subtractGradient, u faces. Interior faces only — i in [1, nx-1] — and a face
// touching a solid keeps its prescribed velocity, which IS the u.n = 0 wall
// condition. Bounds match subtractGradient.ts exactly.
@compute @workgroup_size(8, 8)
fn subtract_u(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x == 0u || gid.x >= params.nx || gid.y >= params.ny) { return; }
  let i = i32(gid.x);
  let j = i32(gid.y);
  if (solidOrOutside(i - 1, j) || solidOrOutside(i, j)) { return; }
  u[idxU(gid.x, gid.y)] -= params.gradScale * (p[idxP(gid.x, gid.y)] - p[idxP(gid.x - 1u, gid.y)]);
}

// subtractGradient, v faces — symmetric, j in [1, ny-1].
@compute @workgroup_size(8, 8)
fn subtract_v(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.nx || gid.y == 0u || gid.y >= params.ny) { return; }
  let i = i32(gid.x);
  let j = i32(gid.y);
  if (solidOrOutside(i, j - 1) || solidOrOutside(i, j)) { return; }
  v[idxV(gid.x, gid.y)] -= params.gradScale * (p[idxP(gid.x, gid.y)] - p[idxP(gid.x, gid.y - 1u)]);
}

// applyOutflow: the no-backflow clamp plus zero-gradient extrapolation on
// every edge Air cell (the why, including the energy-budget argument, is in
// boundaries.ts). A 1D dispatch: thread t owns row t's left/right edges (u)
// and column t's bottom/top edges (v) — all four writes are t's alone, so
// there is nothing to race. Edge work is ~2(nx+ny) cells, three orders below
// the field kernels; this exists for residency, not speed.
@compute @workgroup_size(64)
fn outflow(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  let nx = params.nx;
  let ny = params.ny;
  // Each edge clamps toward ITS OWN outward normal — min on the low edges,
  // max on the high ones — and the clamped value is what gets copied outward.
  if (t < ny) {
    if (label[idxP(0u, t)] == AIR) {
      let out = min(u[idxU(1u, t)], 0.0);
      u[idxU(1u, t)] = out;
      u[idxU(0u, t)] = out;
    }
    if (label[idxP(nx - 1u, t)] == AIR) {
      let out = max(u[idxU(nx - 1u, t)], 0.0);
      u[idxU(nx - 1u, t)] = out;
      u[idxU(nx, t)] = out;
    }
  }
  if (t < nx) {
    if (label[idxP(t, 0u)] == AIR) {
      let out = min(v[idxV(t, 1u)], 0.0);
      v[idxV(t, 1u)] = out;
      v[idxV(t, 0u)] = out;
    }
    if (label[idxP(t, ny - 1u)] == AIR) {
      let out = max(v[idxV(t, ny - 1u)], 0.0);
      v[idxV(t, ny - 1u)] = out;
      v[idxV(t, ny)] = out;
    }
  }
}

// Dye fade, ported off the host. Runs BEFORE the advection rather than after
// it, which the CPU order permits because advection is linear in the dye and
// dyeKeep is a spatial constant: keep * A(q) == A(keep * q), MacCormack's
// correction and its clamp-to-donor-stencil included, since a positive scalar
// commutes with all of it. Doing it here means one buffer, no ping-pong, and
// no dependence on which of dyeA/dyeB the scheme happens to end in.
//
// All three channels in one dispatch, and no labels: a solid's dye fades with
// everything else, exactly as the host loop faded it. gid.y runs over the
// three planes' rows END TO END — the buffer is three nx*ny planes back to
// back, so (x, y) with y in [0, 3*ny) addresses all of it. Deliberately 2D
// and not a flat 1D dispatch: 3*nx*ny/64 is 97k workgroups at 1920x1080, past
// the 65535-per-dimension limit every adapter has.
@compute @workgroup_size(8, 8)
fn decay(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.nx || gid.y >= params.ny * CHANNELS) { return; }
  let k = gid.x + gid.y * params.nx;
  dye[k] = dye[k] * params.dyeKeep;
}
