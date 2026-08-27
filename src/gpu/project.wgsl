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
};

/**
 * The dye kernels' own uniform, separate from Params because the dye no longer
 * lives on the velocity grid: nx/ny here are the DYE grid's (see advect.wgsl's
 * Params for the split), and the inlet rectangle below is in dye cells.
 */
struct DyeParams {
  nx: u32,
  ny: u32,
  // exp(-dyeDecay * dt): the fade the host used to apply in its own loop over
  // every dye cell. 1 when the scene has no decay, and then the host skips the
  // dispatch entirely rather than multiplying 6M floats by one.
  keep: f32,
  // The scene's inlet, uploaded once per reset.
  pi0: u32,
  pj0: u32,
  pnx: u32,
  pny: u32,
  // std140 rounds a uniform struct up to 16 bytes; naming the slot beats
  // letting the layout do it silently.
  pad: u32,
};

/** core/stir.ts's brush, in WORLD units. `h` rides along because Params has no
 *  cell size in it and the brush is the only kernel here that needs one. */
struct StirParams {
  cx: f32,
  cy: f32,
  /** 1 / r^2, so the kernel divides nothing. */
  invR2: f32,
  dx: f32,
  dy: f32,
  h: f32,
  pad0: f32,
  pad1: f32,
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
// The advector's dye INPUT, faded in place by decay() before it is advected —
// and, under the fused stepper, the buffer that IS the dye between frames, so
// the inlet kernel writes here too.
@group(0) @binding(6) var<storage, read_write> dye: array<f32>;
@group(0) @binding(7) var<uniform> dyeParams: DyeParams;
// CHANNELS value planes then ONE coverage plane, each pnx*pny — core/dye.ts's
// DyePatch, laid out exactly as it is on the host. Named `inlet` and not
// `patch` because WGSL reserves that word.
@group(0) @binding(8) var<storage, read> inlet: array<f32>;
@group(0) @binding(9) var<uniform> stirParams: StirParams;

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

// commitLabels' face rule, on the device — see core/boundaries.ts.
//
// It exists for ONE caller: the interactive brush, which paints Solid cells
// into `label` while the fields are resident here. Every other kernel below
// rests on the invariant that a face bordering a solid already stores the
// solid's velocity — divergence() says so explicitly and reads faces without
// consulting labels at all, and subtract_*() leaves such a face alone on the
// grounds that it IS the u.n = 0 wall condition. Both are true as long as
// labels only ever change at reset, when the host's commitLabels runs. Paint a
// solid into moving fluid and they stop being true: those faces keep whatever
// the flow had, and the fluid cell next door reads that as flux through a
// wall. This restores it.
//
// The solid test here is IN-DOMAIN ONLY, unlike solidOrOutside above, and the
// difference is load-bearing rather than pedantic. Treating off-grid as solid
// would zero u[0, j] — the prescribed inlet — on the first stroke, and every
// channel scene would quietly lose its free stream.
fn solidCell(i: i32, j: i32) -> bool {
  if (i < 0 || j < 0 || i >= i32(params.nx) || j >= i32(params.ny)) {
    return false;
  }
  return label[idxP(u32(i), u32(j))] == SOLID;
}

// One dispatch covering both face grids: (nx+1) x (ny+1) threads, each doing
// the u face and the v face it owns where those exist. p needs no pass — the
// smoother skips non-fluid cells and subtract_*() reads p only after checking
// both neighbours are non-solid, so a stale pressure inside a new solid is
// never read by anything.
@compute @workgroup_size(8, 8)
fn commit_labels(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = i32(gid.x);
  let j = i32(gid.y);
  if (gid.x <= params.nx && gid.y < params.ny) {
    if (solidCell(i - 1, j) || solidCell(i, j)) { u[idxU(gid.x, gid.y)] = 0.0; }
  }
  if (gid.x < params.nx && gid.y <= params.ny) {
    if (solidCell(i, j - 1) || solidCell(i, j)) { v[idxV(gid.x, gid.y)] = 0.0; }
  }
}

// core/stir.ts's brush — the mouse dragging through the fluid.
//
// A direct transcription, including the two bounds rules, and both are
// load-bearing: interior faces only, so the channel's prescribed inlet at
// u[0, j] cannot be edited by dragging over it, and nothing touching a solid,
// so the wall invariant the rest of this file rests on survives a stir. They
// are subtract_*()'s bounds exactly, for exactly the same reasons.
//
// Dispatched only on the frames a drag is actually happening, so the
// whole-grid sweep costs nothing the rest of the time — which is why it does
// not bother restricting itself to the brush's bounding box the way the host
// version does. On the host that loop is 2M iterations of real work; here it
// is one cheap kernel that the scheduler runs wide.
@compute @workgroup_size(8, 8)
fn stir(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let j = gid.y;
  let si = i32(i);
  let sj = i32(j);
  // u faces sit at (i*h, (j+0.5)*h) and v faces at ((i+0.5)*h, j*h). Sampling
  // the bump at each face's OWN position, not at its cell centre, is what
  // keeps the brush round on a staggered grid.
  if (i >= 1u && i < params.nx && j < params.ny) {
    if (!solidCell(si - 1, sj) && !solidCell(si, sj)) {
      let d = vec2<f32>(
        f32(i) * stirParams.h - stirParams.cx,
        (f32(j) + 0.5) * stirParams.h - stirParams.cy,
      );
      u[idxU(i, j)] += stirParams.dx * exp(-dot(d, d) * stirParams.invR2);
    }
  }
  if (i < params.nx && j >= 1u && j < params.ny) {
    if (!solidCell(si, sj - 1) && !solidCell(si, sj)) {
      let d = vec2<f32>(
        (f32(i) + 0.5) * stirParams.h - stirParams.cx,
        f32(j) * stirParams.h - stirParams.cy,
      );
      v[idxV(i, j)] += stirParams.dy * exp(-dot(d, d) * stirParams.invR2);
    }
  }
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
  if (gid.x >= dyeParams.nx || gid.y >= dyeParams.ny * CHANNELS) { return; }
  let k = gid.x + gid.y * dyeParams.nx;
  dye[k] = dye[k] * dyeParams.keep;
}

// ---------------------------------------------------------------- dye source
//
// The scene's inlet, ported off the host for the same reason everything else
// here was: the host is where `Fields.dye` lives, and running the source there
// is what forced the whole field up the bus every frame — 25 MB at 1920x1080,
// and four times that once the dye grid is refined. core/dye.ts has the
// DyePatch description this consumes, and the host half, which must stay
// arithmetically identical or the CPU and GPU engines drift at the inlet.

/** applyDyePatch: a rectangle of prescribed dye, lerped in by its coverage
 *  plane so a band emitter can leave the rows outside its band alone. */
@compute @workgroup_size(8, 8)
fn dye_patch(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= dyeParams.pnx || gid.y >= dyeParams.pny) { return; }
  let di = dyeParams.pi0 + gid.x;
  let dj = dyeParams.pj0 + gid.y;
  if (di >= dyeParams.nx || dj >= dyeParams.ny) { return; }
  let pcells = dyeParams.pnx * dyeParams.pny;
  let s = gid.x + gid.y * dyeParams.pnx;
  let cov = inlet[CHANNELS * pcells + s];
  if (cov == 0.0) { return; }
  let cells = dyeParams.nx * dyeParams.ny;
  let k = di + dj * dyeParams.nx;
  for (var c = 0u; c < CHANNELS; c++) {
    let o = k + c * cells;
    dye[o] += cov * (inlet[c * pcells + s] - dye[o]);
  }
}
