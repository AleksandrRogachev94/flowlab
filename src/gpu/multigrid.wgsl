// The three grid-transfer kernels of the multigrid V-cycle, plus label
// coarsening. The SMOOTHER is deliberately absent: it is redBlack.wgsl,
// unchanged, driven with scale = -1 so that (sum - scale*div)/count becomes
// (sum + b)/count — the correction equation. The algorithm and its CPU mirror
// live in core/pressureMultigrid.ts; read that header first. This file is the
// same operators in WGSL, kernel by kernel.
//
// "fine" below always means level l, "coarse" level l+1. One uniform struct
// serves every kernel: dims of both levels, bound per level pair. Each entry
// point declares only the buffers it touches, so the four auto layouts stay
// small and a bind group per (kernel, level) wires them up host-side.

struct Params {
  // Fine (level l) dimensions.
  nx: u32,
  ny: u32,
  // Coarse (level l+1) dimensions.
  cnx: u32,
  cny: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Level l: solution (p on the finest level, correction e below), rhs b,
// residual r, labels.
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> r: array<f32>;
@group(0) @binding(4) var<storage, read> label: array<u32>;
// Level l+1: the restrict target and the prolong source.
@group(0) @binding(5) var<storage, read> eCoarse: array<f32>;
@group(0) @binding(6) var<storage, read_write> bCoarse: array<f32>;
@group(0) @binding(7) var<storage, read_write> xCoarse: array<f32>;
@group(0) @binding(8) var<storage, read_write> labelCoarse: array<u32>;

const FLUID: u32 = 0u;
const AIR: u32 = 1u;
const SOLID: u32 = 2u;

// Same as redBlack.wgsl — bounds first, because the flat index wraps.
fn solidOrOutside(i: i32, j: i32) -> bool {
  if (i < 0 || j < 0 || i >= i32(params.nx) || j >= i32(params.ny)) {
    return true;
  }
  return label[u32(i) + u32(j) * params.nx] == SOLID;
}

// r = b - (count*x - sum): the residual of the level's equation, and 0 in
// every non-fluid or walled-in cell — restriction sums r blindly, so these
// zeros are what keep non-fluid children out of the coarse RHS.
@compute @workgroup_size(8, 8)
fn residual(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let j = gid.y;
  if (i >= params.nx || j >= params.ny) {
    return;
  }
  let k = i + j * params.nx;
  r[k] = 0.0;
  if (label[k] != FLUID) {
    return;
  }
  let ii = i32(i);
  let jj = i32(j);
  var count = 0.0;
  var sum = 0.0;
  if (!solidOrOutside(ii + 1, jj)) { count += 1.0; sum += x[k + 1u]; }
  if (!solidOrOutside(ii - 1, jj)) { count += 1.0; sum += x[k - 1u]; }
  if (!solidOrOutside(ii, jj + 1)) { count += 1.0; sum += x[k + params.nx]; }
  if (!solidOrOutside(ii, jj - 1)) { count += 1.0; sum += x[k - params.nx]; }
  if (count == 0.0) {
    return;
  }
  r[k] = b[k] - (count * x[k] - sum);
}

// Coarse rhs = SUM of the four children's residuals (the h^2-scaled form makes
// full weighting a plain sum — pressureMultigrid.ts header), and the coarse
// correction zeroed in the same pass: its initial guess is always 0, since it
// estimates an error that was just measured. Dispatched over COARSE cells.
@compute @workgroup_size(8, 8)
fn restrictResidual(@builtin(global_invocation_id) gid: vec3<u32>) {
  let I = gid.x;
  let J = gid.y;
  if (I >= params.cnx || J >= params.cny) {
    return;
  }
  var sum = 0.0;
  for (var dj = 0u; dj < 2u; dj++) {
    for (var di = 0u; di < 2u; di++) {
      let i = 2u * I + di;
      let j = 2u * J + dj;
      if (i < params.nx && j < params.ny) {
        sum += r[i + j * params.nx];
      }
    }
  }
  let K = I + J * params.cnx;
  bCoarse[K] = sum;
  xCoarse[K] = 0.0;
}

// Nearest coarse neighbour on one axis: a fine cell's centre sits a
// quarter-cell off its parent's, toward +1 for odd indices and -1 for even.
// Clamping at the domain edge extends e constantly outward — the discrete
// Neumann condition, matching what the walls impose on p itself.
fn sideNeighbour(idx: u32, fineIdx: u32, count: u32) -> u32 {
  if ((fineIdx & 1u) == 1u) {
    return min(idx + 1u, count - 1u);
  }
  if (idx == 0u) {
    return 0u;
  }
  return idx - 1u;
}

// x += bilinear(e), fluid cells only: weights 9/16, 3/16, 3/16, 1/16 from the
// quarter-cell offset above. Non-fluid coarse cells contribute their stored
// e = 0 (Air: the correct Dirichlet value; Solid: restrict zeroed it), which
// dilutes the correction beside a wall instead of special-casing it.
@compute @workgroup_size(8, 8)
fn prolong(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let j = gid.y;
  if (i >= params.nx || j >= params.ny) {
    return;
  }
  let k = i + j * params.nx;
  if (label[k] != FLUID) {
    return;
  }
  let I = i / 2u;
  let J = j / 2u;
  let I2 = sideNeighbour(I, i, params.cnx);
  let J2 = sideNeighbour(J, j, params.cny);
  x[k] += 0.5625 * eCoarse[I + J * params.cnx]
        + 0.1875 * (eCoarse[I2 + J * params.cnx] + eCoarse[I + J2 * params.cnx])
        + 0.0625 * eCoarse[I2 + J2 * params.cnx];
}

// One coarse label from up to four fine ones, priority Air > Fluid > Solid —
// the reasoning (and the convergence cost of getting it wrong either way) is
// on coarsenLabels() in pressureMultigrid.ts. Dispatched over COARSE cells,
// once per level pair per solve, before any cycle runs.
@compute @workgroup_size(8, 8)
fn coarsenLabels(@builtin(global_invocation_id) gid: vec3<u32>) {
  let I = gid.x;
  let J = gid.y;
  if (I >= params.cnx || J >= params.cny) {
    return;
  }
  var air = false;
  var fluid = false;
  for (var dj = 0u; dj < 2u; dj++) {
    for (var di = 0u; di < 2u; di++) {
      let i = 2u * I + di;
      let j = 2u * J + dj;
      if (i < params.nx && j < params.ny) {
        let l = label[i + j * params.nx];
        air = air || l == AIR;
        fluid = fluid || l == FLUID;
      }
    }
  }
  var out = SOLID;
  if (air) {
    out = AIR;
  } else if (fluid) {
    out = FLUID;
  }
  labelCoarse[I + J * params.cnx] = out;
}
