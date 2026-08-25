// Red-black SOR sweep of the pressure Poisson equation — the GPU counterpart
// of gaussSeidelSweep() in src/core/pressure.ts. Same equation, same boundary
// rules, different traversal order. The reasoning, the measurements and the
// accuracy trade are in docs/WEBGPU.md; this is the short version.
//
// The CPU sweep is fast *because* it reads partially-updated neighbours: cell
// (i,j) uses the already-new p[i-1,j] from earlier in the same sweep. That is
// a serial dependency chain across the whole grid, and it means nothing on a
// GPU where thousands of threads run in an order nobody controls.
//
// Colouring the grid like a checkerboard fixes this exactly. In the 5-point
// stencil a cell's four neighbours are ALWAYS the opposite colour:
//
//        B         (i, j+1)          colour = (i + j) & 1
//     B  R  B      (i-1,j) (i,j) (i+1,j)
//        B         (i, j-1)
//
// So a pass over the red cells reads only black values and writes only red
// ones: race-free, and still genuinely in-place Gauss-Seidel rather than
// Jacobi. Two passes (red, then black) make one full sweep, and the black
// pass sees the brand-new red values — that is where the Gauss-Seidel
// speed-up comes from.
//
// It is not free. Losing the sweep ordering also loses its long-range
// information transport, so red-black needs a LOWER omega than the classical
// optimum under a short sweep budget (docs/WEBGPU.md, "The omega trap").

struct Params {
  nx: u32,
  ny: u32,
  // rho * h^2 / dt. Turns a divergence into a pressure; see pressure.ts.
  scale: f32,
  // SOR relaxation. 1 = plain Gauss-Seidel; 1.6 here — see main.ts.
  omega: f32,
  // Which half of the checkerboard this dispatch updates: 0 or 1.
  color: u32,
  // Uniform structs must be a multiple of 16 bytes. Explicit padding rather
  // than trusting the layout rules to round up silently.
  pad0: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// read_write, and a SINGLE buffer — no ping-pong. That is the whole point of
// the colouring: Rule 2's "never read the array you write" is suspended here
// exactly as it is for the CPU sweep, and for the same reason.
@group(0) @binding(1) var<storage, read_write> p: array<f32>;
@group(0) @binding(2) var<storage, read> divergence: array<f32>;
// Uint8Array has no WGSL equivalent; the host widens `label` to u32 on upload.
@group(0) @binding(3) var<storage, read> label: array<u32>;

const FLUID: u32 = 0u;
const SOLID: u32 = 2u;

// Mirrors grid.ts's isSolidOrOutside(): off-grid counts as solid, which is how
// the closed box gets its walls without a ghost ring. The bounds test must
// come first — the flat index wraps at row boundaries, so an out-of-range
// (i,j) would silently alias a real cell instead of failing.
fn solidOrOutside(i: i32, j: i32) -> bool {
  if (i < 0 || j < 0 || i >= i32(params.nx) || j >= i32(params.ny)) {
    return true;
  }
  return label[u32(i) + u32(j) * params.nx] == SOLID;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let j = gid.y;
  // The dispatch is rounded up to whole 8x8 workgroups, so the last row and
  // column of workgroups run threads that are off the grid. Every WGSL kernel
  // needs this guard; there is no partial workgroup.
  if (i >= params.nx || j >= params.ny) {
    return;
  }
  // Half the threads in every workgroup exit here. That looks wasteful and is
  // the simplest correct thing: the alternative — dispatching nx/2 threads per
  // row and mapping thread t to i = 2*t + ((j ^ color) & 1) — halves the thread
  // count but makes every memory access stride-2, which costs more in
  // coalescing than the idle lanes cost in occupancy. Worth measuring later,
  // not worth guessing at now.
  if (((i + j) & 1u) != params.color) {
    return;
  }

  let k = i + j * params.nx;
  // Only FLUID cells get an equation. Air cells ARE the Dirichlet condition
  // p = 0, and never writing them is what pins their value.
  if (label[k] != FLUID) {
    return;
  }

  let ii = i32(i);
  let jj = i32(j);
  var count = 0.0;
  var sum = 0.0;

  // No Air branch, deliberately: an Air neighbour is non-solid so it counts
  // toward the diagonal, and its stored p is 0 so it adds nothing to the sum.
  // That is exactly Bridson's row. The invariant it rests on — p is 0 in every
  // Air cell — is established by commitLabels() and never disturbed.
  if (!solidOrOutside(ii + 1, jj)) { count += 1.0; sum += p[k + 1u]; }
  if (!solidOrOutside(ii - 1, jj)) { count += 1.0; sum += p[k - 1u]; }
  if (!solidOrOutside(ii, jj + 1)) { count += 1.0; sum += p[k + params.nx]; }
  if (!solidOrOutside(ii, jj - 1)) { count += 1.0; sum += p[k - params.nx]; }

  // A fluid cell walled in on all four sides has no equation to solve.
  if (count == 0.0) {
    return;
  }

  let pGS = (sum - params.scale * divergence[k]) / count;
  p[k] = (1.0 - params.omega) * p[k] + params.omega * pGS;
}
