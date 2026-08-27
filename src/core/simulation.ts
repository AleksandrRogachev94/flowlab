import type { AdvectionScheme } from './advect.ts';
import { CpuAdvector, type Advector } from './advector.ts';
import { applyOutflow, commitLabels } from './boundaries.ts';
import { computeDivergence } from './divergence.ts';
import {
  Cell,
  createFields,
  createGrid,
  type FieldArray,
  type FieldCtor,
  type Fields,
  type Grid,
} from './grid.ts';
import { applyDyePatch, type DyePatch } from './dye.ts';
import { Profiler } from './profiler.ts';
import { cpuPressureSolver, type PressureSolver } from './pressureSolver.ts';
import { subtractGradient } from './subtractGradient.ts';

/**
 * The fused-step seam: everything from velocity advection through dye
 * advection as ONE device submit, replacing the advector + solver + host
 * kernels for that stretch of step(). An interface here (like PressureSolver)
 * so core/ keeps not depending on gpu/; the implementation is
 * gpu/stepGpu.ts, and its comment explains what stays resident.
 *
 * Returns solver iterations (V-cycles), like PressureSolver.solve. On return,
 * f.u / f.v / f.dye hold the step's results; f.p is NOT updated — the
 * pressure lives on the device and warm-starts there. That staleness is safe
 * (nothing on the host reads p between solves; a solver switched in later
 * merely warm-starts from an old field) but it is a contract worth knowing.
 */
export interface GpuStep {
  step(
    f: Fields,
    scheme: AdvectionScheme,
    dt: number,
    scale: number,
    gradScale: number,
    /** exp(-dyeDecay * dt), applied on the device — see the call site. 1 is
     *  off, and is what a scene without decay passes. */
    dyeKeep: number,
    perf: Profiler,
  ): Promise<number>;
  /** Host fields changed under it (reset, frames on a CPU solver); the next
   *  step() must re-upload u, v, p, label AND dye before trusting the device
   *  copy. */
  invalidate(): void;
  /** The scene's dye inlet, re-imposed on the device every step. Constant in
   *  time, so this is called once per reset rather than per frame. */
  setDyePatch(patch: DyePatch | null): void;
  /**
   * Copies the device's dye into the host arrays.
   *
   * The fused path does NOT do this per frame — the renderer reads the device
   * buffer directly, which is the whole reason the dye can stay resident — so
   * the host mirror is stale for as long as this path is driving. Anything
   * that needs it (a CPU solver taking over, the 2D dye view) has to ask.
   */
  readDye(dye: FieldArray[]): Promise<void>;
}

/** Writes an initial velocity field. Matches the scene helpers' signature. */
export type Seed = (g: Grid, u: FieldArray, v: FieldArray) => void;

/**
 * Writes an initial dye field. Deliberately separate from Seed rather than one
 * combined `(g, f) => void`: dye is passive, so any tracer pattern is valid
 * with any velocity scene, and keeping the two apart lets them be mixed freely
 * — and lets the numerics tests seed velocity with no dye at all.
 */
export type DyeSeed = (dg: Grid, dye: FieldArray[]) => void;

/**
 * Re-imposed every step: a seed is initial data, a source is a boundary
 * condition on the tracer that advection keeps carrying away.
 *
 * A factory for a DyePatch rather than a closure that writes the field, and
 * the change is what lets the dye live on the device: a patch is a few
 * kilobytes of rectangle that both a host loop and a compute kernel can
 * consume, where a closure could only ever run on the host and so forced the
 * whole dye field up the bus every frame. Every emitter in scenes/ was
 * already a fixed rectangle of prescribed values — see core/dye.ts.
 *
 * Called with BOTH grids: `dg` is where the patch's cells live, `g` is what a
 * thickness quoted "in cells inward from the wall" has always meant, and the
 * two stop being the same grid at dyeScale > 1.
 */
export type DyeSource = (dg: Grid, g: Grid) => DyePatch;

/**
 * Writes the Fluid/Air/Solid layout: obstacles, and open outlets. Runs BEFORE
 * the velocity seed, so a seed can read nothing from it — labels are geometry,
 * and commitLabels() afterwards is what reconciles the two.
 */
export type LabelSeed = (g: Grid, label: Uint8Array) => void;

/**
 * Everything a scene supplies. An object rather than positional arguments
 * because the four parts are independent — any velocity seed composes with any
 * tracer, and the geometry is orthogonal to both — and because the ORDER they
 * are applied in is a correctness detail reset() owns, not the caller.
 */
export interface SceneSpec {
  labels?: LabelSeed;
  seed?: Seed;
  dye?: DyeSeed;
  dyeSource?: DyeSource;
}

export interface SimulationParams {
  /**
   * Target for CFL = u_max * dt / h. Semi-Lagrangian is stable at any dt, so
   * this is purely an accuracy knob, and the curve peaks near 1: measured on
   * Taylor-Green at N=64, energy kept is 50% at CFL 4, 71% at CFL 1, 68% at
   * CFL 0.13. The penalty is far steeper above 1 than below.
   */
  cflTarget: number;
  /**
   * 'semiLagrangian' — one backtrace, heavily dissipative.
   * 'macCormack'     — three backtraces, second-order, far less dissipative.
   * Switchable at runtime (it is read fresh each step), so one Simulation can
   * be flipped mid-run and the two schemes compared on the same state.
   */
  scheme: AdvectionScheme;
  /** Cap, since a nearly-still flow would otherwise ask for an unbounded dt. */
  dtMax: number;
  /** Cap on sweeps; the solve usually exits earlier once `tol` is met. */
  pressureIters: number;
  /**
   * Fraction of the incoming divergence (RMS) left when the solve may stop.
   * Relative, so it needs no retuning when dt, h or flow speed change.
   */
  tol: number;
  /**
   * SOR relaxation, defaulting to the optimum 2/(1+sin(pi/N)). This is
   * GRID-DEPENDENT (1.906 at N=64, 1.976 at N=256) and a value tuned at one
   * resolution converges badly at another: at N=256, omega=1.8 needed 241
   * sweeps per solve against 127 at the optimum.
   */
  omega: number;
  rho: number;
  /**
   * Dye fade, per second of sim time; 0 is off. Half-life is ln2 / dyeDecay.
   * Only matters with a source: a permanent source in a recirculating box
   * saturates every cell eventually, and decay is what turns dye from "has
   * this fluid ever been near the nozzle" into "how recently was it".
   *
   * Independent of N: the step applies exp(-dyeDecay * dt), so elapsed SIM
   * TIME sets the fade, not the number of steps taken to cover it. The scene
   * knob worth thinking in is a distance — dye fades to 1/e after travelling
   * U / dyeDecay, so pick that first and divide.
   */
  dyeDecay: number;
  /**
   * Recompute the post-step residual (`div`) only every Nth step. 1 keeps it
   * exact, which is what the headless reference and the tests want.
   *
   * It is a DIAGNOSTIC — nothing in the step reads it — and at 1920x1080 the
   * scan costs more than the dye advection it follows, so the browser asks for
   * 4: the readout is smoothed and redrawn at frame rate either way, and no
   * decision anywhere is made on a 60 Hz residual that a 15 Hz one would get
   * wrong. It is still computed from what actually came back from the device.
   *
   * 0 means NEVER, which is what the browser sets while the readout that
   * displays it is off — the whole point of a diagnostic nothing reads is that
   * it should not be computed.
   */
  residualEvery: number;
}

/**
 * Classical optimal SOR factor for an nx by ny Poisson problem:
 * omega = 2 / (1 + sqrt(1 - rho^2)), rho = (cos(pi/nx) + cos(pi/ny)) / 2,
 * where rho is the Jacobi iteration's spectral radius.
 *
 * Reduces exactly to the familiar 2 / (1 + sin(pi/n)) when nx === ny, since
 * sqrt(1 - cos^2) = sin. A channel is far from square, and the aspect ratio
 * genuinely moves the optimum — using the short side alone under-relaxes badly.
 */
export function optimalOmega(nx: number, ny: number = nx): number {
  const rho = 0.5 * (Math.cos(Math.PI / nx) + Math.cos(Math.PI / ny));
  return 2 / (1 + Math.sqrt(1 - rho * rho));
}

export const defaultParams: SimulationParams = {
  cflTarget: 2.0, // initially it was 1. Bridson's book mentions 5.
  scheme: 'semiLagrangian',
  dtMax: 1 / 30,
  pressureIters: 100,
  tol: 5e-3, // strict headless-reference default; the browser passes looser (see main.ts)
  omega: 0, // replaced by optimalOmega(nx, ny) in the constructor
  rho: 1.0,
  dyeDecay: 0,
  residualEvery: 1,
};

/**
 * One advect -> project loop over a MAC grid. No DOM (Rule 1), so it runs
 * headlessly and can be instantiated twice to diff two schemes on one seed.
 */
export class Simulation {
  readonly g: Grid;
  /**
   * The grid the dye lives on: `g` refined by the constructor's `dyeScale`.
   *
   * Dye is passive — it is carried by the velocity and feeds nothing back — so
   * it is free to be resolved finer than the field carrying it, and that is
   * the cheapest fidelity there is: the pressure solve is a dozen-odd passes
   * per step and dye advection is one, so refining the tracer alone costs a
   * fraction of what refining the whole simulation would.
   *
   * What it buys is the resolution to HOLD a filament the flow has stretched
   * thin, instead of letting interpolation eat it — and, when the dye grid
   * reaches the display's own resolution, an end to the upscale blur the 2D
   * blit used to add on top. What it does NOT buy is new small-scale motion:
   * the carrier is bilinearly interpolated and so effectively band-limited at
   * `g`, which is why main.ts caps the scale at 2.
   */
  readonly dyeG: Grid;
  readonly f: Fields;
  /** Divergence remaining AFTER the last projection — the solver's residual. */
  readonly div: Float64Array;
  readonly params: SimulationParams;

  time = 0;
  /** Timestep the last step() actually used; 0 before the first step. */
  dt = 0;
  /** Sweeps the last pressure solve used, out of params.pressureIters. */
  iters = 0;
  /** Per-phase timings for the last step(). Always recording — see profiler.ts. */
  readonly perf = new Profiler();
  /**
   * Swappable at runtime, so one Simulation can be flipped between CPU and GPU
   * mid-run and the two compared on identical state — the same trick
   * params.scheme already uses for advection.
   */
  solver: PressureSolver = cpuPressureSolver;
  /** Swappable for the same reason, and on the same terms — see Advector. */
  advector: Advector;
  /**
   * When set, step() hands the whole advect -> project -> dye stretch to it
   * and `advector`/`solver` sit unused — the phase-wise seams above remain the
   * comparison ladder, this is the production path. main.ts sets and clears it
   * as the solver cycles.
   */
  stepper: GpuStep | null = null;

  /**
   * maxFaceSpeed()'s answer for the CURRENT u/v, or null if they have moved
   * since. Three callers want it per frame — step()'s CFL, the view's arrow
   * scale, and the `cfl` readout — and the last two run on the SAME field the
   * next step()'s CFL will read, so caching collapses three full scans of u
   * and v into one. Only step() and reset() write velocity, so those two
   * places are the whole invalidation surface.
   */
  private uMax: number | null = null;

  /** Steps taken, for params.residualEvery's phase. Monotonic across resets —
   *  nothing depends on WHICH steps get a fresh residual, only how often. */
  private steps = 0;

  // Rule 3: advection can't run in place (u self-advects), so these are
  // allocated once and ping-ponged by reference swap, never copied.
  private uNext: FieldArray;
  private vNext: FieldArray;
  private dyeNext: FieldArray[];

  /** Set by reset(), so switching scenes can't leave an emitter running. */
  private dyePatch: DyePatch | null = null;

  /**
   * `h = 1 / ny`, so the domain is always exactly ONE unit tall and nx/ny
   * units wide. Square grids are unaffected (h = 1/n as before) and every
   * existing scene's world coordinates still mean what they did; a channel
   * just extends to the right. Anchoring the unit to the height rather than
   * the width is what keeps a cylinder diameter or a jet band the same
   * physical size when the domain gets longer.
   */
  /**
   * `fieldCtor` picks the fields' precision — the option grid.ts's FieldArray
   * union exists for. Float64Array stays the default so every headless test
   * and benchmark keeps its reference precision; the browser passes
   * Float32Array (see main.ts for the measured why). The ping-pong buffers
   * must share the ctor, since a swap makes them BE the fields; `div` stays
   * f64 — it never swaps, and it is the diagnostic worth keeping exact.
   */
  constructor(
    nx: number,
    ny: number = nx,
    params: Partial<SimulationParams> = {},
    fieldCtor: FieldCtor = Float64Array,
    /** Whole-number refinement of the dye grid — see `dyeG`. 1 is the old
     *  behaviour and what every test and benchmark uses. Whole numbers only,
     *  so the two grids cover exactly the same rectangle with no half-cell to
     *  reconcile at the outflow edge. */
    dyeScale = 1,
  ) {
    this.params = { ...defaultParams, omega: optimalOmega(nx, ny), ...params };
    this.g = createGrid(nx, ny, 1 / ny);
    const s = Math.max(1, Math.round(dyeScale));
    this.dyeG = s === 1 ? this.g : createGrid(nx * s, ny * s, 1 / (ny * s));
    this.f = createFields(this.g, fieldCtor, this.dyeG);
    this.div = new Float64Array(this.f.p.length);
    this.uNext = new fieldCtor(this.f.u.length);
    this.vNext = new fieldCtor(this.f.v.length);
    this.dyeNext = this.f.dye.map(() => new fieldCtor(this.dyeG.nx * this.dyeG.ny));
    this.advector = new CpuAdvector(this.g, this.dyeG);
  }

  /**
   * Back to t = 0. Clears p too (or the warm start begins from a field solving
   * a different problem) and `label` (or switching scenes could leave an
   * obstacle behind). Every part is optional.
   *
   * The order is the point: commitLabels runs AFTER the seeds, so a velocity
   * seed may paint the whole field (`u.fill(U)`) without knowing where the
   * obstacles are — whatever it wrote inside a solid is cleaned up before
   * computeDivergence, which reads faces without consulting labels, could see
   * it. The dye source runs last, applied once at dt = 0, since a boundary
   * condition on the tracer must already hold at t = 0.
   */
  reset(scene: SceneSpec = {}): void {
    const { g, f } = this;
    f.u.fill(0);
    f.v.fill(0);
    f.p.fill(0);
    f.label.fill(Cell.Fluid);
    for (const c of f.dye) c.fill(0);

    scene.labels?.(g, f.label);
    scene.seed?.(g, f.u, f.v);
    scene.dye?.(this.dyeG, f.dye);
    commitLabels(g, f);

    // Built once, not per step: an inlet is a fixed rectangle of prescribed
    // values, and nothing about it depends on t or dt. Applied here at t = 0
    // too, since a boundary condition on the tracer must already hold then.
    this.dyePatch = scene.dyeSource?.(this.dyeG, g) ?? null;
    if (this.dyePatch) applyDyePatch(this.dyeG, f.dye, this.dyePatch);
    this.stepper?.setDyePatch(this.dyePatch);
    this.time = 0;
    this.dt = 0;
    this.uMax = null;
    // Everything above rewrote the host fields behind the device's back.
    this.stepper?.invalidate();
  }

  /**
   * Largest face velocity — what bounds CFL, since the backtrace samples
   * faces. Not the cell-centred speed used for display.
   *
   * Indexed loops, NOT `for (const x of ...)`. On a typed array the iterator
   * protocol is not free and V8 does not always see through it: over the 1.6M
   * faces of a 1024x768 grid, for-of measured 9.9 ms against 0.8 ms indexed —
   * per step, on the CPU, inside the `advect` phase. That was more than the
   * whole GPU advection it sat next to.
   */
  maxFaceSpeed(): number {
    if (this.uMax !== null) return this.uMax;
    const { u, v } = this.f;
    let m = 0;
    for (let k = 0; k < u.length; k++) m = Math.max(m, Math.abs(u[k]));
    for (let k = 0; k < v.length; k++) m = Math.max(m, Math.abs(v[k]));
    this.uMax = m;
    return m;
  }

  /**
   * Async ONLY because of the GPU solver's readback (see PressureSolver). The
   * CPU path never yields to the event loop for real: `await` on a plain
   * number resolves in a microtask, before the next frame or timer can run, so
   * nothing can observe the simulation mid-step.
   */
  async step(): Promise<void> {
    const { g, f, params: p } = this;
    this.perf.begin();
    const uMax = this.maxFaceSpeed();
    // scale/gradScale both carry dt, so they are rebuilt every step; hoisting
    // them would silently use a stale timestep.
    const dt = uMax > 1e-9 ? Math.min(p.dtMax, (p.cflTarget * g.h) / uMax) : p.dtMax;
    const scale = (p.rho * g.h * g.h) / dt;
    const gradScale = dt / (p.rho * g.h);
    // Its own phase because it is not free and it used to hide inside whatever
    // was marked next: maxFaceSpeed scans every face, which at 1920x1080 is
    // ~4.2M reads and the largest host loop left in the frame now that the
    // residual is off by default. A GPU reduction is the fix when it matters.
    this.perf.mark('cfl');

    if (this.stepper) {
      // The fused path: one submit covers everything from here to the dye
      // advection, and the phases collapse into upload / solve / readback.
      // The residual is still computed HERE, on what actually came back —
      // that keeps it an honest check on the device's output, not an on-device
      // number vouching for itself.
      const keep = p.dyeDecay > 0 ? Math.exp(-p.dyeDecay * dt) : 1;
      this.iters = await this.stepper.step(f, p.scheme, dt, scale, gradScale, keep, this.perf);
      if (p.residualEvery > 0 && this.steps % p.residualEvery === 0)
        computeDivergence(g, f.u, f.v, this.div);
      this.perf.mark('residual');
    } else {
      await this.advector.velocity(g, p.scheme, f.u, f.v, this.uNext, this.vNext, f.label, dt);
      [f.u, this.uNext] = [this.uNext, f.u];
      [f.v, this.vNext] = [this.vNext, f.v];
      this.perf.mark('advect');

      computeDivergence(g, f.u, f.v, this.div);
      this.perf.mark('div');

      this.iters = await this.solver.solve(
        g,
        f.p,
        this.div,
        f.label,
        scale,
        p.pressureIters,
        p.omega,
        p.tol,
      );
      this.perf.mark('pressure');

      subtractGradient(g, f.p, f.u, f.v, f.label, gradScale);
      // Before the residual and before dye rides it: this is part of producing
      // the final velocity field, not a diagnostic. No-op on a closed box.
      applyOutflow(g, f.u, f.v, f.label);
      this.perf.mark('gradient');

      // now the residual
      if (p.residualEvery > 0 && this.steps % p.residualEvery === 0)
        computeDivergence(g, f.u, f.v, this.div);
      this.perf.mark('residual');

      // Dye rides the PROJECTED velocity, which is why this sits after the
      // solve instead of alongside advectVelocity: a divergence-free carrier
      // cannot artificially concentrate or thin the tracer. Passive, so
      // nothing here feeds back into u/v and the step is otherwise unaffected
      // by it.
      //
      // Channels are fully independent — same velocity, no coupling — which is
      // what lets the GPU advector carry all three on one backtrace.
      await this.advector.dye(g, p.scheme, f.u, f.v, f.dye, this.dyeNext, f.label, dt, this.dyeG);
      for (let c = 0; c < f.dye.length; c++) {
        [f.dye[c], this.dyeNext[c]] = [this.dyeNext[c], f.dye[c]];
      }
    }
    // u and v have moved; the next maxFaceSpeed() must rescan.
    this.uMax = null;

    // Decay BEFORE the source, so the source region stays exactly at its set
    // value. Exponential in dt, not a fixed factor per step, or the fade rate
    // would change with the timestep.
    //
    // Skipped under the fused stepper: it applies the same factor on the
    // device (project.wgsl's `decay`), which is legitimate because advection
    // is LINEAR in the dye and the factor is a spatial constant — scaling
    // before or after the backtrace gives the same field. 6M multiplies a
    // frame is the difference, and at 1920x1080 that was the single most
    // expensive thing left on the host.
    if (p.dyeDecay > 0 && !this.stepper) {
      const keep = Math.exp(-p.dyeDecay * dt);
      for (const c of f.dye) for (let k = 0; k < c.length; k++) c[k] *= keep;
    }

    // After advection, not before: the source must hold its value at the END
    // of the step, or the inlet is carried off within the same step. Skipped
    // under the fused stepper, which applies the same patch on the device.
    if (this.dyePatch && !this.stepper) applyDyePatch(this.dyeG, f.dye, this.dyePatch);
    this.perf.mark('dye');

    this.dt = dt;
    this.time += dt;
    this.steps++;
  }

  /** CFL actually achieved by the last step — dtMax may have capped it. */
  get cfl(): number {
    return (this.maxFaceSpeed() * this.dt) / this.g.h;
  }
}
