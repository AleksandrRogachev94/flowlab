import {
  advectScalar,
  advectScalarMacCormack,
  advectVelocity,
  advectVelocityMacCormack,
  type AdvectionScheme,
} from './advect.ts';
import { applyOutflow, commitLabels } from './boundaries.ts';
import { computeDivergence } from './divergence.ts';
import { Cell, createFields, createGrid, type FieldArray, type Fields, type Grid } from './grid.ts';
import { Profiler } from './profiler.ts';
import { cpuPressureSolver, type PressureSolver } from './pressureSolver.ts';
import { subtractGradient } from './subtractGradient.ts';

/** Writes an initial velocity field. Matches the scene helpers' signature. */
export type Seed = (g: Grid, u: FieldArray, v: FieldArray) => void;

/**
 * Writes an initial dye field. Deliberately separate from Seed rather than one
 * combined `(g, f) => void`: dye is passive, so any tracer pattern is valid
 * with any velocity scene, and keeping the two apart lets them be mixed freely
 * — and lets the numerics tests seed velocity with no dye at all.
 */
export type DyeSeed = (g: Grid, dye: FieldArray[]) => void;

/**
 * Re-imposed every step: a seed is initial data, a source is a boundary
 * condition on the tracer that advection keeps carrying away. `dt` is there
 * for sources that inject at a rate; one holding a region at a fixed value
 * ignores it.
 */
export type DyeSource = (g: Grid, dye: FieldArray[], dt: number) => void;

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
};

/**
 * One advect -> project loop over a MAC grid. No DOM (Rule 1), so it runs
 * headlessly and can be instantiated twice to diff two schemes on one seed.
 */
export class Simulation {
  readonly g: Grid;
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

  // Rule 3: advection can't run in place (u self-advects), so these are
  // allocated once and ping-ponged by reference swap, never copied.
  private uNext: FieldArray;
  private vNext: FieldArray;
  private dyeNext: FieldArray[];
  // MacCormack's forward pass, kept so the correction can compare against it.
  // One dye buffer is enough for all channels: they are advected one at a time.
  private uHat: FieldArray;
  private vHat: FieldArray;
  private dyeHat: FieldArray;

  /** Set by reset(), so switching scenes can't leave an emitter running. */
  private dyeSource: DyeSource | null = null;

  /**
   * `h = 1 / ny`, so the domain is always exactly ONE unit tall and nx/ny
   * units wide. Square grids are unaffected (h = 1/n as before) and every
   * existing scene's world coordinates still mean what they did; a channel
   * just extends to the right. Anchoring the unit to the height rather than
   * the width is what keeps a cylinder diameter or a jet band the same
   * physical size when the domain gets longer.
   */
  constructor(nx: number, ny: number = nx, params: Partial<SimulationParams> = {}) {
    this.params = { ...defaultParams, omega: optimalOmega(nx, ny), ...params };
    this.g = createGrid(nx, ny, 1 / ny);
    this.f = createFields(this.g, Float64Array);
    this.div = new Float64Array(this.f.p.length);
    this.uNext = new Float64Array(this.f.u.length);
    this.vNext = new Float64Array(this.f.v.length);
    this.dyeNext = this.f.dye.map(() => new Float64Array(this.f.p.length));
    this.uHat = new Float64Array(this.f.u.length);
    this.vHat = new Float64Array(this.f.v.length);
    this.dyeHat = new Float64Array(this.f.p.length);
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
    scene.dye?.(g, f.dye);
    commitLabels(g, f);

    this.dyeSource = scene.dyeSource ?? null;
    this.dyeSource?.(g, f.dye, 0);
    this.time = 0;
    this.dt = 0;
  }

  /** Largest face velocity — what bounds CFL, since the backtrace samples
   *  faces. Not the cell-centred speed used for display. */
  maxFaceSpeed(): number {
    let m = 0;
    for (const x of this.f.u) m = Math.max(m, Math.abs(x));
    for (const x of this.f.v) m = Math.max(m, Math.abs(x));
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

    if (p.scheme === 'macCormack') {
      advectVelocityMacCormack(
        g,
        f.u,
        f.v,
        this.uHat,
        this.vHat,
        this.uNext,
        this.vNext,
        f.label,
        dt,
      );
    } else {
      advectVelocity(g, f.u, f.v, this.uNext, this.vNext, f.label, dt);
    }
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

    computeDivergence(g, f.u, f.v, this.div); // now the residual
    this.perf.mark('residual');

    // Dye rides the PROJECTED velocity, which is why this sits after the solve
    // instead of alongside advectVelocity: a divergence-free carrier cannot
    // artificially concentrate or thin the tracer. Passive, so nothing here
    // feeds back into u/v and the step is otherwise unaffected by it.
    //
    // Channels are fully independent — same velocity, no coupling — so this is
    // a plain loop rather than anything vector-valued.
    for (let c = 0; c < f.dye.length; c++) {
      if (p.scheme === 'macCormack') {
        advectScalarMacCormack(g, f.u, f.v, f.dye[c], this.dyeHat, this.dyeNext[c], f.label, dt);
      } else {
        advectScalar(g, f.u, f.v, f.dye[c], this.dyeNext[c], f.label, dt);
      }
      [f.dye[c], this.dyeNext[c]] = [this.dyeNext[c], f.dye[c]];
    }

    // Decay BEFORE the source, so the source region stays exactly at its set
    // value. Exponential in dt, not a fixed factor per step, or the fade rate
    // would change with the timestep.
    if (p.dyeDecay > 0) {
      const keep = Math.exp(-p.dyeDecay * dt);
      for (const c of f.dye) for (let k = 0; k < c.length; k++) c[k] *= keep;
    }

    // After advection, not before: the source must hold its value at the END
    // of the step, or the stamp is carried off within the same step.
    this.dyeSource?.(g, f.dye, dt);
    this.perf.mark('dye');

    this.dt = dt;
    this.time += dt;
  }

  /** CFL actually achieved by the last step — dtMax may have capped it. */
  get cfl(): number {
    return (this.maxFaceSpeed() * this.dt) / this.g.h;
  }
}
