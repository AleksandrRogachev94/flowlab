import { advectVelocity } from './advect.ts';
import { computeDivergence } from './divergence.ts';
import { createFields, createGrid, type FieldArray, type Fields, type Grid } from './grid.ts';
import { solvePressure } from './pressure.ts';
import { subtractGradient } from './subtractGradient.ts';

/** Writes an initial velocity field. Matches the scene helpers' signature. */
export type Seed = (g: Grid, u: FieldArray, v: FieldArray) => void;

export interface SimulationParams {
  /**
   * Target for CFL = u_max * dt / h. Semi-Lagrangian is stable at any dt, so
   * this is purely an accuracy knob, and the curve peaks near 1: measured on
   * Taylor-Green at N=64, energy kept is 50% at CFL 4, 71% at CFL 1, 68% at
   * CFL 0.13. The penalty is far steeper above 1 than below.
   */
  cflTarget: number;
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
}

/** Classical optimal SOR factor for an n x n Poisson problem. */
export function optimalOmega(n: number): number {
  return 2 / (1 + Math.sin(Math.PI / n));
}

export const defaultParams: SimulationParams = {
  cflTarget: 2.0, // initially it was 1. Bridson's book mentions 5.
  dtMax: 1 / 30,
  pressureIters: 500,
  tol: 5e-3, // initially it was 1e-3
  omega: 0, // replaced by optimalOmega(n) in the constructor
  rho: 1.0,
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

  // Rule 3: advection can't run in place (u self-advects), so these are
  // allocated once and ping-ponged by reference swap, never copied.
  private uNext: FieldArray;
  private vNext: FieldArray;

  constructor(n: number, params: Partial<SimulationParams> = {}) {
    this.params = { ...defaultParams, omega: optimalOmega(n), ...params };
    this.g = createGrid(n, n, 1 / n);
    this.f = createFields(this.g, Float64Array);
    this.div = new Float64Array(this.f.p.length);
    this.uNext = new Float64Array(this.f.u.length);
    this.vNext = new Float64Array(this.f.v.length);
  }

  /** Back to t = 0. Clears p too, or the warm start begins from a field
   *  solving a different problem. */
  reset(seed: Seed): void {
    this.f.u.fill(0);
    this.f.v.fill(0);
    this.f.p.fill(0);
    seed(this.g, this.f.u, this.f.v);
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

  step(): void {
    const { g, f, params: p } = this;
    const uMax = this.maxFaceSpeed();
    // scale/gradScale both carry dt, so they are rebuilt every step; hoisting
    // them would silently use a stale timestep.
    const dt = uMax > 1e-9 ? Math.min(p.dtMax, (p.cflTarget * g.h) / uMax) : p.dtMax;
    const scale = (p.rho * g.h * g.h) / dt;
    const gradScale = dt / (p.rho * g.h);

    advectVelocity(g, f.u, f.v, this.uNext, this.vNext, f.label, dt);
    [f.u, this.uNext] = [this.uNext, f.u];
    [f.v, this.vNext] = [this.vNext, f.v];

    computeDivergence(g, f.u, f.v, this.div);
    this.iters = solvePressure(g, f.p, this.div, f.label, scale, p.pressureIters, p.omega, p.tol);
    subtractGradient(g, f.p, f.u, f.v, f.label, gradScale);
    computeDivergence(g, f.u, f.v, this.div); // now the residual

    this.dt = dt;
    this.time += dt;
  }

  /** CFL actually achieved by the last step — dtMax may have capped it. */
  get cfl(): number {
    return (this.maxFaceSpeed() * this.dt) / this.g.h;
  }
}
