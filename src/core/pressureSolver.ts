import type { FieldArray, Grid } from './grid.ts';
import { solvePressure } from './pressure.ts';
import { solvePressureRedBlack } from './pressureRedBlack.ts';

/**
 * The one seam between the CPU and GPU implementations.
 *
 * Why the pressure solve and nothing else: it is a pure function of
 * (div, label, scale) that returns p, with no other coupling to the step. Any
 * solver that produces a p removing the same divergence is interchangeable —
 * the rest of step() cannot tell which one ran, and that property is exactly
 * what makes the CPU-vs-GPU comparison in PLAN.md §8 a controlled experiment
 * rather than a comparison of two whole programs.
 *
 * `solve` may return a promise, because WebGPU has no synchronous readback:
 * getting p back off the device is `await buffer.mapAsync()` and there is no
 * alternative. That single fact is why Simulation.step() is async. On the CPU
 * path the promise is never created at all — the number is returned directly
 * and `await` on a non-promise costs one microtask tick.
 */
export interface PressureSolver {
  /** Shown in the readout, so a screenshot says which solver produced it. */
  readonly name: string;
  /**
   * Fills `p` with a pressure whose gradient removes `div`. Returns the
   * number of sweeps actually used.
   *
   * `p` is read as well as written: its incoming contents are the previous
   * frame's solution, and warm-starting from it is worth real iterations.
   */
  solve(
    g: Grid,
    p: FieldArray,
    div: FieldArray,
    label: Uint8Array,
    scale: number,
    iterations: number,
    omega: number,
    tol: number,
  ): number | Promise<number>;
}

/** Lexicographic SOR with direction cycling — the reference implementation. */
export const cpuPressureSolver: PressureSolver = {
  name: 'cpu-sor',
  solve: (g, p, div, label, scale, iterations, omega, tol) =>
    solvePressure(g, p, div, label, scale, iterations, omega, tol),
};

/**
 * The GPU's algorithm, on the CPU, in float64. The control that separates
 * "red-black is wrong" from "my WGSL is wrong" — see pressureRedBlack.ts.
 */
export const cpuRedBlackSolver: PressureSolver = {
  name: 'cpu-rbsor',
  solve: (g, p, div, label, scale, iterations, omega, tol) =>
    solvePressureRedBlack(g, p, div, label, scale, iterations, omega, tol),
};
