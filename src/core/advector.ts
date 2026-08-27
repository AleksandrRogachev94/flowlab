import {
  advectScalar,
  advectScalarMacCormack,
  advectVelocity,
  advectVelocityMacCormack,
  type AdvectionScheme,
} from './advect.ts';
import type { FieldArray, Grid } from './grid.ts';

/**
 * The advection seam, the twin of PressureSolver — one interface, a CPU
 * implementation and a GPU one, swappable at runtime so the two can be
 * compared on identical state.
 *
 * Two methods rather than one because the two calls sit on OPPOSITE sides of
 * the pressure solve: velocity advects u^n, dye rides the projected u^{n+1}.
 * They are otherwise the same kernel, which is exactly why one object owns
 * both — on the GPU they share the carrier and label buffers, and the dye
 * call reuses what the velocity call already uploaded.
 *
 * ORDERING CONTRACT, relied on by the GPU implementation: within a step,
 * velocity() runs before dye(), and `label` does not change between them.
 * Simulation.step() is the only caller and does exactly that.
 *
 * Either method may return a promise — the GPU has no synchronous readback.
 * The CPU path returns undefined and `await` on it costs one microtask.
 */
export interface Advector {
  /** Shown in the readout, so a screenshot says what produced the picture. */
  readonly name: string;

  /** Self-advection of the velocity field. NOT in place: uIn must survive. */
  velocity(
    g: Grid,
    scheme: AdvectionScheme,
    uIn: FieldArray,
    vIn: FieldArray,
    uOut: FieldArray,
    vOut: FieldArray,
    label: Uint8Array,
    dt: number,
  ): void | Promise<void>;

  /**
   * Passive dye channels carried by (u, v), which must be the PROJECTED
   * velocity — a divergence-free carrier neither concentrates nor thins a
   * tracer. Channels are independent; nothing couples them.
   *
   * `dg` is the grid the dye lives on, defaulting to the velocity grid. See
   * advectScalar for what the split means and why it is nearly free.
   */
  dye(
    g: Grid,
    scheme: AdvectionScheme,
    u: FieldArray,
    v: FieldArray,
    qIn: FieldArray[],
    qOut: FieldArray[],
    label: Uint8Array,
    dt: number,
    dg?: Grid,
  ): void | Promise<void>;
}

/**
 * The reference implementation: core/advect.ts, called directly.
 *
 * It owns MacCormack's scratch buffers because they are an implementation
 * detail of the scheme, not state anyone else can read — the GPU
 * implementation keeps its equivalents on the device and nothing outside
 * either class needs to know either exists.
 */
export class CpuAdvector implements Advector {
  readonly name = 'cpu';

  private readonly uHat: Float64Array;
  private readonly vHat: Float64Array;
  /** One buffer serves every dye channel: they are advected one at a time. */
  private readonly qHat: Float64Array;

  constructor(g: Grid, dg: Grid = g) {
    this.uHat = new Float64Array((g.nx + 1) * g.ny);
    this.vHat = new Float64Array(g.nx * (g.ny + 1));
    this.qHat = new Float64Array(dg.nx * dg.ny);
  }

  velocity(
    g: Grid,
    scheme: AdvectionScheme,
    uIn: FieldArray,
    vIn: FieldArray,
    uOut: FieldArray,
    vOut: FieldArray,
    label: Uint8Array,
    dt: number,
  ): void {
    if (scheme === 'macCormack') {
      advectVelocityMacCormack(g, uIn, vIn, this.uHat, this.vHat, uOut, vOut, label, dt);
    } else {
      advectVelocity(g, uIn, vIn, uOut, vOut, label, dt);
    }
  }

  dye(
    g: Grid,
    scheme: AdvectionScheme,
    u: FieldArray,
    v: FieldArray,
    qIn: FieldArray[],
    qOut: FieldArray[],
    label: Uint8Array,
    dt: number,
    dg: Grid = g,
  ): void {
    for (let c = 0; c < qIn.length; c++) {
      if (scheme === 'macCormack') {
        advectScalarMacCormack(g, u, v, qIn[c], this.qHat, qOut[c], label, dt, dg);
      } else {
        advectScalar(g, u, v, qIn[c], qOut[c], label, dt, dg);
      }
    }
  }
}
