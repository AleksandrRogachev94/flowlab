import type { FieldArray, Grid } from '../core/grid.ts';
import { fluidDivRms } from '../core/pressure.ts';
import {
  levelSizes,
  MG_COARSE_SWEEPS,
  MG_CYCLES,
  MG_OMEGA,
  MG_POST_SWEEPS,
  MG_PRE_SWEEPS,
} from '../core/pressureMultigrid.ts';
import type { PressureSolver } from '../core/pressureSolver.ts';
import { smooth } from '../core/profiler.ts';
import type { GpuContext } from './device.ts';
import type { GpuTimings } from './pressureGpu.ts';
import multigridSource from './multigrid.wgsl?raw';
import redBlackSource from './redBlack.wgsl?raw';

/**
 * The V-cycle of core/pressureMultigrid.ts on the device — same operators,
 * same constants, f32. Read that file for the algorithm; this one is only the
 * plumbing, and the plumbing follows pressureGpu.ts closely: upload the fine
 * (p, div, label), record every dispatch of every cycle into ONE compute pass
 * in ONE submit, read the fine p back as the single await.
 *
 * What is genuinely new against the SOR solver:
 *
 *   - A buffer set PER LEVEL (x, b, r, label), allocated once. Total extra
 *     memory is 4/3 of the fine grid per field. Nothing is ever read back
 *     from the coarse levels — they exist only between dispatches.
 *   - The SMOOTHER pipeline is compiled from redBlack.wgsl, unchanged, with
 *     scale = -1 in its params so its equation becomes (sum + b)/count. The
 *     shader that 200 sweeps of gputest pinned against the CPU is the shader
 *     multigrid runs; only multigrid.wgsl's four transfer kernels are new
 *     code, and they carry their own gputest.
 *   - Labels are coarsened ON the device, a chain of tiny dispatches at the
 *     head of the same pass, so the host neither loops over N cells a frame
 *     nor owns a "did labels change" staleness bug.
 */

/** Must match @workgroup_size in both .wgsl files. */
const WORKGROUP = 8;

/** redBlack.wgsl's uniform block: nx, ny, scale, omega, color + padding. */
const SMOOTH_PARAMS_BYTES = 32;
/** multigrid.wgsl's uniform block: nx, ny, cnx, cny. */
const LEVEL_PARAMS_BYTES = 16;

interface LevelBuffers {
  nx: number;
  ny: number;
  groupsX: number;
  groupsY: number;
  x: GPUBuffer;
  b: GPUBuffer;
  r: GPUBuffer;
  label: GPUBuffer;
}

export class GpuMultigridSolver implements PressureSolver {
  readonly name = 'gpu-mg';

  private readonly device: GPUDevice;
  private readonly cycles: number;
  private readonly bytes: number;

  private readonly levels: LevelBuffers[];
  private readonly readBuf: GPUBuffer;

  private readonly smoothPipeline: GPUComputePipeline;
  /** [level][colour] */
  private readonly smoothGroups: [GPUBindGroup, GPUBindGroup][];
  private readonly residualPipeline: GPUComputePipeline;
  private readonly restrictPipeline: GPUComputePipeline;
  private readonly prolongPipeline: GPUComputePipeline;
  private readonly coarsenPipeline: GPUComputePipeline;
  /** [level 0..L-2], each kernel's resources for the (l, l+1) pair. */
  private readonly residualGroups: GPUBindGroup[];
  private readonly restrictGroups: GPUBindGroup[];
  private readonly prolongGroups: GPUBindGroup[];
  private readonly coarsenGroups: GPUBindGroup[];

  private readonly querySet: GPUQuerySet | null = null;
  private readonly queryResolve: GPUBuffer | null = null;
  private readonly queryRead: GPUBuffer | null = null;

  // Host staging allocated once, never in the frame loop.
  private readonly pF32: Float32Array<ArrayBuffer>;
  private readonly bF32: Float32Array<ArrayBuffer>;
  private readonly labelU32: Uint32Array<ArrayBuffer>;

  readonly timings: GpuTimings = { upload: 0, wait: 0, device: 0 };
  hasDeviceTime = false;

  private busy = false;

  constructor(
    private readonly ctx: GpuContext,
    g: Grid,
    cycles = MG_CYCLES,
  ) {
    const { device } = ctx;
    this.device = device;
    this.cycles = cycles;
    this.bytes = g.nx * g.ny * 4;
    this.pF32 = new Float32Array(g.nx * g.ny);
    this.bF32 = new Float32Array(g.nx * g.ny);
    this.labelU32 = new Uint32Array(g.nx * g.ny);

    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    this.levels = levelSizes(g.nx, g.ny).map((s, l) => {
      const bytes = s.nx * s.ny * 4;
      const mk = (name: string, extra = 0): GPUBuffer =>
        device.createBuffer({ label: `mg-${name}-${l}`, size: bytes, usage: storage | extra });
      return {
        nx: s.nx,
        ny: s.ny,
        groupsX: Math.ceil(s.nx / WORKGROUP),
        groupsY: Math.ceil(s.ny / WORKGROUP),
        // COPY_SRC on x only at the fine level: the one buffer that comes back.
        x: mk('x', l === 0 ? GPUBufferUsage.COPY_SRC : 0),
        b: mk('b'),
        r: mk('r'),
        label: mk('label'),
      };
    });
    this.readBuf = device.createBuffer({
      label: 'mg-readback',
      size: this.bytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const smoothModule = device.createShaderModule({ code: redBlackSource });
    const mgModule = device.createShaderModule({ code: multigridSource });
    const pipe = (module: GPUShaderModule, entryPoint: string): GPUComputePipeline =>
      device.createComputePipeline({
        label: `mg-${entryPoint}`,
        layout: 'auto',
        compute: { module, entryPoint },
      });
    this.smoothPipeline = pipe(smoothModule, 'main');
    this.residualPipeline = pipe(mgModule, 'residual');
    this.restrictPipeline = pipe(mgModule, 'restrictResidual');
    this.prolongPipeline = pipe(mgModule, 'prolong');
    this.coarsenPipeline = pipe(mgModule, 'coarsenLabels');

    // Every uniform is constant for the solver's lifetime — dims are geometry,
    // scale = -1 and omega = MG_OMEGA are part of the algorithm, and the
    // frame-varying rho*h^2/dt is folded into b on upload. So all params are
    // written once, here, and solve() never touches a uniform again.
    const uniform = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
    const smoothParams = this.levels.map((lv, l) =>
      ([0, 1] as const).map((colour) => {
        const buf = device.createBuffer({
          label: `mg-smooth-params-${l}-${colour}`,
          size: SMOOTH_PARAMS_BYTES,
          usage: uniform,
        });
        const data = new ArrayBuffer(SMOOTH_PARAMS_BYTES);
        const u = new Uint32Array(data);
        const f = new Float32Array(data);
        u[0] = lv.nx;
        u[1] = lv.ny;
        f[2] = -1; // scale: turns the SOR kernel's (sum - scale*div) into (sum + b)
        f[3] = MG_OMEGA;
        u[4] = colour;
        device.queue.writeBuffer(buf, 0, data);
        return buf;
      }),
    );
    const levelParams = this.levels.slice(0, -1).map((lv, l) => {
      const buf = device.createBuffer({
        label: `mg-level-params-${l}`,
        size: LEVEL_PARAMS_BYTES,
        usage: uniform,
      });
      const next = this.levels[l + 1];
      device.queue.writeBuffer(buf, 0, new Uint32Array([lv.nx, lv.ny, next.nx, next.ny]));
      return buf;
    });

    const group = (
      pipeline: GPUComputePipeline,
      entries: { binding: number; buffer: GPUBuffer }[],
    ): GPUBindGroup =>
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: entries.map((e) => ({ binding: e.binding, resource: { buffer: e.buffer } })),
      });

    this.smoothGroups = this.levels.map((lv, l) => [
      group(this.smoothPipeline, [
        { binding: 0, buffer: smoothParams[l][0] },
        { binding: 1, buffer: lv.x },
        { binding: 2, buffer: lv.b },
        { binding: 3, buffer: lv.label },
      ]),
      group(this.smoothPipeline, [
        { binding: 0, buffer: smoothParams[l][1] },
        { binding: 1, buffer: lv.x },
        { binding: 2, buffer: lv.b },
        { binding: 3, buffer: lv.label },
      ]),
    ]);
    // Binding numbers follow multigrid.wgsl's fixed slots; each auto layout
    // only contains the slots its entry point actually uses.
    this.residualGroups = levelParams.map((params, l) =>
      group(this.residualPipeline, [
        { binding: 0, buffer: params },
        { binding: 1, buffer: this.levels[l].x },
        { binding: 2, buffer: this.levels[l].b },
        { binding: 3, buffer: this.levels[l].r },
        { binding: 4, buffer: this.levels[l].label },
      ]),
    );
    this.restrictGroups = levelParams.map((params, l) =>
      group(this.restrictPipeline, [
        { binding: 0, buffer: params },
        { binding: 3, buffer: this.levels[l].r },
        { binding: 6, buffer: this.levels[l + 1].b },
        { binding: 7, buffer: this.levels[l + 1].x },
      ]),
    );
    this.prolongGroups = levelParams.map((params, l) =>
      group(this.prolongPipeline, [
        { binding: 0, buffer: params },
        { binding: 1, buffer: this.levels[l].x },
        { binding: 4, buffer: this.levels[l].label },
        { binding: 5, buffer: this.levels[l + 1].x },
      ]),
    );
    this.coarsenGroups = levelParams.map((params, l) =>
      group(this.coarsenPipeline, [
        { binding: 0, buffer: params },
        { binding: 4, buffer: this.levels[l].label },
        { binding: 8, buffer: this.levels[l + 1].label },
      ]),
    );

    if (ctx.timestamps) {
      this.querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
      this.queryResolve = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      this.queryRead = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
    }
  }

  /** Records one V-cycle, recursively — the recursion happens at ENCODE time
   *  on the host; the GPU sees a flat, ordered list of dispatches. */
  private recordVCycle(pass: GPUComputePassEncoder, l: number): void {
    const lv = this.levels[l];
    const smoothPair = (): void => {
      pass.setPipeline(this.smoothPipeline);
      pass.setBindGroup(0, this.smoothGroups[l][0]);
      pass.dispatchWorkgroups(lv.groupsX, lv.groupsY);
      pass.setBindGroup(0, this.smoothGroups[l][1]);
      pass.dispatchWorkgroups(lv.groupsX, lv.groupsY);
    };
    if (l === this.levels.length - 1) {
      for (let s = 0; s < MG_COARSE_SWEEPS; s++) smoothPair();
      return;
    }
    const next = this.levels[l + 1];
    for (let s = 0; s < MG_PRE_SWEEPS; s++) smoothPair();
    pass.setPipeline(this.residualPipeline);
    pass.setBindGroup(0, this.residualGroups[l]);
    pass.dispatchWorkgroups(lv.groupsX, lv.groupsY);
    pass.setPipeline(this.restrictPipeline);
    pass.setBindGroup(0, this.restrictGroups[l]);
    pass.dispatchWorkgroups(next.groupsX, next.groupsY);
    this.recordVCycle(pass, l + 1);
    pass.setPipeline(this.prolongPipeline);
    pass.setBindGroup(0, this.prolongGroups[l]);
    pass.dispatchWorkgroups(lv.groupsX, lv.groupsY);
    for (let s = 0; s < MG_POST_SWEEPS; s++) smoothPair();
  }

  /**
   * `iterations`, `omega` and `tol` are accepted for interface compatibility
   * and IGNORED, for the reasons split across the other two solvers: the sweep
   * budget and its omega are SOR tuning (a V-cycle's effort is `cycles`, its
   * smoother wants MG_OMEGA — see pressureMultigrid.ts), and a tolerance needs
   * a residual readback that costs as much as the cycle it might save (see
   * pressureGpu.ts). Returns V-cycles run, so sim.iters reads in cycles.
   */
  async solve(
    g: Grid,
    p: FieldArray,
    div: FieldArray,
    label: Uint8Array,
    scale: number,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars -- see the doc comment
    iterations: number,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars -- see the doc comment
    omega: number,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars -- see the doc comment
    tol: number,
  ): Promise<number> {
    if (this.busy) throw new Error('GpuMultigridSolver.solve() is not re-entrant');
    if (this.ctx.lost) throw new Error('GPU device lost');

    // Same guard as every solver: p = 0 is the exact answer here, and a
    // warm-started leftover would re-inject divergence.
    if (fluidDivRms(div, label) === 0) {
      p.fill(0);
      return 0;
    }

    this.busy = true;
    try {
      const t0 = performance.now();
      const fine = this.levels[0];
      // b = -scale * div is the fine rhs; folding scale in here is what lets
      // every uniform stay constant (see the constructor).
      this.pF32.set(p);
      for (let k = 0; k < this.bF32.length; k++) this.bF32[k] = -scale * div[k];
      this.labelU32.set(label);
      const { queue } = this.device;
      queue.writeBuffer(fine.x, 0, this.pF32);
      queue.writeBuffer(fine.b, 0, this.bF32);
      queue.writeBuffer(fine.label, 0, this.labelU32);
      const t1 = performance.now();

      const encoder = this.device.createCommandEncoder({ label: 'mg-solve' });
      const pass = encoder.beginComputePass({
        label: 'mg-vcycles',
        timestampWrites: this.querySet
          ? { querySet: this.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 }
          : undefined,
      });
      // Labels first: each level's labels derive from the one above, and the
      // pass's implicit barriers order the chain correctly.
      pass.setPipeline(this.coarsenPipeline);
      for (let l = 0; l < this.levels.length - 1; l++) {
        pass.setBindGroup(0, this.coarsenGroups[l]);
        pass.dispatchWorkgroups(this.levels[l + 1].groupsX, this.levels[l + 1].groupsY);
      }
      for (let c = 0; c < this.cycles; c++) this.recordVCycle(pass, 0);
      pass.end();

      encoder.copyBufferToBuffer(fine.x, 0, this.readBuf, 0, this.bytes);
      if (this.querySet && this.queryResolve && this.queryRead) {
        encoder.resolveQuerySet(this.querySet, 0, 2, this.queryResolve, 0);
        encoder.copyBufferToBuffer(this.queryResolve, 0, this.queryRead, 0, 16);
      }
      queue.submit([encoder.finish()]);

      await this.readBuf.mapAsync(GPUMapMode.READ);
      p.set(new Float32Array(this.readBuf.getMappedRange()));
      this.readBuf.unmap();

      this.timings.upload = smooth(this.timings.upload, t1 - t0);
      this.timings.wait = smooth(this.timings.wait, performance.now() - t1);
      const device = await this.readDeviceTime();
      this.hasDeviceTime = Number.isFinite(device);
      this.timings.device = smooth(this.timings.device, device);
      return this.cycles;
    } finally {
      this.busy = false;
    }
  }

  /** Same contract as GpuPressureSolver.readDeviceTime. */
  private async readDeviceTime(): Promise<number> {
    if (!this.queryRead) return NaN;
    await this.queryRead.mapAsync(GPUMapMode.READ);
    const ns = new BigUint64Array(this.queryRead.getMappedRange());
    const delta = ns[1] - ns[0];
    this.queryRead.unmap();
    return delta > 0n ? Number(delta) / 1e6 : NaN;
  }
}
