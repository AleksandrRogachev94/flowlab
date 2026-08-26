import type { AdvectionScheme } from '../core/advect.ts';
import type { Fields, Grid } from '../core/grid.ts';
import { smooth, type Profiler } from '../core/profiler.ts';
import type { GpuStep } from '../core/simulation.ts';
import type { GpuAdvector } from './advectGpu.ts';
import type { GpuContext } from './device.ts';
import type { GpuMultigridSolver } from './multigridGpu.ts';
import type { GpuTimings } from './pressureGpu.ts';
import projectSource from './project.wgsl?raw';

/**
 * The whole step in ONE submit — the ending docs/WEBGPU.md §8 promised when it
 * counted the three round trips: advect -> divergence -> V-cycles -> subtract
 * gradient -> outflow -> dye, recorded back to back, with the velocity and the
 * pressure RESIDENT on the device between frames.
 *
 * This class owns no field buffers and no solver pipelines. It COMPOSES the
 * two classes that already have them — GpuAdvector contributes its buffers and
 * dispatch recording, GpuMultigridSolver its level stack and V-cycle — and
 * adds only the three glue kernels of project.wgsl. That is deliberate: the
 * standalone classes stay behind their seams for the G/A comparisons and the
 * gputests, and there is exactly one copy of every pipeline.
 *
 * WHAT STILL CROSSES THE BUS, per frame:
 *
 *   up:    dye (the host mirror is authoritative — the scene's DyeSource
 *          closure runs there, and staying host-side is what keeps every
 *          existing source working unchanged), plus two small uniforms. The
 *          fade that used to share that reason is now project.wgsl's `decay`.
 *   down:  u, v (the CFL clamp, the residual diagnostic and the arrows all
 *          read them) and dye (the canvas draw does).
 *
 * `p` never crosses at all: it warm-starts the next solve from where it
 * already lives. u and v cross DOWN only — the next step's advection reads the
 * device copy, which is bit-identical to what the host just received.
 *
 * THE RESIDENT-VELOCITY TRICK: after the advection pass, the result (uA/uB) is
 * copied back over uIn/vIn. From there on uIn/vIn ARE the velocity — the
 * divergence reads them, the projection updates them in place, the dye
 * advection's existing bind groups already point their carrier at them, and
 * next frame's advection consumes them — so no bind group anywhere changes.
 *
 * invalidate() is the escape hatch: reset(), or switching back from a CPU
 * solver, leaves the host holding truth, and the next step() starts by
 * uploading u, v, label and p once. Labels also re-coarsen then, and ONLY
 * then — a static scene never re-runs the chain.
 */

/** Must match @workgroup_size in project.wgsl. */
const WORKGROUP = 8;
const OUTFLOW_WORKGROUP = 64;

/** project.wgsl's uniform block: nx, ny, divCoef, gradScale, dyeKeep. Five
 *  words, but a uniform struct rounds up to a multiple of 16 — hence 32. */
const PARAMS_BYTES = 32;

/** Must match CHANNELS in project.wgsl / DYE_CHANNELS in core/grid.ts. */
const DYE_CHANNELS = 3;

export class GpuStepper implements GpuStep {
  private readonly device: GPUDevice;
  private readonly nx: number;
  private readonly ny: number;
  private readonly h: number;
  private readonly cells: number;

  private readonly paramsBuf: GPUBuffer;
  private readonly paramsData = new ArrayBuffer(PARAMS_BYTES);
  private readonly paramsU32 = new Uint32Array(this.paramsData);
  private readonly paramsF32 = new Float32Array(this.paramsData);

  private readonly divergencePipeline: GPUComputePipeline;
  private readonly subtractUPipeline: GPUComputePipeline;
  private readonly subtractVPipeline: GPUComputePipeline;
  private readonly outflowPipeline: GPUComputePipeline;
  private readonly decayPipeline: GPUComputePipeline;
  private readonly divergenceGroup: GPUBindGroup;
  private readonly subtractUGroup: GPUBindGroup;
  private readonly subtractVGroup: GPUBindGroup;
  private readonly outflowGroup: GPUBindGroup;
  private readonly decayGroup: GPUBindGroup;

  private readonly querySet: GPUQuerySet | null = null;
  private readonly queryResolve: GPUBuffer | null = null;
  private readonly queryRead: GPUBuffer | null = null;

  /** Same panel contract as the standalone GPU solvers, but `device` here
   *  covers the WHOLE step's pass, not just the pressure solve. */
  readonly timings: GpuTimings = { upload: 0, wait: 0, device: 0 };
  hasDeviceTime = false;

  /** True whenever the HOST holds truth the device hasn't seen. Starts true:
   *  the first step is exactly the "host just changed everything" case. */
  private dirty = true;
  private busy = false;

  constructor(
    private readonly ctx: GpuContext,
    g: Grid,
    private readonly adv: GpuAdvector,
    private readonly mg: GpuMultigridSolver,
  ) {
    const { device } = ctx;
    this.device = device;
    this.nx = g.nx;
    this.ny = g.ny;
    this.h = g.h;
    this.cells = g.nx * g.ny;

    this.paramsBuf = device.createBuffer({
      label: 'project-params',
      size: PARAMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const module = device.createShaderModule({ label: 'project', code: projectSource });
    const pipe = (entryPoint: string): GPUComputePipeline =>
      device.createComputePipeline({
        label: `project-${entryPoint}`,
        layout: 'auto',
        compute: { module, entryPoint },
      });
    this.divergencePipeline = pipe('divergence');
    this.subtractUPipeline = pipe('subtract_u');
    this.subtractVPipeline = pipe('subtract_v');
    this.outflowPipeline = pipe('outflow');
    this.decayPipeline = pipe('decay');

    // Binding numbers are project.wgsl's fixed slots; each auto layout keeps
    // only the slots its entry point uses. The buffers are the OTHERS' — the
    // advector's resident velocity and labels, the multigrid's fine x and b —
    // which is the composition in one place.
    const fine = mg.levels[0];
    const group = (
      pipeline: GPUComputePipeline,
      entries: { binding: number; buffer: GPUBuffer }[],
    ): GPUBindGroup =>
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: entries.map((e) => ({ binding: e.binding, resource: { buffer: e.buffer } })),
      });
    this.divergenceGroup = group(this.divergencePipeline, [
      { binding: 0, buffer: this.paramsBuf },
      { binding: 3, buffer: adv.uIn },
      { binding: 4, buffer: adv.vIn },
      { binding: 5, buffer: fine.b },
    ]);
    this.subtractUGroup = group(this.subtractUPipeline, [
      { binding: 0, buffer: this.paramsBuf },
      { binding: 1, buffer: adv.labelBuf },
      { binding: 2, buffer: fine.x },
      { binding: 3, buffer: adv.uIn },
    ]);
    this.subtractVGroup = group(this.subtractVPipeline, [
      { binding: 0, buffer: this.paramsBuf },
      { binding: 1, buffer: adv.labelBuf },
      { binding: 2, buffer: fine.x },
      { binding: 4, buffer: adv.vIn },
    ]);
    this.outflowGroup = group(this.outflowPipeline, [
      { binding: 0, buffer: this.paramsBuf },
      { binding: 1, buffer: adv.labelBuf },
      { binding: 3, buffer: adv.uIn },
      { binding: 4, buffer: adv.vIn },
    ]);
    this.decayGroup = group(this.decayPipeline, [
      { binding: 0, buffer: this.paramsBuf },
      { binding: 6, buffer: adv.dyeIn },
    ]);

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

  /** The host wrote u/v/p/label (reset, a scene change, frames stepped on a
   *  CPU solver). The next step() re-uploads them and re-coarsens labels. */
  invalidate(): void {
    this.dirty = true;
  }

  async step(
    f: Fields,
    scheme: AdvectionScheme,
    dt: number,
    scale: number,
    gradScale: number,
    dyeKeep: number,
    perf: Profiler,
  ): Promise<number> {
    if (this.busy) throw new Error('GpuStepper.step() is not re-entrant');
    if (this.ctx.lost) throw new Error('GPU device lost');
    this.busy = true;
    try {
      const { queue } = this.device;
      const { adv, mg } = this;
      const fine = mg.levels[0];
      const t0 = performance.now();

      if (this.dirty) {
        // The one full host -> device sync. Labels go up twice — the fine
        // grid the kernels read, and the multigrid chain's root — and p goes
        // up so the warm start continues from whatever the host last held.
        adv.uHost.set(f.u);
        adv.vHost.set(f.v);
        adv.labelHost.set(f.label);
        mg.pF32.set(f.p);
        queue.writeBuffer(adv.uIn, 0, adv.uHost);
        queue.writeBuffer(adv.vIn, 0, adv.vHost);
        queue.writeBuffer(adv.labelBuf, 0, adv.labelHost);
        queue.writeBuffer(fine.label, 0, adv.labelHost);
        queue.writeBuffer(fine.x, 0, mg.pF32);
      }
      // Dye every frame: the scene's source ran on the host mirror since the
      // last readback, so the device copy is one write behind. (Decay used to
      // be the other reason and is now the `decay` kernel below.)
      //
      // Float32 fields go STRAIGHT to the queue, one write per channel at its
      // own offset — writeBuffer stages its own copy, so routing 25 MB a frame
      // through dyeHost as well was paying for that twice. dyeHost is still
      // the path for a Float64 sim, where the copy is also the conversion.
      if (f.dye[0] instanceof Float32Array) {
        for (let c = 0; c < f.dye.length; c++) {
          // The cast is the FieldArray union's unparameterized buffer type,
          // not a claim about the data: fields are `new ctor(n)`, never shared.
          queue.writeBuffer(adv.dyeIn, c * this.cells * 4, f.dye[c] as Float32Array<ArrayBuffer>);
        }
      } else {
        for (let c = 0; c < f.dye.length; c++) adv.dyeHost.set(f.dye[c], c * this.cells);
        queue.writeBuffer(adv.dyeIn, 0, adv.dyeHost);
      }
      adv.writeParams(dt);
      this.writeParams(scale, gradScale, dyeKeep);
      const t1 = performance.now();
      perf.mark('upload');

      const mac = scheme === 'macCormack';
      const encoder = this.device.createCommandEncoder({ label: 'gpu-step' });

      const pass1 = encoder.beginComputePass({
        label: 'step-advect',
        timestampWrites: this.querySet
          ? { querySet: this.querySet, beginningOfPassWriteIndex: 0 }
          : undefined,
      });
      adv.recordVelocity(pass1, mac);
      pass1.end();
      // The resident-velocity trick (class comment): fold the advected result
      // back into uIn/vIn so one buffer is "the velocity" for everything after.
      encoder.copyBufferToBuffer(mac ? adv.uB : adv.uA, 0, adv.uIn, 0, adv.uBytes);
      encoder.copyBufferToBuffer(mac ? adv.vB : adv.vA, 0, adv.vIn, 0, adv.vBytes);

      const pass2 = encoder.beginComputePass({
        label: 'step-project-dye',
        timestampWrites: this.querySet
          ? { querySet: this.querySet, endOfPassWriteIndex: 1 }
          : undefined,
      });
      this.dispatch(pass2, this.divergencePipeline, this.divergenceGroup, this.nx, this.ny);
      if (this.dirty) mg.recordCoarsenLabels(pass2);
      for (let c = 0; c < mg.cycles; c++) mg.recordVCycle(pass2, 0);
      this.dispatch(pass2, this.subtractUPipeline, this.subtractUGroup, this.nx + 1, this.ny);
      this.dispatch(pass2, this.subtractVPipeline, this.subtractVGroup, this.nx, this.ny + 1);
      pass2.setPipeline(this.outflowPipeline);
      pass2.setBindGroup(0, this.outflowGroup);
      pass2.dispatchWorkgroups(Math.ceil(Math.max(this.nx, this.ny) / OUTFLOW_WORKGROUP));
      // Before the dye advection reads dyeIn, and legitimately so: see the
      // linearity argument in project.wgsl. Skipped outright when the scene
      // has no decay — keep is exactly 1 then, and multiplying 6M floats by
      // one is not free just because it is a no-op.
      if (dyeKeep !== 1) {
        this.dispatch(pass2, this.decayPipeline, this.decayGroup, this.nx, this.ny * DYE_CHANNELS);
      }
      adv.recordDye(pass2, mac);
      pass2.end();

      encoder.copyBufferToBuffer(adv.uIn, 0, adv.velRead, 0, adv.uBytes);
      encoder.copyBufferToBuffer(adv.vIn, 0, adv.velRead, adv.uBytes, adv.vBytes);
      encoder.copyBufferToBuffer(mac ? adv.dyeB : adv.dyeA, 0, adv.dyeRead, 0, adv.dyeLen * 4);
      if (this.querySet && this.queryResolve && this.queryRead) {
        encoder.resolveQuerySet(this.querySet, 0, 2, this.queryResolve, 0);
        encoder.copyBufferToBuffer(this.queryResolve, 0, this.queryRead, 0, 16);
      }
      queue.submit([encoder.finish()]);
      this.dirty = false;

      // One await for both maps — two sequential awaits would be two stalls.
      await Promise.all([
        adv.velRead.mapAsync(GPUMapMode.READ),
        adv.dyeRead.mapAsync(GPUMapMode.READ),
      ]);
      perf.mark('solve');

      // Copy out BEFORE unmap — unmap() detaches the ranges silently.
      const vel = adv.velRead.getMappedRange();
      f.u.set(new Float32Array(vel, 0, adv.uLen));
      f.v.set(new Float32Array(vel, adv.uBytes, adv.vLen));
      adv.velRead.unmap();
      const dye = adv.dyeRead.getMappedRange();
      for (let c = 0; c < f.dye.length; c++) {
        f.dye[c].set(new Float32Array(dye, c * this.cells * 4, this.cells));
      }
      adv.dyeRead.unmap();
      perf.mark('readback');

      this.timings.upload = smooth(this.timings.upload, t1 - t0);
      this.timings.wait = smooth(this.timings.wait, performance.now() - t1);
      const device = await this.readDeviceTime();
      this.hasDeviceTime = Number.isFinite(device);
      this.timings.device = smooth(this.timings.device, device);
      return mg.cycles;
    } finally {
      this.busy = false;
    }
  }

  private dispatch(
    pass: GPUComputePassEncoder,
    pipeline: GPUComputePipeline,
    group: GPUBindGroup,
    width: number,
    height: number,
  ): void {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(width / WORKGROUP), Math.ceil(height / WORKGROUP));
  }

  private writeParams(scale: number, gradScale: number, dyeKeep: number): void {
    this.paramsU32[0] = this.nx;
    this.paramsU32[1] = this.ny;
    // divergence() writes the multigrid rhs directly: b = -scale * div, with
    // the 1/h of the divergence itself folded in — see project.wgsl.
    this.paramsF32[2] = -scale / this.h;
    this.paramsF32[3] = gradScale;
    this.paramsF32[4] = dyeKeep;
    this.device.queue.writeBuffer(this.paramsBuf, 0, this.paramsData);
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
