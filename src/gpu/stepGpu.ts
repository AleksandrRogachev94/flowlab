import type { AdvectionScheme } from '../core/advect.ts';
import { PATCH_PLANES, type DyePatch } from '../core/dye.ts';
import type { FieldArray, Fields, Grid } from '../core/grid.ts';
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
 * gradient -> outflow -> dye -> inlet, recorded back to back, with the
 * velocity, the pressure AND the dye resident on the device between frames.
 *
 * This class owns no field buffers and no solver pipelines. It COMPOSES the
 * two classes that already have them — GpuAdvector contributes its buffers and
 * dispatch recording, GpuMultigridSolver its level stack and V-cycle — and
 * adds only the glue kernels of project.wgsl. That is deliberate: the
 * standalone classes stay behind their seams for the G/A comparisons and the
 * gputests, and there is exactly one copy of every pipeline.
 *
 * WHAT STILL CROSSES THE BUS, per frame:
 *
 *   up:    two small uniforms.
 *   down:  u and v. The CFL clamp, the residual diagnostic and the arrows all
 *          read them.
 *
 * `p` never crosses at all: it warm-starts the next solve from where it
 * already lives. u and v cross DOWN only — the next step's advection reads the
 * device copy, which is bit-identical to what the host just received.
 *
 * THE DYE USED TO CROSS BOTH WAYS, and it was by far the largest thing on the
 * bus: three channels of every cell, 25 MB each way at 1920x1080, and four
 * times that once the dye grid is refined past the velocity grid. Both
 * directions had the same cause — the host held the authoritative copy,
 * because the scene's dye source ran there and the canvas draw read it there.
 * Neither is true any more. The source is a DyePatch that a small kernel below
 * consumes (core/dye.ts describes it, and holds the host half that must stay
 * arithmetically identical), and the picture is drawn from the device buffer
 * by viz/dyeGpu.ts. What is left is readDye(), which the host asks for only
 * when it is about to take the dye back.
 *
 * THE RESIDENT-FIELD TRICK, now used twice: after each advection pass the
 * result is copied back over the INPUT buffer. From there on uIn/vIn ARE the
 * velocity and dyeIn IS the dye — the divergence reads them, the projection
 * updates them in place, the renderer draws them, and next frame's advection
 * consumes them — so no bind group anywhere changes. The dye copy is ~30 MB of
 * device-local traffic at medium quality (about 3% of a 60 fps frame on an M4
 * Air), bought in exchange for not threading a ping-pong flag through ten bind
 * groups; if it ever shows up in a profile, that is the trade to revisit.
 *
 * THE READBACK RUNS A FRAME BEHIND. step() submits and returns; the map is
 * awaited at the top of the NEXT step (drainReadback), not the bottom of its
 * own. Awaiting it in place made the frame strictly serial — the device
 * finished, then the host did its CFL scan and its draw with the device idle,
 * then the next frame was submitted — which measured as ~4.8 ms of device work
 * inside a 17.7 ms frame. Returning early lets the two overlap, and costs only
 * that f.u/f.v are one step old (see drainReadback for why that is safe).
 *
 * invalidate() is the escape hatch: reset(), or switching back from a CPU
 * solver, leaves the host holding truth, and the next step() starts by
 * uploading u, v, label, p and dye once. Labels also re-coarsen then, and ONLY
 * then — a static scene never re-runs the chain. It also makes the next drain
 * DISCARD whatever was in flight, which would otherwise overwrite the host's
 * new fields with the device's old ones.
 */

/** Must match @workgroup_size in project.wgsl. */
const WORKGROUP = 8;
const OUTFLOW_WORKGROUP = 64;

/** project.wgsl's Params: nx, ny, divCoef, gradScale. */
const PARAMS_BYTES = 16;

/** project.wgsl's DyeParams: eight words, all of them named there. */
const DYE_PARAMS_BYTES = 32;

/** Must match CHANNELS in project.wgsl / DYE_CHANNELS in core/grid.ts. */
const DYE_CHANNELS = 3;

export class GpuStepper implements GpuStep {
  private readonly device: GPUDevice;
  private readonly nx: number;
  private readonly ny: number;
  private readonly h: number;
  private readonly dg: Grid;
  private readonly dyeCells: number;

  private readonly paramsBuf: GPUBuffer;
  private readonly paramsData = new ArrayBuffer(PARAMS_BYTES);
  private readonly paramsU32 = new Uint32Array(this.paramsData);
  private readonly paramsF32 = new Float32Array(this.paramsData);

  private readonly dyeParamsBuf: GPUBuffer;
  private readonly dyeParamsData = new ArrayBuffer(DYE_PARAMS_BYTES);
  private readonly dyeParamsU32 = new Uint32Array(this.dyeParamsData);
  private readonly dyeParamsF32 = new Float32Array(this.dyeParamsData);

  private readonly divergencePipeline: GPUComputePipeline;
  private readonly subtractUPipeline: GPUComputePipeline;
  private readonly subtractVPipeline: GPUComputePipeline;
  private readonly outflowPipeline: GPUComputePipeline;
  private readonly decayPipeline: GPUComputePipeline;
  private readonly patchPipeline: GPUComputePipeline;
  private readonly divergenceGroup: GPUBindGroup;
  private readonly subtractUGroup: GPUBindGroup;
  private readonly subtractVGroup: GPUBindGroup;
  private readonly outflowGroup: GPUBindGroup;
  private readonly decayGroup: GPUBindGroup;

  /** The scene's inlet. Both are null until setDyePatch, and both are replaced
   *  together — the buffer's size follows the rect, so the group must too. */
  private patch: DyePatch | null = null;
  private patchBuf: GPUBuffer | null = null;
  private patchGroup: GPUBindGroup | null = null;

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

  /**
   * LAST frame's readbacks, still in flight — the map is started right after
   * the submit and awaited at the TOP of the next step, never at the bottom of
   * its own. See "THE READBACK RUNS A FRAME BEHIND" in the class comment.
   *
   * Both or neither: they are started together and drained together, and the
   * timestamps have to follow the velocity or reading them would reintroduce
   * exactly the stall this removes.
   */
  private velMap: Promise<undefined> | null = null;
  private timeMap: Promise<undefined> | null = null;

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
    this.dg = adv.dg;
    this.dyeCells = adv.dg.nx * adv.dg.ny;

    const uniform = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
    this.paramsBuf = device.createBuffer({
      label: 'project-params',
      size: PARAMS_BYTES,
      usage: uniform,
    });
    this.dyeParamsBuf = device.createBuffer({
      label: 'dye-params',
      size: DYE_PARAMS_BYTES,
      usage: uniform,
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
    this.patchPipeline = pipe('dye_patch');

    // Binding numbers are project.wgsl's fixed slots; each auto layout keeps
    // only the slots its entry point uses. The buffers are the OTHERS' — the
    // advector's resident velocity, dye and labels, the multigrid's fine x and
    // b — which is the composition in one place.
    const fine = mg.levels[0];
    this.divergenceGroup = this.group(this.divergencePipeline, [
      { binding: 0, buffer: this.paramsBuf },
      { binding: 3, buffer: adv.uIn },
      { binding: 4, buffer: adv.vIn },
      { binding: 5, buffer: fine.b },
    ]);
    this.subtractUGroup = this.group(this.subtractUPipeline, [
      { binding: 0, buffer: this.paramsBuf },
      { binding: 1, buffer: adv.labelBuf },
      { binding: 2, buffer: fine.x },
      { binding: 3, buffer: adv.uIn },
    ]);
    this.subtractVGroup = this.group(this.subtractVPipeline, [
      { binding: 0, buffer: this.paramsBuf },
      { binding: 1, buffer: adv.labelBuf },
      { binding: 2, buffer: fine.x },
      { binding: 4, buffer: adv.vIn },
    ]);
    this.outflowGroup = this.group(this.outflowPipeline, [
      { binding: 0, buffer: this.paramsBuf },
      { binding: 1, buffer: adv.labelBuf },
      { binding: 3, buffer: adv.uIn },
      { binding: 4, buffer: adv.vIn },
    ]);
    this.decayGroup = this.group(this.decayPipeline, [
      { binding: 6, buffer: adv.dyeIn },
      { binding: 7, buffer: this.dyeParamsBuf },
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

  private group(
    pipeline: GPUComputePipeline,
    entries: { binding: number; buffer: GPUBuffer }[],
  ): GPUBindGroup {
    return this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: entries.map((e) => ({ binding: e.binding, resource: { buffer: e.buffer } })),
    });
  }

  /** The host wrote u/v/p/label/dye (reset, a scene change, frames stepped on
   *  a CPU solver). The next step() re-uploads them and re-coarsens labels. */
  invalidate(): void {
    this.dirty = true;
  }

  /**
   * The scene's dye inlet. Uploaded here rather than per step: it is constant
   * in time, which is the whole reason it can be a rectangle of numbers
   * instead of a closure that has to run on the host.
   */
  setDyePatch(patch: DyePatch | null): void {
    this.patch = patch;
    this.patchBuf?.destroy();
    this.patchBuf = null;
    this.patchGroup = null;
    if (!patch) return;
    // Cheap, and it is the one place the two layouts could silently disagree:
    // project.wgsl indexes `patch` as CHANNELS value planes then a coverage
    // plane, and reads garbage rather than failing if the host sent fewer.
    if (patch.data.length !== PATCH_PLANES * patch.nx * patch.ny) {
      throw new Error('dye patch is not PATCH_PLANES planes of nx*ny');
    }
    this.patchBuf = this.device.createBuffer({
      label: 'dye-patch',
      size: patch.data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    // The cast is the typed-array buffer parameter, not a claim about the
    // data: a patch is `new Float32Array(n)` and never shared.
    this.device.queue.writeBuffer(this.patchBuf, 0, patch.data as Float32Array<ArrayBuffer>);
    this.patchGroup = this.group(this.patchPipeline, [
      { binding: 6, buffer: this.adv.dyeIn },
      { binding: 7, buffer: this.dyeParamsBuf },
      { binding: 8, buffer: this.patchBuf },
    ]);
  }

  /** See GpuStep.readDye: the host mirror is stale while this path drives, and
   *  this is the one way to make it true again. */
  async readDye(dye: FieldArray[]): Promise<void> {
    if (this.busy) throw new Error('GpuStepper.readDye() during step()');
    if (this.ctx.lost) throw new Error('GPU device lost');
    const { adv } = this;
    const encoder = this.device.createCommandEncoder({ label: 'read-dye' });
    encoder.copyBufferToBuffer(adv.dyeIn, 0, adv.dyeRead, 0, adv.dyeLen * 4);
    this.device.queue.submit([encoder.finish()]);
    await adv.dyeRead.mapAsync(GPUMapMode.READ);
    const mapped = adv.dyeRead.getMappedRange();
    for (let c = 0; c < dye.length; c++) {
      dye[c].set(new Float32Array(mapped, c * this.dyeCells * 4, this.dyeCells));
    }
    adv.dyeRead.unmap();
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

      // Last frame's results first, and BEFORE the dirty upload below reads
      // f.u/f.v — a drain that landed after it would overwrite exactly the
      // host data the upload is there to send.
      const tDrain = performance.now();
      await this.drainReadback(f);
      const t0 = performance.now();
      this.timings.wait = smooth(this.timings.wait, t0 - tDrain);
      perf.mark('readback');

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
        this.uploadDye(f.dye);
      }
      adv.writeParams(dt);
      this.writeParams(scale, gradScale);
      this.writeDyeParams(dyeKeep);
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
      // The resident-field trick (class comment): fold the advected result
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
        this.dispatch(
          pass2,
          this.decayPipeline,
          this.decayGroup,
          this.dg.nx,
          this.dg.ny * DYE_CHANNELS,
        );
      }
      adv.recordDye(pass2, mac);
      pass2.end();

      encoder.copyBufferToBuffer(adv.uIn, 0, adv.velRead, 0, adv.uBytes);
      encoder.copyBufferToBuffer(adv.vIn, 0, adv.velRead, adv.uBytes, adv.vBytes);
      // dyeIn becomes the advected dye, the same fold the velocity just had.
      encoder.copyBufferToBuffer(mac ? adv.dyeB : adv.dyeA, 0, adv.dyeIn, 0, adv.dyeLen * 4);

      // The inlet, AFTER the advection rather than before it, because a
      // Dirichlet condition on the tracer must hold at the END of the step —
      // applied first it would simply be carried off within the same step.
      // Its own pass because it has to follow a copy, and a copy cannot be
      // recorded inside one. A few columns of dispatch; the pass costs more
      // than the kernel.
      if (this.patchGroup && this.patch) {
        const pass3 = encoder.beginComputePass({ label: 'step-dye-inlet' });
        this.dispatch(pass3, this.patchPipeline, this.patchGroup, this.patch.nx, this.patch.ny);
        pass3.end();
      }

      if (this.querySet && this.queryResolve && this.queryRead) {
        encoder.resolveQuerySet(this.querySet, 0, 2, this.queryResolve, 0);
        encoder.copyBufferToBuffer(this.queryResolve, 0, this.queryRead, 0, 16);
      }
      queue.submit([encoder.finish()]);
      this.dirty = false;

      // Started, NOT awaited: this is the whole optimization. The caller gets
      // control back with the submit still executing, so the host's own frame
      // work — the CFL scan, the draw — overlaps the device instead of
      // queueing behind it. Drained at the top of the next step().
      this.velMap = adv.velRead.mapAsync(GPUMapMode.READ);
      this.timeMap = this.queryRead?.mapAsync(GPUMapMode.READ) ?? null;
      perf.mark('solve');

      this.timings.upload = smooth(this.timings.upload, t1 - t0);
      return mg.cycles;
    } finally {
      this.busy = false;
    }
  }

  /**
   * Last frame's velocity and timestamps, copied into the host fields.
   *
   * Runs at the top of the next step rather than the bottom of its own, which
   * is what lets the device work while the host does. By the time this is
   * reached the submit has usually long since retired, so the await is
   * typically free — `timings.wait` is what it actually cost, and it is the
   * number to watch if this ever stops being true.
   *
   * ONE FRAME STALE, deliberately. f.u/f.v are the previous step's output, so
   * the CFL bound they feed is one step behind. That is safe because the bound
   * is a target rather than a limit — semi-Lagrangian advection is
   * unconditionally stable, `cflTarget` is an accuracy knob (see
   * SimulationParams), and a step's velocity cannot change enough in one frame
   * to matter to it. The DEVICE's own copy is never stale: it advects from
   * uIn/vIn, which it wrote itself.
   */
  private async drainReadback(f: Fields): Promise<void> {
    if (!this.velMap) return;
    const { adv } = this;
    await Promise.all([this.velMap, this.timeMap]);
    this.velMap = null;
    this.timeMap = null;

    // A reset or a CPU frame happened while this was in flight, so the host
    // now holds truth and these numbers are from before it. Unmap and drop
    // them — writing them into f.u/f.v would undo whatever the host just did.
    if (!this.dirty) {
      // Copy out BEFORE unmap — unmap() detaches the ranges silently.
      const vel = adv.velRead.getMappedRange();
      f.u.set(new Float32Array(vel, 0, adv.uLen));
      f.v.set(new Float32Array(vel, adv.uBytes, adv.vLen));
    }
    adv.velRead.unmap();

    if (this.queryRead) {
      const ns = new BigUint64Array(this.queryRead.getMappedRange());
      const delta = ns[1] - ns[0];
      this.queryRead.unmap();
      const device = delta > 0n ? Number(delta) / 1e6 : NaN;
      this.hasDeviceTime = Number.isFinite(device);
      this.timings.device = smooth(this.timings.device, device);
    }
  }

  /**
   * The one place the dye still goes UP, and it runs on an invalidate rather
   * than per frame.
   *
   * Float32 fields go STRAIGHT to the queue, one write per channel at its own
   * offset — writeBuffer stages its own copy, so routing it through dyeHost as
   * well would pay for that twice. dyeHost is still the path for a Float64
   * sim, where the copy is also the conversion.
   */
  private uploadDye(dye: FieldArray[]): void {
    const { queue } = this.device;
    const { adv } = this;
    if (dye[0] instanceof Float32Array) {
      for (let c = 0; c < dye.length; c++) {
        // The cast is the FieldArray union's unparameterized buffer type, not
        // a claim about the data: fields are `new ctor(n)`, never shared.
        queue.writeBuffer(adv.dyeIn, c * this.dyeCells * 4, dye[c] as Float32Array<ArrayBuffer>);
      }
    } else {
      for (let c = 0; c < dye.length; c++) adv.dyeHost.set(dye[c], c * this.dyeCells);
      queue.writeBuffer(adv.dyeIn, 0, adv.dyeHost);
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

  private writeParams(scale: number, gradScale: number): void {
    this.paramsU32[0] = this.nx;
    this.paramsU32[1] = this.ny;
    // divergence() writes the multigrid rhs directly: b = -scale * div, with
    // the 1/h of the divergence itself folded in — see project.wgsl.
    this.paramsF32[2] = -scale / this.h;
    this.paramsF32[3] = gradScale;
    this.device.queue.writeBuffer(this.paramsBuf, 0, this.paramsData);
  }

  private writeDyeParams(keep: number): void {
    const p = this.patch;
    this.dyeParamsU32[0] = this.dg.nx;
    this.dyeParamsU32[1] = this.dg.ny;
    this.dyeParamsF32[2] = keep;
    this.dyeParamsU32[3] = p ? p.i0 : 0;
    this.dyeParamsU32[4] = p ? p.j0 : 0;
    this.dyeParamsU32[5] = p ? p.nx : 0;
    this.dyeParamsU32[6] = p ? p.ny : 0;
    this.device.queue.writeBuffer(this.dyeParamsBuf, 0, this.dyeParamsData);
  }

  /** See GpuAdvector.destroy. The field buffers belong to the advector and the
   *  solver this composes; the patch buffer is this class's own. */
  destroy(): void {
    // An in-flight map rejects when its buffer goes away, and a rejection
    // nobody is awaiting is an unhandled one. Attaching the handler here is
    // enough; the result is genuinely not wanted.
    this.velMap?.catch(() => {});
    this.timeMap?.catch(() => {});
    this.velMap = null;
    this.timeMap = null;
    this.paramsBuf.destroy();
    this.dyeParamsBuf.destroy();
    this.patchBuf?.destroy();
    this.queryResolve?.destroy();
    this.queryRead?.destroy();
    this.querySet?.destroy();
  }
}
