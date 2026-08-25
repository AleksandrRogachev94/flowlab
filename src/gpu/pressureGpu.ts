import type { FieldArray, Grid } from '../core/grid.ts';
import { fluidDivRms } from '../core/pressure.ts';
import type { PressureSolver } from '../core/pressureSolver.ts';
import { smooth } from '../core/profiler.ts';
import type { GpuContext } from './device.ts';
import shaderSource from './redBlack.wgsl?raw';

/**
 * Red-black SOR on the GPU, drop-in behind the PressureSolver seam.
 * The algorithm and the measurements are in docs/WEBGPU.md; this file is the
 * plumbing.
 *
 * SHAPE OF THE FRAME. Only the pressure solve runs on the device, so every
 * frame does a round trip: upload (p, div, label) -> 2 * iterations dispatches
 * -> read p back. Two things keep that cheap:
 *
 *   - ALL the dispatches go into ONE command encoder and ONE submit. 192
 *     dispatches cost one CPU->GPU handoff, not 192. WebGPU orders dispatches
 *     within a compute pass and inserts the storage barriers between them, so
 *     "red sees the previous black" is guaranteed with no explicit sync.
 *   - The readback is the only await. It stalls the CPU until the GPU is done,
 *     which is fine here: the very next thing in step() is subtractGradient,
 *     which needs p, so there is nothing else to be doing.
 *
 * The round trip is the price of a PARTIAL port. When advection moves to the
 * device too, the fields simply stay there and this upload/download
 * disappears — but the shader, the bind groups and the dispatch loop are
 * unchanged by that, which is why this is a first step rather than throwaway
 * work. The readback survives as the mechanism for diffing GPU against CPU.
 *
 * PRECISION: f32 on the device against Float64Array on the host. That is one
 * of the two variables PLAN.md §8 studies, so it is deliberate. Measured cost:
 * none so far — 200 sweeps agree with the f64 CPU red-black solver to 1.7e-7
 * on a peak of 1.4e-1, four orders below the solver's own residual.
 */

/**
 * Threads per workgroup, one axis. MUST match @workgroup_size in
 * redBlack.wgsl — WGSL cannot import a constant from TypeScript, so this is a
 * hand-kept pair. 8x8 = 64 threads is the standard starting point: a multiple
 * of the 32/64-wide execution unit on every vendor, and small enough that a
 * ragged grid edge wastes little.
 */
const WORKGROUP = 8;

/** nx, ny, scale, omega, color + 3 words of padding to a 16-byte multiple. */
const PARAMS_BYTES = 32;

/**
 * What the perf overlay reports for the GPU path. All milliseconds, and all
 * SMOOTHED here rather than by the caller — these are per-frame values and the
 * overlay repaints at 6 Hz, so raw samples would show one arbitrary frame in
 * ten. Smoothing at the source means every consumer gets the same number and
 * nobody downstream has to remember to average.
 */
export interface GpuTimings {
  /** Host-side: f64 -> f32 conversion plus queue.writeBuffer. */
  upload: number;
  /** Host-side: awaiting mapAsync. This is the stall, and it CONTAINS the
   *  compute — the GPU only finishes once we wait. */
  wait: number;
  /** Device-side: what the GPU really spent inside the compute pass, from
   *  timestamp queries. `wait - device` is the round trip's own cost. NaN
   *  when 'timestamp-query' is unavailable. */
  device: number;
}

export class GpuPressureSolver implements PressureSolver {
  readonly name = 'gpu-rbsor';

  private readonly device: GPUDevice;
  private readonly cells: number;
  private readonly bytes: number;
  private readonly groupsX: number;
  private readonly groupsY: number;

  private readonly pBuf: GPUBuffer;
  private readonly divBuf: GPUBuffer;
  private readonly labelBuf: GPUBuffer;
  private readonly readBuf: GPUBuffer;
  /** One per colour. Identical except for `color`; see the constructor. */
  private readonly paramsBuf: [GPUBuffer, GPUBuffer];
  private readonly bindGroup: [GPUBindGroup, GPUBindGroup];
  private readonly pipeline: GPUComputePipeline;

  private readonly querySet: GPUQuerySet | null = null;
  private readonly queryResolve: GPUBuffer | null = null;
  private readonly queryRead: GPUBuffer | null = null;

  // Rule 3: host staging allocated once, never in the frame loop.
  private readonly pF32: Float32Array<ArrayBuffer>;
  private readonly divF32: Float32Array<ArrayBuffer>;
  private readonly labelU32: Uint32Array<ArrayBuffer>;
  private readonly paramsData = new ArrayBuffer(PARAMS_BYTES);
  private readonly paramsU32 = new Uint32Array(this.paramsData);
  private readonly paramsF32 = new Float32Array(this.paramsData);

  /** Last frame's breakdown, for the overlay. */
  readonly timings: GpuTimings = { upload: 0, wait: 0, device: 0 };
  /**
   * Whether `timings.device` means anything. It is seeded to 0 rather than NaN
   * so the average can never be poisoned, which leaves 0 ambiguous between
   * "no timestamp-query on this adapter" and "immeasurably fast" — this flag
   * is what disambiguates them for the overlay.
   */
  hasDeviceTime = false;

  /** Guards against a second solve() starting while one is in flight — the
   *  readback buffer can only be mapped by one caller at a time. */
  private busy = false;

  constructor(
    private readonly ctx: GpuContext,
    g: Grid,
  ) {
    const { device } = ctx;
    this.device = device;
    this.cells = g.nx * g.ny;
    this.bytes = this.cells * 4;
    this.groupsX = Math.ceil(g.nx / WORKGROUP);
    this.groupsY = Math.ceil(g.ny / WORKGROUP);

    this.pF32 = new Float32Array(this.cells);
    this.divF32 = new Float32Array(this.cells);
    this.labelU32 = new Uint32Array(this.cells);

    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    // COPY_SRC on p only: it is the one buffer that comes back.
    this.pBuf = device.createBuffer({
      label: 'pressure',
      size: this.bytes,
      usage: storage | GPUBufferUsage.COPY_SRC,
    });
    this.divBuf = device.createBuffer({ label: 'divergence', size: this.bytes, usage: storage });
    this.labelBuf = device.createBuffer({ label: 'label', size: this.bytes, usage: storage });
    // MAP_READ cannot be combined with STORAGE, which is why the result is
    // copied into a second buffer rather than mapped in place.
    this.readBuf = device.createBuffer({
      label: 'pressure-readback',
      size: this.bytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const uniform = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
    this.paramsBuf = [
      device.createBuffer({ label: 'params-red', size: PARAMS_BYTES, usage: uniform }),
      device.createBuffer({ label: 'params-black', size: PARAMS_BYTES, usage: uniform }),
    ];

    this.pipeline = device.createComputePipeline({
      label: 'red-black-sor',
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: shaderSource }), entryPoint: 'main' },
    });

    // Two bind groups differing in ONE field. The alternative — rewriting the
    // colour into a single uniform between dispatches — does not work:
    // queue.writeBuffer is ordered against submits, not against commands
    // inside an encoder, so both dispatches would read whichever value landed
    // last. Two buffers is 64 bytes and removes the whole question.
    const layout = this.pipeline.getBindGroupLayout(0);
    const groupFor = (colour: 0 | 1): GPUBindGroup =>
      device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: { buffer: this.paramsBuf[colour] } },
          { binding: 1, resource: { buffer: this.pBuf } },
          { binding: 2, resource: { buffer: this.divBuf } },
          { binding: 3, resource: { buffer: this.labelBuf } },
        ],
      });
    this.bindGroup = [groupFor(0), groupFor(1)];

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

  /** Packs one colour's uniform block and uploads it. */
  private writeParams(colour: 0 | 1, nx: number, ny: number, scale: number, omega: number): void {
    this.paramsU32[0] = nx;
    this.paramsU32[1] = ny;
    this.paramsF32[2] = scale;
    this.paramsF32[3] = omega;
    this.paramsU32[4] = colour;
    this.device.queue.writeBuffer(this.paramsBuf[colour], 0, this.paramsData);
  }

  /**
   * `tol` is accepted for interface compatibility and IGNORED: an early exit
   * needs the residual, the residual needs a device-wide reduction and a
   * second readback, and that readback costs about as much as the sweeps it
   * would save. The browser runs a fixed sweep budget anyway (main.ts passes
   * tol = 0) for frame-pacing reasons, so nothing is lost today.
   */
  async solve(
    g: Grid,
    p: FieldArray,
    div: FieldArray,
    label: Uint8Array,
    scale: number,
    iterations: number,
    omega: number,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars -- see the doc comment
    tol: number,
  ): Promise<number> {
    if (this.busy) throw new Error('GpuPressureSolver.solve() is not re-entrant');
    if (this.ctx.lost) throw new Error('GPU device lost');

    // Same short-circuit as the CPU solvers, and it must stay: subtractGradient
    // applies p unconditionally, so a warm-started leftover on an already
    // divergence-free field would re-inject divergence rather than remove it.
    if (fluidDivRms(div, label) === 0) {
      p.fill(0);
      return 0;
    }
    if (iterations <= 0) return 0;

    this.busy = true;
    try {
      const t0 = performance.now();
      // Uploading p every frame IS the warm start — p holds last frame's
      // solution, so the device needs no state of its own between frames. A
      // resident p would save ~0.2 ms and would have to be kept in sync with
      // every CPU-side write (reset, commitLabels); not worth it until the
      // whole step lives on the device and the question disappears.
      this.pF32.set(p);
      this.divF32.set(div);
      this.labelU32.set(label);
      const { queue } = this.device;
      queue.writeBuffer(this.pBuf, 0, this.pF32);
      queue.writeBuffer(this.divBuf, 0, this.divF32);
      queue.writeBuffer(this.labelBuf, 0, this.labelU32);
      this.writeParams(0, g.nx, g.ny, scale, omega);
      this.writeParams(1, g.nx, g.ny, scale, omega);
      const t1 = performance.now();

      const encoder = this.device.createCommandEncoder({ label: 'pressure-solve' });
      const pass = encoder.beginComputePass({
        label: 'red-black',
        timestampWrites: this.querySet
          ? { querySet: this.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 }
          : undefined,
      });
      pass.setPipeline(this.pipeline);
      // One sweep = red pass then black pass. Every dispatch is recorded into
      // the same command buffer and submitted once, so the loop costs no
      // CPU/GPU round trips at all.
      for (let k = 0; k < iterations; k++) {
        pass.setBindGroup(0, this.bindGroup[0]);
        pass.dispatchWorkgroups(this.groupsX, this.groupsY);
        pass.setBindGroup(0, this.bindGroup[1]);
        pass.dispatchWorkgroups(this.groupsX, this.groupsY);
      }
      pass.end();

      encoder.copyBufferToBuffer(this.pBuf, 0, this.readBuf, 0, this.bytes);
      if (this.querySet && this.queryResolve && this.queryRead) {
        encoder.resolveQuerySet(this.querySet, 0, 2, this.queryResolve, 0);
        encoder.copyBufferToBuffer(this.queryResolve, 0, this.queryRead, 0, 16);
      }
      // submit() returns immediately — it queues work, it does not run it.
      queue.submit([encoder.finish()]);

      await this.readBuf.mapAsync(GPUMapMode.READ);
      // Copy out BEFORE unmap: the mapped range is detached by unmap(), and a
      // view onto it becomes zero-length rather than throwing.
      p.set(new Float32Array(this.readBuf.getMappedRange()));
      this.readBuf.unmap();

      this.timings.upload = smooth(this.timings.upload, t1 - t0);
      this.timings.wait = smooth(this.timings.wait, performance.now() - t1);
      const device = await this.readDeviceTime();
      this.hasDeviceTime = Number.isFinite(device);
      this.timings.device = smooth(this.timings.device, device);
      return iterations;
    } finally {
      this.busy = false;
    }
  }

  /** Device-side pass duration in ms, or NaN if timestamps are unavailable or
   *  the driver reported a zero/decreasing pair (some do). */
  private async readDeviceTime(): Promise<number> {
    if (!this.queryRead) return NaN;
    await this.queryRead.mapAsync(GPUMapMode.READ);
    const ns = new BigUint64Array(this.queryRead.getMappedRange());
    const delta = ns[1] - ns[0];
    this.queryRead.unmap();
    return delta > 0n ? Number(delta) / 1e6 : NaN;
  }
}
