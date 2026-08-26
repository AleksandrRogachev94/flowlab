import type { Advector } from '../core/advector.ts';
import type { AdvectionScheme } from '../core/advect.ts';
import { DYE_CHANNELS, type FieldArray, type Grid } from '../core/grid.ts';
import type { GpuContext } from './device.ts';
import shaderSource from './advect.wgsl?raw';

/**
 * Semi-Lagrangian advection on the device, behind the Advector seam.
 * The kernels and the design are in advect.wgsl and docs/WEBGPU.md §8; this
 * file is the plumbing.
 *
 * SHAPE OF A STEP. Two submits, each ending in one readback:
 *
 *   velocity()  upload u, v, label -> 2 (or 4) dispatches -> read u, v back
 *   dye()       upload u, v, dye   -> 1 (or 2) dispatches -> read dye back
 *
 * The uploads in dye() are not waste: subtractGradient ran on the HOST in
 * between and changed u and v, so the device's copy is stale by then. `label`
 * is the exception — nothing between the two calls can touch it, so dye()
 * reuses the buffer velocity() filled. That is the Advector interface's
 * ordering contract, and it is worth a comment because it is the only piece
 * of shared state whose freshness is not obvious from this file alone.
 *
 * Both round trips disappear together, not one at a time: they exist because
 * divergence, the projection and the dye source still run on the host. That
 * port now exists — stepGpu.ts records recordVelocity() and recordDye() into
 * ITS pass, against these same buffers, and never calls velocity()/dye() at
 * all. The methods here remain the standalone path: what runs when only
 * advection is on the GPU, and what the gputests pin against the CPU. The
 * buffers and staging arrays are non-private for exactly that composition;
 * nothing else reads them.
 */

/** Threads per workgroup, one axis. MUST match @workgroup_size in advect.wgsl. */
const WORKGROUP = 8;

/** nx, ny, h, dt — exactly 16 bytes, so no padding is needed. */
const PARAMS_BYTES = 16;

/** Entry points, in the order a MacCormack step dispatches them. */
const ENTRY_POINTS = [
  'advect_u',
  'advect_v',
  'advect_dye',
  'correct_u',
  'correct_v',
  'correct_dye',
] as const;
type EntryPoint = (typeof ENTRY_POINTS)[number];

const ceilDiv = (a: number, b: number): number => Math.ceil(a / b);

export class GpuAdvector implements Advector {
  readonly name = 'gpu';

  private readonly device: GPUDevice;
  private readonly nx: number;
  private readonly ny: number;
  private readonly h: number;
  readonly uLen: number;
  readonly vLen: number;
  readonly dyeLen: number;
  readonly uBytes: number;
  readonly vBytes: number;

  // Inputs: refilled from the host every step on the standalone path. Under
  // the fused stepper they hold the RESIDENT velocity instead — see stepGpu.ts.
  readonly uIn: GPUBuffer;
  readonly vIn: GPUBuffer;
  readonly dyeIn: GPUBuffer;
  readonly labelBuf: GPUBuffer;
  private readonly paramsBuf: GPUBuffer;
  // Outputs. `A` takes the forward pass, `B` the MacCormack correction — so
  // the result is in A for semiLagrangian and in B for macCormack, and that
  // is the only thing the scheme changes on this side.
  readonly uA: GPUBuffer;
  readonly uB: GPUBuffer;
  readonly vA: GPUBuffer;
  readonly vB: GPUBuffer;
  readonly dyeA: GPUBuffer;
  readonly dyeB: GPUBuffer;
  // MAP_READ cannot be combined with STORAGE, so results are copied here.
  // One buffer for u and v together: two mapAsync calls would be two stalls.
  readonly velRead: GPUBuffer;
  readonly dyeRead: GPUBuffer;

  private readonly pipelines: Record<EntryPoint, GPUComputePipeline>;
  /** One per (src, orig, dst) triple the six kernels need. */
  private readonly bind: Record<'uFwd' | 'uCor' | 'vFwd' | 'vCor' | 'dFwd' | 'dCor', GPUBindGroup>;

  // Host staging, allocated once (Rule 3). These also do the f64 -> f32
  // conversion, which is why they are typed arrays and not raw buffers.
  readonly uHost: Float32Array<ArrayBuffer>;
  readonly vHost: Float32Array<ArrayBuffer>;
  readonly dyeHost: Float32Array<ArrayBuffer>;
  readonly labelHost: Uint32Array<ArrayBuffer>;
  private readonly paramsData = new ArrayBuffer(PARAMS_BYTES);
  private readonly paramsU32 = new Uint32Array(this.paramsData);
  private readonly paramsF32 = new Float32Array(this.paramsData);

  /**
   * dye() reuses the label buffer velocity() filled — see the class comment.
   * This covers the one case where there is nothing to reuse: an advector
   * installed part-way through a step, whose first dye() would otherwise read
   * an all-zero (i.e. all-Fluid) buffer and carry dye through the obstacle.
   */
  private labelsUploaded = false;

  constructor(
    private readonly ctx: GpuContext,
    g: Grid,
  ) {
    const { device } = ctx;
    this.device = device;
    this.nx = g.nx;
    this.ny = g.ny;
    this.h = g.h;
    const cells = g.nx * g.ny;
    this.uLen = (g.nx + 1) * g.ny;
    this.vLen = g.nx * (g.ny + 1);
    this.dyeLen = DYE_CHANNELS * cells;
    this.uBytes = this.uLen * 4;
    this.vBytes = this.vLen * 4;

    this.uHost = new Float32Array(this.uLen);
    this.vHost = new Float32Array(this.vLen);
    this.dyeHost = new Float32Array(this.dyeLen);
    this.labelHost = new Uint32Array(cells);

    const input = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const output = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
    const buffer = (label: string, length: number, usage: number): GPUBuffer =>
      device.createBuffer({ label, size: length * 4, usage });

    // COPY_SRC on the velocity inputs is for the fused stepper alone: it holds
    // the projected field in these buffers and reads THEM back, not A/B.
    this.uIn = buffer('u-in', this.uLen, input | GPUBufferUsage.COPY_SRC);
    this.vIn = buffer('v-in', this.vLen, input | GPUBufferUsage.COPY_SRC);
    this.dyeIn = buffer('dye-in', this.dyeLen, input);
    this.labelBuf = buffer('label', cells, input);
    this.uA = buffer('u-hat', this.uLen, output);
    this.uB = buffer('u-out', this.uLen, output);
    this.vA = buffer('v-hat', this.vLen, output);
    this.vB = buffer('v-out', this.vLen, output);
    this.dyeA = buffer('dye-hat', this.dyeLen, output);
    this.dyeB = buffer('dye-out', this.dyeLen, output);

    const readback = GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST;
    this.velRead = buffer('velocity-readback', this.uLen + this.vLen, readback);
    this.dyeRead = buffer('dye-readback', this.dyeLen, readback);
    this.paramsBuf = device.createBuffer({
      label: 'advect-params',
      size: PARAMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // ONE explicit layout for all six pipelines, rather than `layout: 'auto'`
    // per pipeline. Auto-layouts are distinct objects even when identical, so
    // a bind group built for one pipeline is rejected by the next — and the
    // whole point here is that every kernel takes the same six bindings and
    // only the buffers behind them change.
    const storage = (binding: number, readOnly: boolean): GPUBindGroupLayoutEntry => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: readOnly ? 'read-only-storage' : 'storage' },
    });
    const layout = device.createBindGroupLayout({
      label: 'advect-bindings',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform' },
        },
        storage(1, true), // label
        storage(2, true), // carU
        storage(3, true), // carV
        storage(4, true), // src
        storage(5, true), // orig
        storage(6, false), // dst
      ],
    });

    const module = device.createShaderModule({ label: 'advect', code: shaderSource });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    this.pipelines = Object.fromEntries(
      ENTRY_POINTS.map((entryPoint) => [
        entryPoint,
        device.createComputePipeline({
          label: entryPoint,
          layout: pipelineLayout,
          compute: { module, entryPoint },
        }),
      ]),
    ) as Record<EntryPoint, GPUComputePipeline>;

    // The carrier (u^n for velocity, the projected u for dye) is uIn/vIn in
    // every group — only what is sampled and where it lands differ. `orig` is
    // read by the correct_* kernels alone; the forward passes point it at
    // their own source and ignore it, which keeps one layout for all six.
    const group = (name: string, src: GPUBuffer, orig: GPUBuffer, dst: GPUBuffer): GPUBindGroup =>
      device.createBindGroup({
        label: name,
        layout,
        entries: [
          { binding: 0, resource: { buffer: this.paramsBuf } },
          { binding: 1, resource: { buffer: this.labelBuf } },
          { binding: 2, resource: { buffer: this.uIn } },
          { binding: 3, resource: { buffer: this.vIn } },
          { binding: 4, resource: { buffer: src } },
          { binding: 5, resource: { buffer: orig } },
          { binding: 6, resource: { buffer: dst } },
        ],
      });
    this.bind = {
      uFwd: group('u-forward', this.uIn, this.uIn, this.uA),
      uCor: group('u-correct', this.uA, this.uIn, this.uB),
      vFwd: group('v-forward', this.vIn, this.vIn, this.vA),
      vCor: group('v-correct', this.vA, this.vIn, this.vB),
      dFwd: group('dye-forward', this.dyeIn, this.dyeIn, this.dyeA),
      dCor: group('dye-correct', this.dyeA, this.dyeIn, this.dyeB),
    };
  }

  async velocity(
    g: Grid,
    scheme: AdvectionScheme,
    uIn: FieldArray,
    vIn: FieldArray,
    uOut: FieldArray,
    vOut: FieldArray,
    label: Uint8Array,
    dt: number,
  ): Promise<void> {
    if (this.ctx.lost) throw new Error('GPU device lost');
    const { queue } = this.device;
    this.uHost.set(uIn);
    this.vHost.set(vIn);
    this.labelHost.set(label);
    queue.writeBuffer(this.uIn, 0, this.uHost);
    queue.writeBuffer(this.vIn, 0, this.vHost);
    queue.writeBuffer(this.labelBuf, 0, this.labelHost);
    this.labelsUploaded = true;
    this.writeParams(dt);

    const mac = scheme === 'macCormack';
    const encoder = this.device.createCommandEncoder({ label: 'advect-velocity' });
    const pass = encoder.beginComputePass({ label: 'advect-velocity' });
    this.recordVelocity(pass, mac);
    pass.end();
    encoder.copyBufferToBuffer(mac ? this.uB : this.uA, 0, this.velRead, 0, this.uBytes);
    encoder.copyBufferToBuffer(mac ? this.vB : this.vA, 0, this.velRead, this.uBytes, this.vBytes);
    queue.submit([encoder.finish()]);

    await this.velRead.mapAsync(GPUMapMode.READ);
    // Copy out BEFORE unmap: unmap() detaches the range and a view onto it
    // silently becomes zero-length rather than throwing.
    const mapped = this.velRead.getMappedRange();
    uOut.set(new Float32Array(mapped, 0, this.uLen));
    vOut.set(new Float32Array(mapped, this.uBytes, this.vLen));
    this.velRead.unmap();
  }

  async dye(
    g: Grid,
    scheme: AdvectionScheme,
    u: FieldArray,
    v: FieldArray,
    qIn: FieldArray[],
    qOut: FieldArray[],
    label: Uint8Array,
    dt: number,
  ): Promise<void> {
    if (this.ctx.lost) throw new Error('GPU device lost');
    const { queue } = this.device;
    // u and v again, because the host projected them since velocity() ran.
    // `label` is NOT re-uploaded — see the class comment and labelsUploaded.
    if (!this.labelsUploaded) {
      this.labelHost.set(label);
      queue.writeBuffer(this.labelBuf, 0, this.labelHost);
      this.labelsUploaded = true;
    }
    this.uHost.set(u);
    this.vHost.set(v);
    for (let c = 0; c < qIn.length; c++) this.dyeHost.set(qIn[c], c * this.nx * this.ny);
    queue.writeBuffer(this.uIn, 0, this.uHost);
    queue.writeBuffer(this.vIn, 0, this.vHost);
    queue.writeBuffer(this.dyeIn, 0, this.dyeHost);
    this.writeParams(dt);

    const mac = scheme === 'macCormack';
    const encoder = this.device.createCommandEncoder({ label: 'advect-dye' });
    const pass = encoder.beginComputePass({ label: 'advect-dye' });
    this.recordDye(pass, mac);
    pass.end();
    encoder.copyBufferToBuffer(mac ? this.dyeB : this.dyeA, 0, this.dyeRead, 0, this.dyeLen * 4);
    queue.submit([encoder.finish()]);

    await this.dyeRead.mapAsync(GPUMapMode.READ);
    const mapped = this.dyeRead.getMappedRange();
    const cells = this.nx * this.ny;
    for (let c = 0; c < qOut.length; c++) {
      qOut[c].set(new Float32Array(mapped, c * cells * 4, cells));
    }
    this.dyeRead.unmap();
  }

  /**
   * The velocity dispatches alone, into a pass the CALLER owns — the seam the
   * fused stepper composes over. u and v are independent gathers over the same
   * read-only inputs, so the order of the two forward passes is free; the
   * correction must follow both, and the barrier that guarantees it is
   * inserted by WebGPU between dispatches. Result lands in uA/vA
   * (semiLagrangian) or uB/vB (macCormack).
   */
  recordVelocity(pass: GPUComputePassEncoder, mac: boolean): void {
    this.dispatch(pass, 'advect_u', this.bind.uFwd, this.nx + 1, this.ny);
    this.dispatch(pass, 'advect_v', this.bind.vFwd, this.nx, this.ny + 1);
    if (mac) {
      this.dispatch(pass, 'correct_u', this.bind.uCor, this.nx + 1, this.ny);
      this.dispatch(pass, 'correct_v', this.bind.vCor, this.nx, this.ny + 1);
    }
  }

  /** The dye dispatches — one for all three channels: they share the
   *  backtrace. Result in dyeA (semiLagrangian) or dyeB (macCormack). */
  recordDye(pass: GPUComputePassEncoder, mac: boolean): void {
    this.dispatch(pass, 'advect_dye', this.bind.dFwd, this.nx, this.ny);
    if (mac) this.dispatch(pass, 'correct_dye', this.bind.dCor, this.nx, this.ny);
  }

  /** Bounds-guarded in the shader, so overshooting the field is safe. */
  private dispatch(
    pass: GPUComputePassEncoder,
    entryPoint: EntryPoint,
    group: GPUBindGroup,
    width: number,
    height: number,
  ): void {
    pass.setPipeline(this.pipelines[entryPoint]);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(ceilDiv(width, WORKGROUP), ceilDiv(height, WORKGROUP));
  }

  writeParams(dt: number): void {
    this.paramsU32[0] = this.nx;
    this.paramsU32[1] = this.ny;
    this.paramsF32[2] = this.h;
    this.paramsF32[3] = dt;
    this.device.queue.writeBuffer(this.paramsBuf, 0, this.paramsData);
  }
}
