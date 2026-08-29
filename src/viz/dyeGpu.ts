import type { Grid } from '../core/grid.ts';
import type { GpuContext } from '../gpu/device.ts';
import { paletteCode, type DyePalette } from './colormaps.ts';
import dyeSource from './dye.wgsl?raw';

/**
 * Draws the dye view straight from the buffer the fused step leaves it in.
 *
 * This is the half of dye residency that faces the screen — gpu/stepGpu.ts is
 * the half that faces the solver. Together they are what removes the per-frame
 * round trip: the source kernels mean the host no longer has to write the dye,
 * and this means it no longer has to read it.
 *
 * It owns a canvas of its own rather than sharing the 2D one, because a canvas
 * has exactly one context for its lifetime and the other three views are still
 * drawn with '2d'. The two are stacked: this one at the back, the 2D one on
 * top and cleared to transparent while the dye view is up, so the solids mask
 * and the arrows keep working unchanged (see FieldView.draw's `dyeOnDevice`).
 *
 * Sized in DEVICE pixels with no cap, which the 2D canvas deliberately has —
 * there the backing store is upscaled from one pixel per cell, so oversampling
 * bought nothing and cost a bigger drawImage. Here the source can genuinely be
 * at display resolution, and that 1:1 mapping is most of what refining the dye
 * grid is for.
 */

/** nx, ny, w, h, palette, and three words of padding: a uniform struct's size
 *  rounds up to 16 bytes, so naming the slack beats letting the layout add
 *  it. */
const PARAMS_BYTES = 32;

export class DyeRenderer {
  private readonly device: GPUDevice;
  private readonly gpu: GPUCanvasContext;
  private readonly pipeline: GPURenderPipeline;
  private readonly group: GPUBindGroup;
  private readonly paramsBuf: GPUBuffer;
  private readonly paramsData = new Uint32Array(PARAMS_BYTES / 4);

  constructor(
    ctx: GpuContext,
    private readonly canvas: HTMLCanvasElement,
    private readonly dg: Grid,
    dyeBuf: GPUBuffer,
  ) {
    const { device } = ctx;
    this.device = device;

    const gpu = canvas.getContext('webgpu');
    if (!gpu) throw new Error('webgpu canvas context unavailable');
    this.gpu = gpu;
    const format = navigator.gpu.getPreferredCanvasFormat();
    // 'opaque': the fragment shader always writes alpha 1, and telling the
    // compositor so saves it a blend against the page behind.
    gpu.configure({ device, format, alphaMode: 'opaque' });

    this.paramsBuf = device.createBuffer({
      label: 'dye-view-params',
      size: PARAMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const module = device.createShaderModule({ label: 'dye-view', code: dyeSource });
    this.pipeline = device.createRenderPipeline({
      label: 'dye-view',
      layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    });
    this.group = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuf } },
        { binding: 1, resource: { buffer: dyeBuf } },
      ],
    });
  }

  /** Matches the backing store to the window at full device resolution. Safe
   *  to call every frame: assigning the same width is a no-op, and assigning a
   *  different one is what a resize needs. */
  fit(): void {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(window.innerWidth * dpr));
    const h = Math.max(1, Math.round(window.innerHeight * dpr));
    if (this.canvas.width !== w) this.canvas.width = w;
    if (this.canvas.height !== h) this.canvas.height = h;
  }

  /** `palette` is the SCENE's, not a display preference: it says what the
   *  three channels mean — see viz/colormaps.ts. */
  draw(palette: DyePalette = 'rgb'): void {
    this.fit();
    this.paramsData[0] = this.dg.nx;
    this.paramsData[1] = this.dg.ny;
    this.paramsData[2] = this.canvas.width;
    this.paramsData[3] = this.canvas.height;
    this.paramsData[4] = paletteCode(palette);
    this.device.queue.writeBuffer(this.paramsBuf, 0, this.paramsData);

    const encoder = this.device.createCommandEncoder({ label: 'dye-view' });
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.gpu.getCurrentTexture().createView(),
          // 'clear' rather than 'load': the triangle covers every pixel, so
          // there is nothing to preserve and clearing is the cheaper start.
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        },
      ],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.group);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  destroy(): void {
    this.paramsBuf.destroy();
    // Releases the swap chain; the canvas element itself is reused.
    this.gpu.unconfigure();
  }
}
