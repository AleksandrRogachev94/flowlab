/**
 * WebGPU device acquisition, and nothing else.
 *
 * Separate from the solver because device setup is the part with all the ways
 * to fail — no browser support, no adapter, a rejected device request, a
 * device lost mid-run — and none of them should be tangled up with the
 * numerics. Every failure resolves to `null` rather than throwing, because the
 * only sane response is "keep running on the CPU", which the caller can do.
 */

/** Everything downstream needs, plus what the perf overlay wants to report. */
export interface GpuContext {
  device: GPUDevice;
  /** Vendor/architecture strings if the browser will say; often blank. */
  info: GPUAdapterInfo | null;
  /** Whether device-side timestamps are available — see pressureGpu.ts. */
  timestamps: boolean;
  /** Set if the device is lost; the caller should fall back to the CPU. */
  lost: boolean;
}

/**
 * Returns a ready device, or null with the reason logged.
 *
 * `timestamp-query` is requested opportunistically: it is what separates "the
 * GPU spent 3 ms" from "we waited 12 ms", and without it the overlay can only
 * report wall clock, which on the GPU path is mostly synchronisation. Chrome
 * exposes it by default on most hardware but not all, so it is optional
 * rather than required — asking for an unsupported feature makes
 * requestDevice reject outright.
 */
export async function initGpu(): Promise<GpuContext | null> {
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    console.warn(
      '[gpu] navigator.gpu missing — WebGPU needs a secure context and a supporting browser',
    );
    return null;
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    console.warn('[gpu] no compatible adapter');
    return null;
  }

  const timestamps = adapter.features.has('timestamp-query');
  const device = await adapter.requestDevice({
    requiredFeatures: timestamps ? ['timestamp-query'] : [],
  });

  const ctx: GpuContext = {
    device,
    info: adapter.info ?? null,
    timestamps,
    lost: false,
  };

  // A lost device is permanent — every subsequent call is a no-op that
  // silently produces nothing. Flagging it lets the caller switch back to the
  // CPU solver instead of rendering a frozen picture and blaming the physics.
  void device.lost.then((reason) => {
    ctx.lost = true;
    console.error(`[gpu] device lost: ${reason.reason} — ${reason.message}`);
  });

  // Without this, a bad shader or a mis-sized buffer surfaces as a console
  // warning at best and a wrong picture at worst. Uncaptured errors are the
  // main debugging channel WGSL has; PLAN.md §Phase 2 warns about exactly this.
  device.addEventListener('uncapturederror', (e) => {
    console.error('[gpu] uncaptured error:', e.error.message);
  });

  return ctx;
}

/** One-line description for the readout, e.g. "apple / metal-3". */
export function describeGpu(ctx: GpuContext): string {
  const parts = [ctx.info?.vendor, ctx.info?.architecture].filter(Boolean);
  return parts.length > 0 ? parts.join(' / ') : 'webgpu';
}
