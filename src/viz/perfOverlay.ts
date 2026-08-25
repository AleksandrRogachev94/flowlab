import { smooth, type Profiler } from '../core/profiler.ts';

/**
 * Per-phase timing panel: a <pre> that gets a string written into it.
 *
 * It measures exactly one thing itself — the frame rate, which is the interval
 * between its own update() calls and which nothing else is positioned to see.
 * Everything else arrives already measured and already smoothed, from
 * core/profiler.ts and from the GPU solver, so the headless benchmark
 * (src/bench/step.ts) and this panel report the same numbers rather than being
 * two different definitions of "a frame".
 *
 * Formatting lives here rather than in main.ts on purpose: the caller knows
 * WHAT it has, the panel knows how it should READ. Handing the panel a
 * pre-formatted string is how column headers end up missing and units end up
 * implied.
 *
 * The bars are text (block characters), which is why this needs no layout code
 * at all: a monospace <pre> already aligns columns, so the whole panel is one
 * template literal per repaint.
 */

/** Repaints per second. Digits that change 60 times a second are unreadable,
 *  and an overlay costing 2 ms while reporting a 30 ms budget measures itself. */
const PAINT_HZ = 6;

/** Column widths, in characters. Named so the header row and the value rows
 *  cannot drift apart — that drift is exactly what makes a panel unreadable. */
const W_NAME = 14;
const W_MS = 7;
const W_BAR = 14;
const W_PCT = 5;
const W_PEAK = 10;

/** The GPU solver's own breakdown. Structurally typed rather than imported, so
 *  viz/ keeps not depending on gpu/. */
export interface GpuPanel {
  /** Host-side f64 -> f32 conversion plus writeBuffer, ms. */
  upload: number;
  /** Awaiting mapAsync, ms. Contains the compute — the GPU only finishes
   *  once we wait — so `device` is a part of this, not an addition to it. */
  wait: number;
  /** Time inside the compute pass per the device's own timestamps, ms. */
  device: number;
  /** Whether `device` is real. See GpuPressureSolver.hasDeviceTime. */
  hasDeviceTime: boolean;
}

export interface PerfUpdate {
  /** The step's phase breakdown — what the bar chart shows. */
  phases: Profiler;
  /** Coarser timings the phases cannot see, e.g. the draw. */
  totals: Profiler;
  /** Which PressureSolver is running, by name. */
  solver: string;
  /** GPU adapter identity, or why there isn't one. NOT a measurement. */
  adapter: string;
  /** Present only while the GPU solver is the one running. */
  gpu?: GpuPanel;
}

const ms = (v: number, width = W_MS): string => v.toFixed(2).padStart(width);

/**
 * Renders the panel. Pure and exported so the layout can be eyeballed from
 * Node without a browser — the alignment is the whole point of the thing and
 * "looks right in my head" is not a check.
 */
export function formatPanel(u: PerfUpdate, frameMs: number): string {
  const { phases, totals } = u;
  const stepMs = phases.totalEma();
  const lines: string[] = [];

  // Line 1 is identity, not timing: which solver, and on what. `adapter` is a
  // vendor/architecture string from the driver — it says WHERE the work ran,
  // and carries no duration at all. It sits apart from every number below.
  lines.push(
    `solver ${u.solver.padEnd(12)} adapter ${u.adapter}`,
    '',
    // Every column is named and carries its unit. The bar and the percentage
    // are one column conceptually — both answer "share of step" — so they get
    // one heading spanning both.
    'phase'.padEnd(W_NAME) +
      'avg ms'.padStart(W_MS) +
      '  ' +
      'share of step'.padEnd(W_BAR + W_PCT) +
      'peak ms'.padStart(W_PEAK),
  );

  for (let k = 0; k < phases.count; k++) {
    const avg = phases.ema[k];
    const pct = stepMs > 0 ? (100 * avg) / stepMs : 0;
    const filled = Math.round((pct / 100) * W_BAR);
    lines.push(
      phases.labels[k].padEnd(W_NAME) +
        ms(avg) +
        '  ' +
        '█'.repeat(filled) +
        '░'.repeat(W_BAR - filled) +
        `${pct.toFixed(0)}%`.padStart(W_PCT) +
        // The decaying high-water mark. Only worth reading when it diverges
        // from the average — which is exactly when there is stutter.
        ms(phases.peak[k], W_PEAK),
    );
  }

  lines.push('');
  lines.push(`${'step'.padEnd(W_NAME)}${ms(stepMs)} ms   (the phases above)`);
  for (let k = 0; k < totals.count; k++) {
    lines.push(`${totals.labels[k].padEnd(W_NAME)}${ms(totals.ema[k])} ms`);
  }
  const fps = frameMs > 0 ? 1000 / frameMs : 0;
  lines.push(`${'frame'.padEnd(W_NAME)}${ms(frameMs)} ms   ${fps.toFixed(1)} fps`);

  if (u.gpu) {
    const { upload, wait, device, hasDeviceTime } = u.gpu;
    lines.push('');
    lines.push('pressure solve on the GPU, ms:');
    // Decomposed so the parts ADD UP and none of them overlap. `wait` contains
    // the compute, so showing it next to `device` invites reading one as extra
    // cost on top of the other; splitting out the remainder is the number that
    // actually decides whether porting the next kernel is worth it.
    lines.push(
      hasDeviceTime
        ? `  upload ${upload.toFixed(2)}  +  on-device ${device.toFixed(2)}  ` +
            `+  round trip ${Math.max(0, wait - device).toFixed(2)}  =  ${(upload + wait).toFixed(2)} total`
        : `  upload ${upload.toFixed(2)}  +  wait ${wait.toFixed(2)}  =  ${(upload + wait).toFixed(2)} total` +
            `\n  (no timestamp-query on this adapter, so the wait cannot be split)`,
    );
  }

  return lines.join('\n');
}

export class PerfOverlay {
  private readonly el: HTMLPreElement;
  private lastPaint = 0;
  private lastFrame = 0;
  /** Smoothed wall time between update() calls — the only thing measured here. */
  private frameMs = 0;

  constructor(host: HTMLElement) {
    this.el = document.createElement('pre');
    this.el.className = 'perf';
    this.el.hidden = true;
    host.append(this.el);
  }

  toggle(): void {
    this.el.hidden = !this.el.hidden;
  }

  /** Call every frame; it decides for itself whether to repaint. */
  update(u: PerfUpdate): void {
    // Before the hidden check, so the frame rate is already warm the moment
    // the panel is switched on rather than climbing from zero for a second.
    const now = performance.now();
    if (this.lastFrame > 0) this.frameMs = smooth(this.frameMs, now - this.lastFrame);
    this.lastFrame = now;

    if (this.el.hidden) return;
    if (now - this.lastPaint < 1000 / PAINT_HZ) return;
    this.lastPaint = now;
    this.el.textContent = formatPanel(u, this.frameMs);
  }
}
