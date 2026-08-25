/**
 * Sequential phase timer for one step() / one frame.
 *
 * Deliberately not a general tree profiler. Every phase in this program runs
 * once, in a fixed order, one after another — so "time since the previous
 * mark" IS the phase duration, and a flat ordered list is the whole data
 * model. No stack, no nesting, no ids.
 *
 * Rule 1 safe: `performance.now()` is a Node global too, so this stays usable
 * from headless benchmarks and tests, not just the browser.
 *
 * Rule 3 safe: the arrays grow on the first frame and are reused forever
 * after. `mark()` allocates nothing once the labels are known.
 *
 * Always on. A handful of `performance.now()` calls cost tens of nanoseconds
 * against a step measured in milliseconds, and a profiler you have to enable
 * is a profiler that is off when you need it.
 */

/**
 * EMA weight on the newest sample. 0.12 gives a ~8-frame time constant: fast
 * enough to show a scene change, slow enough that the digits stay readable
 * instead of flickering every frame.
 */
const ALPHA = 0.12;

/**
 * The project's one definition of "smoothed". Everything that displays a
 * per-frame duration goes through this — the phase table below, the GPU
 * solver's own breakdown, the overlay's frame rate — so every number on
 * screen settles at the same speed. Three copies of `avg + a * (x - avg)`
 * with three different constants is how a panel ends up with one column
 * lagging another for no visible reason.
 *
 * Non-finite samples are ignored rather than propagated: a NaN reaches this
 * whenever the GPU cannot supply a timestamp, and one NaN would otherwise
 * poison the average permanently.
 */
export function smooth(avg: number, sample: number): number {
  return Number.isFinite(sample) ? avg + ALPHA * (sample - avg) : avg;
}

/**
 * Frames a peak survives before it starts decaying away. Long enough that a
 * single stutter stays on screen to be read, short enough that it clears once
 * the cause is fixed.
 */
const PEAK_DECAY = 0.98;

export class Profiler {
  /** Phase names, in execution order. Stable after the first frame. */
  readonly labels: string[] = [];
  /** Exponential moving average, ms — the number worth reading. */
  readonly ema: number[] = [];
  /** Slowly-decaying high-water mark, ms. Catches stutter the EMA hides. */
  readonly peak: number[] = [];

  private cursor = 0;
  private clock = 0;

  /** Starts the clock and rewinds to phase 0. Call once per step/frame. */
  begin(): void {
    this.cursor = 0;
    this.clock = performance.now();
  }

  /**
   * Closes the phase that started at the previous mark (or at begin()) and
   * names it. Phase order is fixed, so the label is just assigned to the slot
   * rather than looked up — a mismatch would be a code change, not a miss.
   */
  mark(label: string): void {
    const now = performance.now();
    const dt = now - this.clock;
    this.clock = now;

    const k = this.cursor++;
    if (k === this.labels.length) {
      // A new slot is seeded with the sample itself; averaging up from 0 would
      // take ~30 frames to stop lying.
      this.labels.push(label);
      this.ema.push(dt);
      this.peak.push(dt);
      return;
    }
    this.labels[k] = label;
    this.ema[k] = smooth(this.ema[k], dt);
    // Floored at the average: a decaying high-water mark can otherwise dip
    // BELOW the EMA while timings fall, and a "peak" column reading less than
    // the "avg" column beside it makes the whole panel look broken.
    this.peak[k] = Math.max(dt, this.ema[k], this.peak[k] * PEAK_DECAY);
  }

  /** Sum of the EMAs — the smoothed frame cost the bars are drawn against. */
  totalEma(): number {
    let sum = 0;
    for (let k = 0; k < this.cursor; k++) sum += this.ema[k];
    return sum;
  }

  /** Number of phases in the last completed frame. */
  get count(): number {
    return this.cursor;
  }

  /** Fresh start — after a resolution change, say, where old numbers mislead. */
  reset(): void {
    this.labels.length = 0;
    this.ema.length = 0;
    this.peak.length = 0;
    this.cursor = 0;
  }
}
