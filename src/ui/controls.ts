/**
 * The demo's chrome: a scene switcher, a settings panel and a status line,
 * floating over a full-bleed canvas.
 *
 * ONE RULE, and everything here follows from it: this module never owns state.
 * Widgets report a (key, value) pair to `onSet` and main.ts decides what that
 * means; `sync()` then pushes the resulting state back into the DOM. Keyboard
 * shortcuts go through the same `onSet`, so a key and its widget cannot drift
 * apart — which is the failure mode the old parallel button/key tables had.
 *
 * It is also deliberately option-list DRIVEN rather than hand-written markup.
 * The dye choices depend on the scene and the solver choices on the engine
 * (scenes/catalog.ts explains why), so those lists change at runtime; building
 * every control from an Option[] means "the dropdown follows the scene" is one
 * call and not a special case.
 */

import type { AdvectionScheme } from '../core/advect.ts';
import type { View } from '../viz/fieldView.ts';

export type Engine = 'gpu' | 'cpu';
export type Quality = 'low' | 'medium' | 'high' | 'ultra';

/** Everything the UI can change. main.ts owns the instance. */
export interface UiState {
  scene: string;
  dye: string;
  view: View;
  arrows: boolean;
  engine: Engine;
  solver: string;
  scheme: AdvectionScheme;
  quality: Quality;
  perf: boolean;
  diagnostics: boolean;
}

export type UiKey = keyof UiState;

export interface Option {
  value: string;
  label: string;
}

export interface SceneOption extends Option {
  /** One sentence under the scene bar, for a viewer who has never heard of a
   *  vortex street. */
  blurb: string;
  /** The brush experiment worth trying here — see Scene.hint. Rendered
   *  dimmer and on its own line, because it is an invitation rather than a
   *  description and should not be mistaken for part of the blurb. */
  hint?: string;
}

/** The lists every control is built from. main.ts supplies them because it is
 *  what knows the grid sizes and the solver objects behind the labels. */
export interface ControlsSpec {
  scenes: SceneOption[];
  dyes: Option[];
  views: Option[];
  engines: Option[];
  solvers: Option[];
  schemes: Option[];
  qualities: Option[];
}

/**
 * The two brush gestures, stated once rather than folded into every scene's
 * hint. Every scene supports both — stirring and drawing are properties of
 * the brush, not of the scene — so repeating "drag to stir, shift-drag to
 * draw" six times would be six copies of the same sentence with a different
 * scene name stapled on. This is the line that TEACHES the gesture; the
 * per-scene hint below it only ever suggests what to try once you have it.
 */
const GESTURE = 'Drag to stir the flow. Shift-drag to draw walls into it.';

type Setter = (key: UiKey, value: string | boolean) => void;

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

/** A label, its control, and the key that operates it — the row every setting
 *  in the panel is made of. */
function row(label: string, control: HTMLElement, key?: string): HTMLElement {
  const wrap = el('label', 'row');
  const name = el('span', 'row-label', label);
  if (key) name.append(kbd(key));
  wrap.append(name, control);
  return wrap;
}

function kbd(key: string): HTMLElement {
  return el('kbd', undefined, key);
}

export class Controls {
  private readonly selects = new Map<UiKey, HTMLSelectElement>();
  private readonly checks = new Map<UiKey, HTMLInputElement>();
  /** Chip rows and segmented switches: a container of button[data-value]. */
  private readonly groups = new Map<UiKey, HTMLElement>();

  private readonly panel: HTMLElement;
  private readonly panelButton: HTMLButtonElement;
  private readonly blurb: HTMLElement;
  private readonly hudEl: HTMLElement;
  private readonly scenes: SceneOption[];

  constructor(
    host: HTMLElement,
    private readonly state: UiState,
    spec: ControlsSpec,
    private readonly onSet: Setter,
    onRestart: () => void,
  ) {
    this.scenes = spec.scenes;

    // --- top bar: identity, the scene switcher, and the two verbs ----------
    const bar = el('div', 'topbar');
    bar.append(el('div', 'brand', 'Stable Fluids'));
    bar.append(this.chipGroup('scene', spec.scenes));

    const right = el('div', 'topbar-right');
    const restart = el('button', 'ghost', 'Restart');
    restart.append(kbd('R'));
    restart.addEventListener('click', onRestart);
    this.panelButton = el('button', 'ghost', 'Controls');
    this.panelButton.append(kbd('C'));
    this.panelButton.addEventListener('click', () => this.togglePanel());
    right.append(restart, this.panelButton);
    bar.append(right);

    this.blurb = el('div', 'blurb');

    // --- panel -------------------------------------------------------------
    this.panel = el('aside', 'panel');
    const picture = el('section');
    picture.append(
      el('h2', undefined, 'Picture'),
      row('View', this.select('view', spec.views), 'D'),
      row('Dye', this.select('dye', spec.dyes), 'T'),
      this.check('arrows', 'Velocity arrows', 'V'),
    );

    const sim = el('section');
    sim.append(
      el('h2', undefined, 'Simulation'),
      row('Runs on', this.segmented('engine', spec.engines), 'G'),
      el('p', 'note', 'CPU runs the same algorithm on one thread — the control, and slow.'),
      row('Resolution', this.select('quality', spec.qualities)),
      el('p', 'note', 'The grid is rebuilt to fill the window, so resizing restarts the flow.'),
    );

    const advanced = el('details', 'advanced');
    const summary = el('summary', undefined, 'Advanced');
    advanced.append(
      summary,
      row('Advection', this.select('scheme', spec.schemes), 'A'),
      row('Pressure solver', this.select('solver', spec.solvers)),
      this.check('perf', 'Performance panel', 'P'),
      this.check('diagnostics', 'Diagnostics readout'),
    );

    this.panel.append(picture, sim, advanced);

    // --- status line -------------------------------------------------------
    this.hudEl = el('div', 'hud');

    host.append(bar, this.blurb, this.panel, this.hudEl);
    this.sync();
  }

  /** Pushes `state` into every widget. The single direction data flows back. */
  sync(): void {
    for (const [key, sel] of this.selects) sel.value = String(this.state[key]);
    for (const [key, box] of this.checks) box.checked = Boolean(this.state[key]);
    for (const [key, group] of this.groups) {
      for (const b of group.querySelectorAll<HTMLButtonElement>('button')) {
        b.setAttribute('aria-pressed', String(b.dataset.value === this.state[key]));
      }
    }
    const scene = this.scenes.find((s) => s.value === this.state.scene);
    this.blurb.textContent = scene?.blurb ?? '';
    // Two separate elements, not one string: GESTURE teaches the brush and
    // the scene's own hint suggests an experiment with it, and those are
    // different enough claims that running them into one line blurred both —
    // a long gesture-plus-experiment sentence read as description again,
    // exactly what .hint's rule exists to avoid. Splitting them one per line
    // keeps each readable as its own short instruction.
    this.blurb.append(el('span', 'hint', GESTURE));
    if (scene?.hint) this.blurb.append(el('span', 'hint-scene', scene.hint));
  }

  /** The scene changed, so the meaningful tracers did too. */
  setDyes(options: Option[]): void {
    this.fill(this.selects.get('dye')!, options);
    this.sync();
  }

  /** The engine changed, so the reachable solvers did too. */
  setSolvers(options: Option[]): void {
    this.fill(this.selects.get('solver')!, options);
    this.sync();
  }

  /** The window changed, so the grid each resolution stands for did too — the
   *  labels quote it, so they are rewritten rather than fixed at build time. */
  setQualities(options: Option[]): void {
    this.fill(this.selects.get('quality')!, options);
    this.sync();
  }

  /** No adapter, or the device was lost: stop offering a switch that cannot
   *  be honoured, rather than letting it silently do nothing. */
  setGpuAvailable(ok: boolean): void {
    const group = this.groups.get('engine');
    const gpu = group?.querySelector<HTMLButtonElement>('button[data-value="gpu"]');
    if (gpu) {
      gpu.disabled = !ok;
      gpu.title = ok ? '' : 'No WebGPU adapter on this browser';
    }
  }

  setHud(text: string): void {
    this.hudEl.textContent = text;
  }

  togglePanel(): void {
    const open = !this.panel.classList.contains('open');
    this.panel.classList.toggle('open', open);
    this.panelButton.setAttribute('aria-expanded', String(open));
  }

  // --- widget factories ----------------------------------------------------

  private fill(sel: HTMLSelectElement, options: Option[]): void {
    sel.replaceChildren(
      ...options.map((o) => {
        const opt = el('option', undefined, o.label);
        opt.value = o.value;
        return opt;
      }),
    );
  }

  private select(key: UiKey, options: Option[]): HTMLSelectElement {
    const sel = el('select');
    this.fill(sel, options);
    sel.addEventListener('change', () => this.onSet(key, sel.value));
    this.selects.set(key, sel);
    return sel;
  }

  private check(key: UiKey, label: string, shortcut?: string): HTMLElement {
    const wrap = el('label', 'row check');
    const box = el('input');
    box.type = 'checkbox';
    box.addEventListener('change', () => this.onSet(key, box.checked));
    const name = el('span', 'row-label', label);
    if (shortcut) name.append(kbd(shortcut));
    wrap.append(box, name);
    this.checks.set(key, box);
    return wrap;
  }

  /** Two or three mutually exclusive choices, shown as one switch — the shape
   *  that says "these are the only options" better than a dropdown does. */
  private segmented(key: UiKey, options: Option[]): HTMLElement {
    return this.buttons(key, options, 'segmented');
  }

  /** The scene switcher. Buttons rather than a dropdown because switching
   *  scenes is the thing this demo is FOR: it should cost one click and be
   *  visible without opening anything. */
  private chipGroup(key: UiKey, options: Option[]): HTMLElement {
    return this.buttons(key, options, 'chips');
  }

  private buttons(key: UiKey, options: Option[], className: string): HTMLElement {
    const group = el('div', className);
    group.setAttribute('role', 'group');
    for (const o of options) {
      const b = el('button', undefined, o.label);
      b.dataset.value = o.value;
      b.addEventListener('click', () => this.onSet(key, o.value));
      group.append(b);
    }
    this.groups.set(key, group);
    return group;
  }
}
