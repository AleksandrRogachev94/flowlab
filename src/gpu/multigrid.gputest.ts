/**
 * Two gates for the GPU multigrid, mirroring redBlack.gputest.ts's role:
 *
 * 1. CORRECTNESS: run the full V-cycle stack (multigrid.wgsl transfers plus
 *    redBlack.wgsl as smoother) on real hardware and diff the resulting p
 *    against the f64 CPU mirror, cycle-for-cycle. pressureMultigrid.test.ts
 *    already pins the CPU mirror against converged SOR, so a failure here is
 *    the shaders or the plumbing (a swapped binding, a wrong dispatch size on
 *    a coarse level, a label chain out of order), never the algorithm.
 *
 * 2. SPEED, as a diagnostic, at the browser's real 1024x768: wall time of a
 *    3-cycle multigrid solve against the 154-sweep red-black budget it
 *    replaces, and the residual each leaves. Printed, not asserted — machines
 *    differ; the numbers in docs/WEBGPU.md §9 came from here.
 *
 * Needs Chrome and a GPU: `npm run test:gpu`.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { Cell, createGrid, idxP } from '../core/grid.ts';
import { rmsRemainingDivergence } from '../core/pressure.ts';
import {
  CpuMultigridSolver,
  levelSizes,
  MG_COARSE_SWEEPS,
  MG_OMEGA,
  MG_POST_SWEEPS,
  MG_PRE_SWEEPS,
} from '../core/pressureMultigrid.ts';
import { evalInBrowser, NoBrowserError } from './chromeHarness.ts';

const REDBLACK_WGSL = readFileSync(new URL('./redBlack.wgsl', import.meta.url), 'utf8');
const MULTIGRID_WGSL = readFileSync(new URL('./multigrid.wgsl', import.meta.url), 'utf8');

const NX = 40;
const NY = 32;
const SCALE = 0.02;
const CYCLES = 4;

/** Same fixture family as redBlack.gputest: Air column, interior solid, rough
 *  RHS — chosen because label coarsening is where multigrid bugs live. */
function fixture(): { label: Uint8Array; div: Float64Array } {
  const g = createGrid(NX, NY, 1 / NY);
  const label = new Uint8Array(NX * NY);
  for (let j = 0; j < NY; j++) label[idxP(g, NX - 1, j)] = Cell.Air;
  for (let j = 12; j < 20; j++) {
    for (let i = 10; i < 16; i++) label[idxP(g, i, j)] = Cell.Solid;
  }
  let seed = 12345;
  const div = new Float64Array(NX * NY);
  for (let k = 0; k < div.length; k++) {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    div[k] = (seed / 0x7fffffff) * 2 - 1;
  }
  return { label, div };
}

/**
 * The in-page mirror of GpuMultigridSolver — inlined for the same reason as
 * the redBlack page script (no module loader in the page), shared with the
 * perf test below. Keep the binding numbers in step with multigrid.wgsl.
 * Declared as a function body: the caller wraps it and supplies
 * `levels/div/label/scale/cycles/pre/post/coarse/omega` plus both sources.
 */
const MG_RUNNER = `
  const cells = levels[0].nx * levels[0].ny, bytes = cells * 4;
  const smoothMod = device.createShaderModule({ code: RB_SRC });
  const mgMod = device.createShaderModule({ code: MG_SRC });
  const diagnostics = [];
  for (const [name, mod] of [['redBlack', smoothMod], ['multigrid', mgMod]]) {
    const info = await mod.getCompilationInfo();
    for (const m of info.messages) diagnostics.push(name + ' ' + m.type + ' ' + m.lineNum + ':' + m.linePos + ' ' + m.message);
  }
  const pipe = (mod, entryPoint) =>
    device.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint } });
  const smoothPipe = pipe(smoothMod, 'main');
  const residualPipe = pipe(mgMod, 'residual');
  const restrictPipe = pipe(mgMod, 'restrictResidual');
  const prolongPipe = pipe(mgMod, 'prolong');
  const coarsenPipe = pipe(mgMod, 'coarsenLabels');

  const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const U = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
  const bufs = levels.map((s, l) => {
    const n = s.nx * s.ny * 4;
    return {
      x: device.createBuffer({ size: n, usage: S | (l === 0 ? GPUBufferUsage.COPY_SRC : 0) }),
      b: device.createBuffer({ size: n, usage: S }),
      r: device.createBuffer({ size: n, usage: S }),
      label: device.createBuffer({ size: n, usage: S }),
      gx: Math.ceil(s.nx / 8), gy: Math.ceil(s.ny / 8),
    };
  });
  const smoothParams = levels.map((s, l) => [0, 1].map((colour) => {
    const buf = device.createBuffer({ size: 32, usage: U });
    const ab = new ArrayBuffer(32), u = new Uint32Array(ab), f = new Float32Array(ab);
    u[0] = s.nx; u[1] = s.ny; f[2] = -1; f[3] = omega; u[4] = colour;
    device.queue.writeBuffer(buf, 0, ab);
    return buf;
  }));
  const levelParams = levels.slice(0, -1).map((s, l) => {
    const buf = device.createBuffer({ size: 16, usage: U });
    device.queue.writeBuffer(buf, 0, new Uint32Array([s.nx, s.ny, levels[l + 1].nx, levels[l + 1].ny]));
    return buf;
  });
  const group = (pipe, entries) => device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: entries.map(([binding, buffer]) => ({ binding, resource: { buffer } })),
  });
  const smoothG = bufs.map((bf, l) => [0, 1].map((c) =>
    group(smoothPipe, [[0, smoothParams[l][c]], [1, bf.x], [2, bf.b], [3, bf.label]])));
  const residualG = levelParams.map((pr, l) =>
    group(residualPipe, [[0, pr], [1, bufs[l].x], [2, bufs[l].b], [3, bufs[l].r], [4, bufs[l].label]]));
  const restrictG = levelParams.map((pr, l) =>
    group(restrictPipe, [[0, pr], [3, bufs[l].r], [6, bufs[l + 1].b], [7, bufs[l + 1].x]]));
  const prolongG = levelParams.map((pr, l) =>
    group(prolongPipe, [[0, pr], [1, bufs[l].x], [4, bufs[l].label], [5, bufs[l + 1].x]]));
  const coarsenG = levelParams.map((pr, l) =>
    group(coarsenPipe, [[0, pr], [4, bufs[l].label], [8, bufs[l + 1].label]]));

  const bF32 = new Float32Array(cells);
  for (let k = 0; k < cells; k++) bF32[k] = -scale * div[k];
  device.queue.writeBuffer(bufs[0].x, 0, new Float32Array(cells));
  device.queue.writeBuffer(bufs[0].b, 0, bF32);
  device.queue.writeBuffer(bufs[0].label, 0, new Uint32Array(label));

  const readBuf = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(coarsenPipe);
  for (let l = 0; l < levels.length - 1; l++) {
    pass.setBindGroup(0, coarsenG[l]);
    pass.dispatchWorkgroups(bufs[l + 1].gx, bufs[l + 1].gy);
  }
  const smoothPair = (l) => {
    pass.setPipeline(smoothPipe);
    pass.setBindGroup(0, smoothG[l][0]); pass.dispatchWorkgroups(bufs[l].gx, bufs[l].gy);
    pass.setBindGroup(0, smoothG[l][1]); pass.dispatchWorkgroups(bufs[l].gx, bufs[l].gy);
  };
  const vcycle = (l) => {
    if (l === levels.length - 1) {
      for (let s = 0; s < coarse; s++) smoothPair(l);
      return;
    }
    for (let s = 0; s < pre; s++) smoothPair(l);
    pass.setPipeline(residualPipe); pass.setBindGroup(0, residualG[l]);
    pass.dispatchWorkgroups(bufs[l].gx, bufs[l].gy);
    pass.setPipeline(restrictPipe); pass.setBindGroup(0, restrictG[l]);
    pass.dispatchWorkgroups(bufs[l + 1].gx, bufs[l + 1].gy);
    vcycle(l + 1);
    pass.setPipeline(prolongPipe); pass.setBindGroup(0, prolongG[l]);
    pass.dispatchWorkgroups(bufs[l].gx, bufs[l].gy);
    for (let s = 0; s < post; s++) smoothPair(l);
  };
  for (let c = 0; c < cycles; c++) vcycle(0);
  pass.end();
  enc.copyBufferToBuffer(bufs[0].x, 0, readBuf, 0, bytes);
  const t0 = performance.now();
  device.queue.submit([enc.finish()]);
  await readBuf.mapAsync(GPUMapMode.READ);
  const wallMs = performance.now() - t0;
  const p = Array.from(new Float32Array(readBuf.getMappedRange()));
  readBuf.unmap();
`;

/** Everything MG_RUNNER needs, as inlined consts. */
function preamble(nx: number, ny: number, cycles: number): string {
  return `
  const levels = ${JSON.stringify(levelSizes(nx, ny))};
  const scale = ${SCALE}, cycles = ${cycles};
  const pre = ${MG_PRE_SWEEPS}, post = ${MG_POST_SWEEPS}, coarse = ${MG_COARSE_SWEEPS}, omega = ${MG_OMEGA};
  const RB_SRC = ${JSON.stringify(REDBLACK_WGSL)};
  const MG_SRC = ${JSON.stringify(MULTIGRID_WGSL)};
  if (!navigator.gpu) return { skip: 'navigator.gpu missing (secure context? headless flags?)' };
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return { skip: 'no GPU adapter' };
  const device = await adapter.requestDevice();
  device.pushErrorScope('validation');
`;
}

interface PageResult {
  skip?: string;
  p?: number[];
  diagnostics?: string[];
  validation?: string | null;
  adapter?: string;
  wallMs?: number;
  sorMs?: number;
  sorP?: number[];
}

test('multigrid.wgsl reproduces the CPU multigrid solver on real hardware', async (t) => {
  const { label, div } = fixture();
  const g = createGrid(NX, NY, 1 / NY);

  const script = `(async () => {
  ${preamble(NX, NY, CYCLES)}
  const div = ${JSON.stringify(Array.from(div))};
  const label = ${JSON.stringify(Array.from(label))};
  ${MG_RUNNER}
  const validation = await device.popErrorScope();
  return { p, diagnostics, wallMs, validation: validation ? validation.message : null,
           adapter: adapter.info ? adapter.info.vendor + '/' + adapter.info.architecture : 'unknown' };
})()`;

  let res: PageResult;
  try {
    res = (await evalInBrowser(script)) as PageResult;
  } catch (e) {
    if (e instanceof NoBrowserError) return t.skip(e.message.split('\n')[0]);
    throw e;
  }
  if (res.skip) return t.skip(res.skip);

  assert.equal(res.validation, null, `WebGPU validation error: ${res.validation}`);
  assert.deepEqual(res.diagnostics, [], 'shaders compiled with diagnostics');

  const expected = new Float64Array(NX * NY);
  new CpuMultigridSolver(CYCLES).solve(g, expected, div, label, SCALE, 0, 0, 0);
  const actual = Float64Array.from(res.p ?? []);
  assert.equal(actual.length, expected.length, 'wrong number of cells came back');

  let pMax = 0;
  let diff = 0;
  for (let k = 0; k < expected.length; k++) {
    pMax = Math.max(pMax, Math.abs(expected[k]));
    diff = Math.max(diff, Math.abs(expected[k] - actual[k]));
  }
  assert.ok(pMax > 1e-3, 'test is vacuous if p is ~0');
  // Same threshold logic as the red-black gate: f32 against f64 costs ~1e-7
  // relative; a plumbing bug misses by orders of magnitude.
  assert.ok(diff < 1e-5 * pMax, `GPU differs from CPU: max ${diff} vs |p|max ${pMax}`);

  const rCpu = rmsRemainingDivergence(g, expected, div, label, SCALE);
  const rGpu = rmsRemainingDivergence(g, actual, div, label, SCALE);
  assert.ok(Math.abs(rGpu - rCpu) < 0.01 * Math.max(rCpu, 1e-12), `residual ${rGpu} vs ${rCpu}`);

  t.diagnostic(
    `adapter ${res.adapter}  |p|max ${pMax.toExponential(3)}  max diff ${diff.toExponential(3)}`,
  );
});

test('multigrid vs red-black SOR at 1024x768 (speed + residual, diagnostic)', async (t) => {
  const NXB = 1024;
  const NYB = 768;
  // 0.15 * max(nx, ny), the browser's real budget for this grid.
  const SWEEPS = Math.round(0.15 * NXB);

  const script = `(async () => {
  ${preamble(NXB, NYB, 3)}
  // Karman-like fixture built in-page: shipping two 786k-cell arrays through
  // JSON is slower than generating them.
  const nx = ${NXB}, ny = ${NYB};
  const div = new Float64Array(nx * ny);
  const label = new Uint32Array(nx * ny);
  for (let j = 0; j < ny; j++) label[nx - 1 + j * nx] = 1;
  const cx = nx / 4, cy = ny / 2, rad = ny / 8;
  for (let j = 0; j < ny; j++)
    for (let i = 0; i < nx; i++)
      if ((i - cx) ** 2 + (j - cy) ** 2 < rad * rad) label[i + j * nx] = 2;
  let seedv = 12345;
  for (let k = 0; k < div.length; k++) {
    seedv = (seedv * 1103515245 + 12345) & 0x7fffffff;
    div[k] = label[k] === 0 ? (seedv / 0x7fffffff) * 2 - 1 : 0;
  }
  ${MG_RUNNER}

  // The incumbent: redBlack.wgsl at the browser's sweep budget and omega.
  const sorPipe = device.createComputePipeline({ layout: 'auto', compute: { module: smoothMod, entryPoint: 'main' } });
  const pBuf = device.createBuffer({ size: bytes, usage: S | GPUBufferUsage.COPY_SRC });
  const dBuf = device.createBuffer({ size: bytes, usage: S });
  const pb = [0, 1].map((c) => {
    const buf = device.createBuffer({ size: 32, usage: U });
    const ab = new ArrayBuffer(32), u = new Uint32Array(ab), f = new Float32Array(ab);
    u[0] = nx; u[1] = ny; f[2] = scale; f[3] = 1.6; u[4] = c;
    device.queue.writeBuffer(buf, 0, ab);
    return buf;
  });
  const sorG = [0, 1].map((c) => group(sorPipe, [[0, pb[c]], [1, pBuf], [2, dBuf], [3, bufs[0].label]]));
  const divF32 = new Float32Array(cells);
  divF32.set(div);
  device.queue.writeBuffer(pBuf, 0, new Float32Array(cells));
  device.queue.writeBuffer(dBuf, 0, divF32);
  const enc2 = device.createCommandEncoder();
  const pass2 = enc2.beginComputePass();
  pass2.setPipeline(sorPipe);
  const gx = Math.ceil(nx / 8), gy = Math.ceil(ny / 8);
  for (let k = 0; k < ${SWEEPS}; k++) {
    pass2.setBindGroup(0, sorG[0]); pass2.dispatchWorkgroups(gx, gy);
    pass2.setBindGroup(0, sorG[1]); pass2.dispatchWorkgroups(gx, gy);
  }
  pass2.end();
  enc2.copyBufferToBuffer(pBuf, 0, readBuf, 0, bytes);
  const t1 = performance.now();
  device.queue.submit([enc2.finish()]);
  await readBuf.mapAsync(GPUMapMode.READ);
  const sorMs = performance.now() - t1;
  const sorP = Array.from(new Float32Array(readBuf.getMappedRange()));
  readBuf.unmap();
  const validation = await device.popErrorScope();
  return { p, sorP, wallMs, sorMs, diagnostics, validation: validation ? validation.message : null,
           adapter: adapter.info ? adapter.info.vendor + '/' + adapter.info.architecture : 'unknown' };
})()`;

  let res: PageResult;
  try {
    res = (await evalInBrowser(script, 120000)) as PageResult;
  } catch (e) {
    if (e instanceof NoBrowserError) return t.skip(e.message.split('\n')[0]);
    throw e;
  }
  if (res.skip) return t.skip(res.skip);
  assert.equal(res.validation, null, `WebGPU validation error: ${res.validation}`);

  // Recreate the fixture to score both answers with the f64 residual.
  const g = createGrid(NXB, NYB, 1 / NYB);
  const label = new Uint8Array(NXB * NYB);
  for (let j = 0; j < NYB; j++) label[idxP(g, NXB - 1, j)] = Cell.Air;
  const cx = NXB / 4;
  const cy = NYB / 2;
  const rad = NYB / 8;
  for (let j = 0; j < NYB; j++) {
    for (let i = 0; i < NXB; i++) {
      if ((i - cx) ** 2 + (j - cy) ** 2 < rad * rad) label[idxP(g, i, j)] = Cell.Solid;
    }
  }
  let seed = 12345;
  const div = new Float64Array(NXB * NYB);
  for (let k = 0; k < div.length; k++) {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    div[k] = label[k] === Cell.Fluid ? (seed / 0x7fffffff) * 2 - 1 : 0;
  }
  const rMg = rmsRemainingDivergence(g, Float64Array.from(res.p ?? []), div, label, SCALE);
  const rSor = rmsRemainingDivergence(g, Float64Array.from(res.sorP ?? []), div, label, SCALE);
  t.diagnostic(
    `adapter ${res.adapter}  mg 3 cycles: ${res.wallMs?.toFixed(1)} ms, residual ${rMg.toExponential(2)}  |  ` +
      `rbsor ${SWEEPS} sweeps: ${res.sorMs?.toFixed(1)} ms, residual ${rSor.toExponential(2)}`,
  );
  // The one hard claim: multigrid must not lose to the budget it replaces on
  // BOTH axes at once. (Either alone may vary by machine.)
  assert.ok(
    (res.wallMs ?? Infinity) < (res.sorMs ?? 0) || rMg < rSor,
    `mg slower AND less converged: ${res.wallMs} ms vs ${res.sorMs} ms, ${rMg} vs ${rSor}`,
  );
});
