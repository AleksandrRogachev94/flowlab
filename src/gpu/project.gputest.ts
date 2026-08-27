/**
 * The correctness gate for project.wgsl, in the mould of the other two: run
 * every kernel in the file on real hardware and diff the results against their
 * f64 CPU originals — computeDivergence, subtractGradient, applyOutflow, the
 * dye fade Simulation.step applies, and core/dye.ts's applyDyePatch — on a
 * fixture with an interior solid and an Air outlet column.
 *
 * The dye_patch kernel matters here more than its size suggests. It is the
 * ONLY copy of an emitter that runs on the device, and its host twin is what
 * the CPU engine runs; the two drifting would show up as the same scene
 * looking different on the two engines, which is exactly the comparison the
 * whole ladder exists to make.
 *
 * The dispatch ORDER matches stepGpu.ts — divergence first, then the
 * projection in place, then the outflow clamp — so the test also pins the one
 * ordering fact the fused step depends on: b is the divergence of the
 * PRE-projection velocity, read before subtract_* mutates it in the same pass.
 *
 * Needs Chrome and a GPU: `npm run test:gpu`.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { applyOutflow } from '../core/boundaries.ts';
import { computeDivergence } from '../core/divergence.ts';
import { applyDyePatch, makeDyePatch } from '../core/dye.ts';
import { Cell, createGrid, idxP } from '../core/grid.ts';
import { subtractGradient } from '../core/subtractGradient.ts';
import { evalInBrowser, NoBrowserError } from './chromeHarness.ts';

const PROJECT_WGSL = readFileSync(new URL('./project.wgsl', import.meta.url), 'utf8');

const NX = 40;
const NY = 32;
const SCALE = 0.02;
const GRAD_SCALE = 0.5;
/** Deliberately not a round number, so a cell scaled twice or zero times is
 *  visible rather than coincidentally right. */
const DYE_KEEP = 0.9137;
const DYE_CHANNELS = 3;
/** The inlet: three columns, with a band of zero coverage so the lerp's
 *  "leave this cell alone" path is exercised rather than assumed. */
const PATCH_COLS = 3;

/** Deterministic pseudo-random fields. Deliberately NOT given the solid-face
 *  invariant: every kernel under test must get its answers from its own bounds
 *  and label guards, not from convenient zeros in the data. */
function fixture(): {
  label: Uint8Array;
  u: Float64Array;
  v: Float64Array;
  p: Float64Array;
  dye: Float64Array;
} {
  const g = createGrid(NX, NY, 1 / NY);
  const label = new Uint8Array(NX * NY);
  // Air outlet column on the right — where outflow() must clamp and copy —
  // and an interior solid block, where subtract_* must skip faces.
  for (let j = 0; j < NY; j++) label[idxP(g, NX - 1, j)] = Cell.Air;
  for (let j = 12; j < 20; j++) {
    for (let i = 10; i < 16; i++) label[idxP(g, i, j)] = Cell.Solid;
  }
  let seed = 98765;
  const rand = (): number => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return (seed / 0x7fffffff) * 2 - 1;
  };
  const u = new Float64Array((NX + 1) * NY);
  const v = new Float64Array(NX * (NY + 1));
  const p = new Float64Array(NX * NY);
  const dye = new Float64Array(DYE_CHANNELS * NX * NY);
  for (let k = 0; k < u.length; k++) u[k] = rand();
  for (let k = 0; k < v.length; k++) v[k] = rand();
  for (let k = 0; k < p.length; k++) p[k] = rand();
  // Nonzero everywhere, including solids: decay() has no label guard, exactly
  // as the host loop it replaces had none.
  for (let k = 0; k < dye.length; k++) dye[k] = 0.25 + 0.5 * Math.abs(rand());
  return { label, u, v, p, dye };
}

const list = (a: Float32Array): number[] => Array.from(a);

interface PageResult {
  skip?: string;
  b?: number[];
  u?: number[];
  v?: number[];
  dye?: number[];
  diagnostics?: string[];
  validation?: string | null;
  adapter?: string;
}

function maxDiff(expected: Float64Array, actual: number[] = [], name = ''): number {
  assert.equal(actual.length, expected.length, `wrong number of ${name} values came back`);
  let diff = 0;
  for (let k = 0; k < expected.length; k++) {
    diff = Math.max(diff, Math.abs(expected[k] - actual[k]));
  }
  return diff;
}

test('project.wgsl reproduces divergence, projection, outflow and the dye kernels', async (t) => {
  const { label, u, v, p, dye } = fixture();
  const g = createGrid(NX, NY, 1 / NY);
  // Values that vary in BOTH directions, so a transposed patch index shows up,
  // and a stripe of zero coverage down the middle of the column band.
  const patch = makeDyePatch(0, 0, PATCH_COLS, NY, (i, j, rgb) => {
    if (j % 5 === 0) return 0;
    rgb[0] = 0.1 * i + 0.01 * j;
    rgb[1] = 1 - 0.02 * j;
    rgb[2] = 0.5;
    return j % 7 === 0 ? 0.4 : 1;
  });

  const script = `(async () => {
  if (!navigator.gpu) return { skip: 'navigator.gpu missing (secure context? headless flags?)' };
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return { skip: 'no GPU adapter' };
  const device = await adapter.requestDevice();
  device.pushErrorScope('validation');

  const nx = ${NX}, ny = ${NY};
  const uLen = (nx + 1) * ny, vLen = nx * (ny + 1), cells = nx * ny;
  const uIn = ${JSON.stringify(Array.from(u))};
  const vIn = ${JSON.stringify(Array.from(v))};
  const pIn = ${JSON.stringify(Array.from(p))};
  const dyeIn = ${JSON.stringify(Array.from(dye))};
  const labelIn = ${JSON.stringify(Array.from(label))};
  const SRC = ${JSON.stringify(PROJECT_WGSL)};

  const mod = device.createShaderModule({ code: SRC });
  const diagnostics = [];
  const info = await mod.getCompilationInfo();
  for (const m of info.messages) diagnostics.push(m.type + ' ' + m.lineNum + ':' + m.linePos + ' ' + m.message);
  const pipe = (entryPoint) =>
    device.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint } });
  const divPipe = pipe('divergence');
  const suPipe = pipe('subtract_u');
  const svPipe = pipe('subtract_v');
  const outPipe = pipe('outflow');
  const decayPipe = pipe('decay');
  const patchPipe = pipe('dye_patch');

  const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const buf = (len) => device.createBuffer({ size: len * 4, usage: S });
  const uBuf = buf(uLen), vBuf = buf(vLen), pBuf = buf(cells), bBuf = buf(cells);
  const labelBuf = buf(cells);
  const dyeBuf = buf(${DYE_CHANNELS} * cells);
  // 16 and 32, matching PARAMS_BYTES and DYE_PARAMS_BYTES in stepGpu.ts.
  const U = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
  const paramsBuf = device.createBuffer({ size: 16, usage: U });
  const pd = new ArrayBuffer(16);
  new Uint32Array(pd).set([nx, ny]);
  // divCoef = -scale / h, gradScale — the same fold stepGpu.ts writes.
  new Float32Array(pd).set([-${SCALE} * ny, ${GRAD_SCALE}], 2);
  device.queue.writeBuffer(paramsBuf, 0, pd);

  // DyeParams. The dye grid IS the velocity grid here: this file pins the
  // kernels' arithmetic, and advect.gputest.ts is where the two grids come
  // apart.
  const dyeParamsBuf = device.createBuffer({ size: 32, usage: U });
  const dpd = new ArrayBuffer(32);
  new Uint32Array(dpd).set([nx, ny, 0, 0, 0, ${PATCH_COLS}, ny]);
  new Float32Array(dpd).set([${DYE_KEEP}], 2);
  device.queue.writeBuffer(dyeParamsBuf, 0, dpd);

  const patchBuf = buf(${DYE_CHANNELS + 1} * ${PATCH_COLS} * ny);
  device.queue.writeBuffer(patchBuf, 0, new Float32Array(${JSON.stringify(list(patch.data))}));
  device.queue.writeBuffer(uBuf, 0, new Float32Array(uIn));
  device.queue.writeBuffer(vBuf, 0, new Float32Array(vIn));
  device.queue.writeBuffer(pBuf, 0, new Float32Array(pIn));
  device.queue.writeBuffer(labelBuf, 0, new Uint32Array(labelIn));
  device.queue.writeBuffer(dyeBuf, 0, new Float32Array(dyeIn));

  const group = (pipe, entries) => device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: entries.map(([binding, buffer]) => ({ binding, resource: { buffer } })),
  });
  const divG = group(divPipe, [[0, paramsBuf], [3, uBuf], [4, vBuf], [5, bBuf]]);
  const suG = group(suPipe, [[0, paramsBuf], [1, labelBuf], [2, pBuf], [3, uBuf]]);
  const svG = group(svPipe, [[0, paramsBuf], [1, labelBuf], [2, pBuf], [4, vBuf]]);
  const outG = group(outPipe, [[0, paramsBuf], [1, labelBuf], [3, uBuf], [4, vBuf]]);
  const decayG = group(decayPipe, [[6, dyeBuf], [7, dyeParamsBuf]]);
  const patchG = group(patchPipe, [[6, dyeBuf], [7, dyeParamsBuf], [8, patchBuf]]);

  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  const gx = Math.ceil(nx / 8), gy = Math.ceil(ny / 8);
  pass.setPipeline(divPipe); pass.setBindGroup(0, divG); pass.dispatchWorkgroups(Math.ceil((nx + 1) / 8), gy);
  pass.setPipeline(suPipe); pass.setBindGroup(0, suG); pass.dispatchWorkgroups(Math.ceil((nx + 1) / 8), gy);
  pass.setPipeline(svPipe); pass.setBindGroup(0, svG); pass.dispatchWorkgroups(gx, Math.ceil((ny + 1) / 8));
  pass.setPipeline(outPipe); pass.setBindGroup(0, outG); pass.dispatchWorkgroups(Math.ceil(Math.max(nx, ny) / 64));
  // Dye, in stepGpu.ts's order: the fade, then the inlet last.
  pass.setPipeline(decayPipe); pass.setBindGroup(0, decayG); pass.dispatchWorkgroups(gx, Math.ceil(ny * ${DYE_CHANNELS} / 8));
  pass.setPipeline(patchPipe); pass.setBindGroup(0, patchG); pass.dispatchWorkgroups(Math.ceil(${PATCH_COLS} / 8), Math.ceil(ny / 8));
  pass.end();
  const dyeLen = ${DYE_CHANNELS} * cells;
  const readBuf = device.createBuffer({
    size: (cells + uLen + vLen + dyeLen) * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  enc.copyBufferToBuffer(bBuf, 0, readBuf, 0, cells * 4);
  enc.copyBufferToBuffer(uBuf, 0, readBuf, cells * 4, uLen * 4);
  enc.copyBufferToBuffer(vBuf, 0, readBuf, (cells + uLen) * 4, vLen * 4);
  enc.copyBufferToBuffer(dyeBuf, 0, readBuf, (cells + uLen + vLen) * 4, dyeLen * 4);
  device.queue.submit([enc.finish()]);
  await readBuf.mapAsync(GPUMapMode.READ);
  const all = new Float32Array(readBuf.getMappedRange());
  const b = Array.from(all.subarray(0, cells));
  const uOut = Array.from(all.subarray(cells, cells + uLen));
  const vOut = Array.from(all.subarray(cells + uLen, cells + uLen + vLen));
  const dyeOut = Array.from(all.subarray(cells + uLen + vLen));
  readBuf.unmap();
  const validation = await device.popErrorScope();
  return { b, u: uOut, v: vOut, dye: dyeOut, diagnostics, validation: validation ? validation.message : null,
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
  assert.deepEqual(res.diagnostics, [], 'shader compiled with diagnostics');

  // The CPU reference, in the SAME order: b from the pre-projection velocity,
  // then the in-place projection, then the clamp.
  const div = new Float64Array(NX * NY);
  computeDivergence(g, u, v, div);
  const bExp = Float64Array.from(div, (d) => -SCALE * d);
  subtractGradient(g, p, u, v, label, GRAD_SCALE);
  applyOutflow(g, u, v, label);
  // The dye half, in the same order the pass recorded it. A cell the 2D decay
  // dispatch missed comes back unscaled and a cell it hit twice comes back
  // squared; both are ~1e-2 off, four orders above the f32 floor below.
  const dyeExp = Float64Array.from(dye, (q) => q * DYE_KEEP);
  const cells = NX * NY;
  const planes = Array.from(
    { length: DYE_CHANNELS },
    (_, c) => new Float64Array(dyeExp.buffer, c * cells * 8, cells),
  );
  applyDyePatch(g, planes, patch);

  const bDiff = maxDiff(bExp, res.b, 'b');
  const uDiff = maxDiff(u, res.u, 'u');
  const vDiff = maxDiff(v, res.v, 'v');
  const dyeDiff = maxDiff(dyeExp, res.dye, 'dye');

  // f32 against f64 on O(1) data costs ~1e-7; a wrong stride, bound or label
  // guard misses by orders of magnitude.
  for (const [name, diff] of [
    ['b', bDiff],
    ['u', uDiff],
    ['v', vDiff],
    ['dye', dyeDiff],
  ] as const) {
    assert.ok(diff < 1e-6, `GPU ${name} differs from CPU: max |diff| ${diff}`);
  }
  t.diagnostic(
    `adapter ${res.adapter}  max diffs: b ${bDiff.toExponential(2)}, ` +
      `u ${uDiff.toExponential(2)}, v ${vDiff.toExponential(2)}, ` +
      `dye ${dyeDiff.toExponential(2)}`,
  );
});
