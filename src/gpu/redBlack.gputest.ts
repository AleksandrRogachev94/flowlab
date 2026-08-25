/**
 * The GPU port's real correctness gate: run redBlack.wgsl on actual hardware
 * and diff it against the CPU red-black solver.
 *
 * Not a `.test.ts`, so `npm test` stays a one-second dependency-free run.
 * This needs Chrome and a GPU: `npm run test:gpu`.
 *
 * WHAT THIS CATCHES that the CPU tests cannot: every mistake in the plumbing
 * — a swapped binding, a wrong workgroup count leaving the last rows
 * untouched, a uniform struct whose padding does not match the shader's, an
 * off-by-one in the flat index. All of those produce a plausible-looking
 * pressure field that is simply wrong, and none of them are visible from
 * TypeScript.
 *
 * The reference is the CPU red-black solver, NOT the lexicographic one, and
 * that is the point: pressure.test.ts already pins red-black against
 * lexicographic in float64, so a failure here is the shader or the plumbing,
 * never the algorithm.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { Cell, createGrid, idxP } from '../core/grid.ts';
import { rmsRemainingDivergence } from '../core/pressure.ts';
import { solvePressureRedBlack } from '../core/pressureRedBlack.ts';
import { evalInBrowser, NoBrowserError } from './chromeHarness.ts';

const WGSL = readFileSync(new URL('./redBlack.wgsl', import.meta.url), 'utf8');

const NX = 40;
const NY = 32;
const SCALE = 0.02;
const OMEGA = 1.85;
const SWEEPS = 200;

/** Air column to pin p, an interior solid to exercise the Neumann branch, and
 *  a rough RHS so every mode is excited rather than only the smooth ones. */
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
 * Mirrors GpuPressureSolver.solve() closely enough to test the same code
 * paths, but inlined: the page has no module loader, so the real class cannot
 * be imported. That duplication is the harness's one real cost — keep the two
 * in step when the bind group layout changes.
 */
function pageScript(label: Uint8Array, div: Float64Array): string {
  return `(async () => {
  const nx = ${NX}, ny = ${NY}, scale = ${SCALE}, omega = ${OMEGA}, iterations = ${SWEEPS};
  const cells = nx * ny, bytes = cells * 4;
  if (!navigator.gpu) return { skip: 'navigator.gpu missing (secure context? headless flags?)' };
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return { skip: 'no GPU adapter' };
  const device = await adapter.requestDevice();
  device.pushErrorScope('validation');

  const module = device.createShaderModule({ code: ${JSON.stringify(WGSL)} });
  const info = await module.getCompilationInfo();
  const diagnostics = info.messages.map(m => m.type + ' ' + m.lineNum + ':' + m.linePos + ' ' + m.message);
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });

  const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const pBuf = device.createBuffer({ size: bytes, usage: S | GPUBufferUsage.COPY_SRC });
  const dBuf = device.createBuffer({ size: bytes, usage: S });
  const lBuf = device.createBuffer({ size: bytes, usage: S });
  const rBuf = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const U = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
  const pb = [device.createBuffer({ size: 32, usage: U }), device.createBuffer({ size: 32, usage: U })];
  const layout = pipeline.getBindGroupLayout(0);
  const bg = [0, 1].map(c => device.createBindGroup({ layout, entries: [
    { binding: 0, resource: { buffer: pb[c] } }, { binding: 1, resource: { buffer: pBuf } },
    { binding: 2, resource: { buffer: dBuf } }, { binding: 3, resource: { buffer: lBuf } }] }));

  for (const c of [0, 1]) {
    const ab = new ArrayBuffer(32), u = new Uint32Array(ab), f = new Float32Array(ab);
    u[0] = nx; u[1] = ny; f[2] = scale; f[3] = omega; u[4] = c;
    device.queue.writeBuffer(pb[c], 0, ab);
  }
  device.queue.writeBuffer(pBuf, 0, new Float32Array(cells));
  device.queue.writeBuffer(dBuf, 0, new Float32Array(${JSON.stringify(Array.from(div))}));
  device.queue.writeBuffer(lBuf, 0, new Uint32Array(${JSON.stringify(Array.from(label))}));

  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  const gx = Math.ceil(nx / 8), gy = Math.ceil(ny / 8);
  for (let k = 0; k < iterations; k++) {
    pass.setBindGroup(0, bg[0]); pass.dispatchWorkgroups(gx, gy);
    pass.setBindGroup(0, bg[1]); pass.dispatchWorkgroups(gx, gy);
  }
  pass.end();
  enc.copyBufferToBuffer(pBuf, 0, rBuf, 0, bytes);
  device.queue.submit([enc.finish()]);
  const validation = await device.popErrorScope();

  await rBuf.mapAsync(GPUMapMode.READ);
  const p = Array.from(new Float32Array(rBuf.getMappedRange()));
  rBuf.unmap();
  return { p, diagnostics, validation: validation ? validation.message : null,
           adapter: adapter.info ? adapter.info.vendor + '/' + adapter.info.architecture : 'unknown' };
})()`;
}

interface PageResult {
  skip?: string;
  p?: number[];
  diagnostics?: string[];
  validation?: string | null;
  adapter?: string;
}

test('redBlack.wgsl reproduces the CPU red-black solver on real hardware', async (t) => {
  const { label, div } = fixture();
  const g = createGrid(NX, NY, 1 / NY);

  let res: PageResult;
  try {
    res = (await evalInBrowser(pageScript(label, div))) as PageResult;
  } catch (e) {
    // No Chrome is a legitimate environment, not a failing shader. A hard
    // failure here would make `npm run test:gpu` useless in CI.
    if (e instanceof NoBrowserError) return t.skip(e.message.split('\n')[0]);
    throw e;
  }
  if (res.skip) return t.skip(res.skip);

  assert.equal(res.validation, null, `WebGPU validation error: ${res.validation}`);
  assert.deepEqual(res.diagnostics, [], `shader compiled with diagnostics`);

  const expected = new Float64Array(NX * NY);
  solvePressureRedBlack(g, expected, div, label, SCALE, SWEEPS, OMEGA, 0);
  const actual = Float64Array.from(res.p ?? []);
  assert.equal(actual.length, expected.length, 'wrong number of cells came back');

  let pMax = 0;
  let diff = 0;
  for (let k = 0; k < expected.length; k++) {
    pMax = Math.max(pMax, Math.abs(expected[k]));
    diff = Math.max(diff, Math.abs(expected[k] - actual[k]));
  }
  assert.ok(pMax > 1e-3, 'test is vacuous if p is ~0');
  // 1e-5 relative: the shader runs in f32 against the reference's f64, so
  // ~1e-7 relative is the floor. A real plumbing bug misses by orders of
  // magnitude more, so this threshold separates the two cleanly.
  assert.ok(diff < 1e-5 * pMax, `GPU differs from CPU: max ${diff} vs |p|max ${pMax}`);

  // Same solution AND the same residual — a p that merely looks similar but
  // does not solve the system would pass the diff and fail this.
  const rCpu = rmsRemainingDivergence(g, expected, div, label, SCALE);
  const rGpu = rmsRemainingDivergence(g, actual, div, label, SCALE);
  assert.ok(Math.abs(rGpu - rCpu) < 0.01 * rCpu, `residual ${rGpu} vs ${rCpu}`);

  t.diagnostic(
    `adapter ${res.adapter}  |p|max ${pMax.toExponential(3)}  max diff ${diff.toExponential(3)}`,
  );
});
