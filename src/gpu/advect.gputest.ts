/**
 * Runs advect.wgsl on real hardware and diffs it against CpuAdvector.
 *
 * Not a `.test.ts`, so `npm test` stays a one-second dependency-free run.
 * This needs Chrome and a GPU: `npm run test:gpu`.
 *
 * WHAT IT CATCHES that core/advect.test.ts cannot: the MAC grid's three
 * different extents ((nx+1)*ny, nx*(ny+1), nx*ny), each with its own
 * half-cell offset and its own bounds guard. A swapped stride or a missing
 * -0.5 produces a picture that still flows — just half a cell wrong, or wrong
 * only in the last row — and nothing in TypeScript can see it.
 *
 * The reference is the CPU advector, so a failure here is the shader or the
 * plumbing, never the scheme: core/advect.test.ts already pins the scheme.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { AdvectionScheme } from '../core/advect.ts';
import { CpuAdvector } from '../core/advector.ts';
import { Cell, createFields, createGrid, idxP, idxU, idxV } from '../core/grid.ts';
import { evalInBrowser, NoBrowserError } from './chromeHarness.ts';

const WGSL = readFileSync(new URL('./advect.wgsl', import.meta.url), 'utf8');

const NX = 40;
const NY = 32;
const G = createGrid(NX, NY, 1 / NY);
/** ~1.5 cells of travel: far enough to cross stencils, short enough that the
 *  backtrace still lands inside the domain over most of the grid. */
const DT = (1.5 * G.h) / 1.2;

/**
 * A shear plus a swirl, an interior solid, and an Air outlet column — so the
 * run exercises the solid copy-through branch, the off-grid-as-wall branch,
 * and a backtrace that clamps against the domain edge.
 */
function fixture() {
  const f = createFields(G, Float64Array);
  for (let j = 0; j < NY; j++) {
    for (let i = 0; i <= NX; i++) {
      const x = i * G.h;
      const y = (j + 0.5) * G.h;
      f.u[idxU(G, i, j)] = 1.2 * Math.sin(3 * y) + 0.4 * Math.cos(5 * x);
    }
  }
  for (let j = 0; j <= NY; j++) {
    for (let i = 0; i < NX; i++) {
      const x = (i + 0.5) * G.h;
      const y = j * G.h;
      f.v[idxV(G, i, j)] = -0.9 * Math.sin(4 * x) + 0.3 * Math.cos(2 * y);
    }
  }
  for (let j = 0; j < NY; j++) f.label[idxP(G, NX - 1, j)] = Cell.Air;
  for (let j = 12; j < 20; j++) {
    for (let i = 10; i < 16; i++) f.label[idxP(G, i, j)] = Cell.Solid;
  }
  // Sharp-edged dye: a smooth blob would hide a MacCormack limiter that never
  // bites, and the limiter is the fiddliest part of the kernel.
  for (let j = 0; j < NY; j++) {
    for (let i = 0; i < NX; i++) {
      const k = idxP(G, i, j);
      f.dye[0][k] = i < NX / 2 ? 1 : 0;
      f.dye[1][k] = (i + j) % 7 < 3 ? 0.8 : 0.1;
      f.dye[2][k] = j > NY / 3 ? 0.5 : 0;
    }
  }
  return f;
}

/**
 * Mirrors GpuAdvector, inlined: the page has no module loader, so the real
 * class cannot be imported. That duplication is the harness's one real cost —
 * keep the two in step when the bindings change.
 */
function pageScript(
  u: Float64Array,
  v: Float64Array,
  dye: Float64Array[],
  label: Uint8Array,
  mac: boolean,
): string {
  const nums = (a: ArrayLike<number>) => JSON.stringify(Array.from(a));
  return `(async () => {
  const nx = ${NX}, ny = ${NY}, h = ${G.h}, dt = ${DT}, mac = ${mac};
  const cells = nx * ny, uLen = (nx + 1) * ny, vLen = nx * (ny + 1), dyeLen = 3 * cells;
  if (!navigator.gpu) return { skip: 'navigator.gpu missing (secure context? headless flags?)' };
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return { skip: 'no GPU adapter' };
  const device = await adapter.requestDevice();
  device.pushErrorScope('validation');

  const module = device.createShaderModule({ code: ${JSON.stringify(WGSL)} });
  const info = await module.getCompilationInfo();
  const diagnostics = info.messages.map(m => m.type + ' ' + m.lineNum + ':' + m.linePos + ' ' + m.message);

  const st = (b, ro) => ({ binding: b, visibility: GPUShaderStage.COMPUTE, buffer: { type: ro ? 'read-only-storage' : 'storage' } });
  const layout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    st(1, true), st(2, true), st(3, true), st(4, true), st(5, true), st(6, false)] });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const pipe = {};
  for (const e of ['advect_u', 'advect_v', 'advect_dye', 'correct_u', 'correct_v', 'correct_dye']) {
    pipe[e] = device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint: e } });
  }

  const IN = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const OUT = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
  const buf = (n, usage) => device.createBuffer({ size: n * 4, usage });
  const uIn = buf(uLen, IN), vIn = buf(vLen, IN), dyeIn = buf(dyeLen, IN), lab = buf(cells, IN);
  const uA = buf(uLen, OUT), uB = buf(uLen, OUT), vA = buf(vLen, OUT), vB = buf(vLen, OUT);
  const dA = buf(dyeLen, OUT), dB = buf(dyeLen, OUT);
  const READ = GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST;
  const velRead = buf(uLen + vLen, READ), dyeRead = buf(dyeLen, READ);
  const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const ab = new ArrayBuffer(16);
  new Uint32Array(ab, 0, 2).set([nx, ny]);
  new Float32Array(ab, 8, 2).set([h, dt]);
  device.queue.writeBuffer(params, 0, ab);
  device.queue.writeBuffer(uIn, 0, new Float32Array(${nums(u)}));
  device.queue.writeBuffer(vIn, 0, new Float32Array(${nums(v)}));
  device.queue.writeBuffer(lab, 0, new Uint32Array(${nums(label)}));
  const dyeFlat = new Float32Array(dyeLen);
  dyeFlat.set(new Float32Array(${nums(dye[0])}), 0);
  dyeFlat.set(new Float32Array(${nums(dye[1])}), cells);
  dyeFlat.set(new Float32Array(${nums(dye[2])}), 2 * cells);
  device.queue.writeBuffer(dyeIn, 0, dyeFlat);

  const group = (src, orig, dst) => device.createBindGroup({ layout, entries: [
    { binding: 0, resource: { buffer: params } }, { binding: 1, resource: { buffer: lab } },
    { binding: 2, resource: { buffer: uIn } }, { binding: 3, resource: { buffer: vIn } },
    { binding: 4, resource: { buffer: src } }, { binding: 5, resource: { buffer: orig } },
    { binding: 6, resource: { buffer: dst } }] });

  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  const go = (entry, bg, w, hgt) => {
    pass.setPipeline(pipe[entry]);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(hgt / 8));
  };
  go('advect_u', group(uIn, uIn, uA), nx + 1, ny);
  go('advect_v', group(vIn, vIn, vA), nx, ny + 1);
  go('advect_dye', group(dyeIn, dyeIn, dA), nx, ny);
  if (mac) {
    go('correct_u', group(uA, uIn, uB), nx + 1, ny);
    go('correct_v', group(vA, vIn, vB), nx, ny + 1);
    go('correct_dye', group(dA, dyeIn, dB), nx, ny);
  }
  pass.end();
  enc.copyBufferToBuffer(mac ? uB : uA, 0, velRead, 0, uLen * 4);
  enc.copyBufferToBuffer(mac ? vB : vA, 0, velRead, uLen * 4, vLen * 4);
  enc.copyBufferToBuffer(mac ? dB : dA, 0, dyeRead, 0, dyeLen * 4);
  device.queue.submit([enc.finish()]);
  const validation = await device.popErrorScope();

  await velRead.mapAsync(GPUMapMode.READ);
  await dyeRead.mapAsync(GPUMapMode.READ);
  const vel = velRead.getMappedRange();
  const out = {
    u: Array.from(new Float32Array(vel, 0, uLen)),
    v: Array.from(new Float32Array(vel, uLen * 4, vLen)),
    dye: Array.from(new Float32Array(dyeRead.getMappedRange())),
    diagnostics, validation: validation ? validation.message : null,
    adapter: adapter.info ? adapter.info.vendor + '/' + adapter.info.architecture : 'unknown',
  };
  velRead.unmap();
  dyeRead.unmap();
  return out;
})()`;
}

interface PageResult {
  skip?: string;
  u?: number[];
  v?: number[];
  dye?: number[];
  diagnostics?: string[];
  validation?: string | null;
  adapter?: string;
}

/** Max |a - b|, and the peak magnitude it should be judged against. */
function compare(expected: ArrayLike<number>, actual: ArrayLike<number>): [number, number] {
  let peak = 0;
  let diff = 0;
  for (let k = 0; k < expected.length; k++) {
    peak = Math.max(peak, Math.abs(expected[k]));
    diff = Math.max(diff, Math.abs(expected[k] - actual[k]));
  }
  return [diff, peak];
}

for (const scheme of ['semiLagrangian', 'macCormack'] as AdvectionScheme[]) {
  test(`advect.wgsl reproduces CpuAdvector (${scheme})`, async (t) => {
    const f = fixture();

    let res: PageResult;
    try {
      res = (await evalInBrowser(
        pageScript(
          f.u as Float64Array,
          f.v as Float64Array,
          f.dye as Float64Array[],
          f.label,
          scheme === 'macCormack',
        ),
      )) as PageResult;
    } catch (e) {
      // No Chrome is a legitimate environment, not a failing shader.
      if (e instanceof NoBrowserError) return t.skip(e.message.split('\n')[0]);
      throw e;
    }
    if (res.skip) return t.skip(res.skip);
    assert.equal(res.validation, null, `WebGPU validation error: ${res.validation}`);
    assert.deepEqual(res.diagnostics, [], 'shader compiled with diagnostics');

    const cpu = new CpuAdvector(G);
    const uOut = new Float64Array(f.u.length);
    const vOut = new Float64Array(f.v.length);
    const dyeOut = f.dye.map((c) => new Float64Array(c.length));
    cpu.velocity(G, scheme, f.u, f.v, uOut, vOut, f.label, DT);
    // The same carrier the page used: the real step advects dye with the
    // PROJECTED velocity, but there is no projection here and the kernel
    // cannot tell the difference.
    cpu.dye(G, scheme, f.u, f.v, f.dye, dyeOut, f.label, DT);

    const dyeFlat = new Float64Array(3 * NX * NY);
    for (let c = 0; c < 3; c++) dyeFlat.set(dyeOut[c], c * NX * NY);

    const cases: [string, Float64Array, number[]][] = [
      ['u', uOut, res.u ?? []],
      ['v', vOut, res.v ?? []],
      ['dye', dyeFlat, res.dye ?? []],
    ];
    const report: string[] = [];
    for (const [name, expected, got] of cases) {
      assert.equal(got.length, expected.length, `${name}: wrong element count came back`);
      const [diff, peak] = compare(expected, got);
      assert.ok(peak > 1e-3, `${name}: test is vacuous, the field is ~0`);
      // 1e-5 relative: f32 on the device against f64 on the host puts the
      // floor near 1e-7, and a real plumbing bug misses by orders more.
      assert.ok(diff < 1e-5 * peak, `${name}: GPU differs by ${diff} on a peak of ${peak}`);
      report.push(`${name} ${(diff / peak).toExponential(1)}`);
    }
    t.diagnostic(`adapter ${res.adapter}  relative diff: ${report.join('  ')}`);
  });
}
