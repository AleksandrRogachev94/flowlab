/**
 * What has to hold for the buoyancy force, and one thing that must NOT.
 *
 * The physics claim worth a test is not "v goes up" — any sign error passes
 * that half the time — but that a hot blob rises AFTER the projection, which
 * is the only velocity anything sees. A uniform temperature is the control:
 * it is a constant force, i.e. a pure gradient, so the solve must absorb all
 * of it and nothing may move. Together those two pin the sign, the seam and
 * the coupling.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { applyBuoyancy } from './buoyancy.ts';
import { Cell, createGrid, idxP, idxV } from './grid.ts';
import { Simulation } from './simulation.ts';

const N = 48;

/** A Simulation with a hot patch of dye and nothing else — no seeded velocity,
 *  so anything that moves was moved by the force under test. */
function hotBox(w: number, blob?: (i: number, j: number) => number) {
  const sim = new Simulation(N, N, { buoyancy: w, dyeDecay: 0, tol: 1e-8 });
  sim.reset({
    dye: (dg, dye) => {
      for (let j = 0; j < dg.ny; j++) {
        for (let i = 0; i < dg.nx; i++) dye[0][idxP(dg, i, j)] = blob ? blob(i, j) : 1;
      }
    },
  });
  return sim;
}

test('a uniform temperature moves nothing: the projection absorbs a constant force', async () => {
  const sim = hotBox(2);
  for (let s = 0; s < 20; s++) await sim.step();
  // A constant body force is the gradient of a linear potential, so the
  // pressure solve can cancel it exactly. Anything left is solver residual,
  // not motion — hence a bound near the solve tolerance rather than zero.
  assert.ok(
    sim.maxFaceSpeed() < 1e-3,
    `uniform heat should not drive a flow; max face speed ${sim.maxFaceSpeed()}`,
  );
});

test('a hot blob rises and the fluid beside it comes down', async () => {
  const g = createGrid(N, N, 1 / N);
  // A disk in the lower half, well clear of every wall.
  const sim = hotBox(4, (i, j) => {
    const dx = (i + 0.5) / N - 0.5;
    const dy = (j + 0.5) / N - 0.3;
    return dx * dx + dy * dy < 0.1 * 0.1 ? 1 : 0;
  });
  for (let s = 0; s < 30; s++) await sim.step();

  // Up THROUGH the blob...
  const centre = sim.f.v[idxV(g, N >> 1, Math.round(0.3 * N))];
  assert.ok(centre > 0.05, `blob should rise; v at its centre is ${centre}`);
  // ...and down beside it, because a closed box conserves volume: nothing can
  // go up without something else coming back down. This is the projection's
  // doing, not the force's — the force only ever pushes one way.
  const beside = sim.f.v[idxV(g, Math.round(0.12 * N), Math.round(0.3 * N))];
  assert.ok(beside < 0, `fluid beside the plume should sink; v is ${beside}`);
});

test("a negative weight sinks, so the sign is the scene's to choose", async () => {
  const up = hotBox(4, (i, j) =>
    j > N / 3 && j < (2 * N) / 3 && i > N / 3 && i < (2 * N) / 3 ? 1 : 0,
  );
  const down = hotBox(-4, (i, j) =>
    j > N / 3 && j < (2 * N) / 3 && i > N / 3 && i < (2 * N) / 3 ? 1 : 0,
  );
  for (let s = 0; s < 10; s++) {
    await up.step();
    await down.step();
  }
  const g = createGrid(N, N, 1 / N);
  const k = idxV(g, N >> 1, N >> 1);
  assert.ok(up.f.v[k] > 0, `+w should lift; v is ${up.f.v[k]}`);
  assert.ok(down.f.v[k] < 0, `-w should sink; v is ${down.f.v[k]}`);
});

test('solid and boundary faces are left alone', () => {
  const g = createGrid(N, N, 1 / N);
  const dye = [new Float64Array(N * N), new Float64Array(N * N), new Float64Array(N * N)];
  dye[0].fill(1);
  const label = new Uint8Array(N * N);
  for (let j = 10; j < 14; j++) {
    for (let i = 10; i < 14; i++) label[idxP(g, i, j)] = Cell.Solid;
  }
  const v = new Float64Array(N * (N + 1));
  applyBuoyancy(g, g, v, dye[0], label, 1, 0.1);

  // Faces INSIDE the solid, and the two on its horizontal surfaces: all three
  // border a solid cell, and the wall condition stored there is what every
  // other kernel reads.
  for (let i = 10; i < 14; i++) {
    for (let j = 10; j <= 14; j++) {
      assert.equal(v[idxV(g, i, j)], 0, `solid face (${i}, ${j}) was written`);
    }
  }
  // The domain's own floor and ceiling are prescribed boundary data.
  for (let i = 0; i < N; i++) {
    assert.equal(v[idxV(g, i, 0)], 0, 'floor face was written');
    assert.equal(v[idxV(g, i, N)], 0, 'ceiling face was written');
  }
  // ...and an ordinary interior face did get the force, or the test above is
  // passing for the wrong reason.
  assert.ok(Math.abs(v[idxV(g, 30, 20)] - 0.1) < 1e-12);
});

test('the force is per unit of SIM TIME, not per step', () => {
  const g = createGrid(N, N, 1 / N);
  const dye = [new Float64Array(N * N), new Float64Array(N * N), new Float64Array(N * N)];
  dye[0].fill(1);
  const label = new Uint8Array(N * N);
  const one = new Float64Array(N * (N + 1));
  const many = new Float64Array(N * (N + 1));
  applyBuoyancy(g, g, one, dye[0], label, 3, 0.4);
  for (let s = 0; s < 4; s++) applyBuoyancy(g, g, many, dye[0], label, 3, 0.1);
  const k = idxV(g, 20, 20);
  assert.ok(Math.abs(one[k] - many[k]) < 1e-12, 'four small steps must equal one big one');
});

test('the dye grid may be finer than the velocity grid', () => {
  const g = createGrid(8, 8, 1 / 8);
  const dg = createGrid(16, 16, 1 / 16);
  const dye = [new Float64Array(256), new Float64Array(256), new Float64Array(256)];
  // A step in the dye at the middle of the DYE grid, which is also the middle
  // of the velocity grid: the face there must see half of it, since the
  // bilinear sample straddles the jump.
  for (let j = 0; j < 16; j++) {
    for (let i = 0; i < 16; i++) dye[0][idxP(dg, i, j)] = j >= 8 ? 1 : 0;
  }
  const v = new Float64Array(8 * 9);
  applyBuoyancy(g, dg, v, dye[0], new Uint8Array(64), 1, 1);
  // The v face at j = 4 sits at y = 0.5, exactly on the step. Dye row 7 (below)
  // and row 8 (above) are its two nearest centres, so it samples 0.5.
  assert.ok(Math.abs(v[idxV(g, 3, 4)] - 0.5) < 1e-12, `got ${v[idxV(g, 3, 4)]}`);
  // Well above the step it sees the full value; well below, nothing.
  assert.ok(Math.abs(v[idxV(g, 3, 6)] - 1) < 1e-12);
  assert.equal(v[idxV(g, 3, 2)], 0);
});
