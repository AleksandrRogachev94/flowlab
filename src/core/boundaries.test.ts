import { test } from 'node:test';
import assert from 'node:assert/strict';
import { applyOutflow, commitLabels } from './boundaries.ts';
import { Cell, createFields, createGrid, idxP, idxU, idxV } from './grid.ts';

const g = createGrid(6, 6, 1 / 6);

test('commitLabels zeroes every face touching a solid', () => {
  const f = createFields(g, Float64Array);
  f.u.fill(1);
  f.v.fill(1);
  f.label[idxP(g, 2, 2)] = Cell.Solid;

  commitLabels(g, f);

  // All four faces of the solid cell, by the per-cell reading of the layout.
  assert.equal(f.u[idxU(g, 2, 2)], 0, 'left face');
  assert.equal(f.u[idxU(g, 3, 2)], 0, 'right face');
  assert.equal(f.v[idxV(g, 2, 2)], 0, 'bottom face');
  assert.equal(f.v[idxV(g, 2, 3)], 0, 'top face');
  // One cell further out is fluid-fluid and must survive.
  assert.equal(f.u[idxU(g, 1, 2)], 1, 'a fluid-fluid face was cleared');
});

test('commitLabels leaves PRESCRIBED domain-boundary faces alone', () => {
  // The regression this exists for: grid.ts's isSolidOrOutside() reports true outside
  // the domain, so reusing it here would zero the outer faces — silently
  // deleting a jet inlet or a free-stream velocity every time labels changed.
  const f = createFields(g, Float64Array);
  f.u.fill(2);
  f.label[idxP(g, 3, 3)] = Cell.Solid;

  commitLabels(g, f);

  for (let j = 0; j < g.ny; j++) {
    assert.equal(f.u[idxU(g, 0, j)], 2, `inlet face u[0,${j}] was cleared`);
    assert.equal(f.u[idxU(g, g.nx, j)], 2, `outlet face u[nx,${j}] was cleared`);
  }
});

test('commitLabels pins p = 0 in Air and clears dye from Solid', () => {
  const f = createFields(g, Float64Array);
  f.p.fill(5);
  for (const c of f.dye) c.fill(1);
  const air = idxP(g, 5, 1);
  const solid = idxP(g, 2, 2);
  f.label[air] = Cell.Air;
  f.label[solid] = Cell.Solid;

  commitLabels(g, f);

  assert.equal(f.p[air], 0, 'Air is the Dirichlet value the sweep reads');
  // Dye stamped before the obstacle existed would otherwise sit frozen inside
  // it forever AND be picked up by neighbouring cells' backtraces.
  for (const c of f.dye) assert.equal(c[solid], 0);
  // Fluid cells are untouched.
  assert.equal(f.p[idxP(g, 1, 1)], 5);
  assert.equal(f.dye[0][idxP(g, 1, 1)], 1);
});

test('applyOutflow extrapolates onto an Air cell boundary face, and only there', () => {
  const f = createFields(g, Float64Array);
  for (let j = 0; j < g.ny; j++) f.u[idxU(g, g.nx - 1, j)] = j + 1;
  // Air on the right edge for the lower half only, so the same call has to
  // both act and not act within one loop.
  for (let j = 0; j < 3; j++) f.label[idxP(g, g.nx - 1, j)] = Cell.Air;

  applyOutflow(g, f.u, f.v, f.label);

  for (let j = 0; j < 3; j++) {
    assert.equal(f.u[idxU(g, g.nx, j)], j + 1, `row ${j} should be extrapolated`);
  }
  for (let j = 3; j < g.ny; j++) {
    assert.equal(f.u[idxU(g, g.nx, j)], 0, `row ${j} is a wall and must stay put`);
  }
});

test('applyOutflow is a no-op on a closed box', () => {
  const f = createFields(g, Float64Array);
  f.u.fill(3);
  f.v.fill(4);
  const u0 = f.u.slice();
  const v0 = f.v.slice();

  applyOutflow(g, f.u, f.v, f.label);

  assert.deepEqual([...f.u], [...u0]);
  assert.deepEqual([...f.v], [...v0]);
});

test('applyOutflow refuses backflow through an outlet, and only backflow', () => {
  // The energy argument in applyOutflow's header: with p = 0 on the outlet,
  // the boundary term of dE/dt is -∮ ½|u|²(u·n), which is a SOURCE wherever
  // the flow reverses. Clamping the normal component is what fixes its sign.
  const f = createFields(g, Float64Array);
  for (let j = 0; j < g.ny; j++) f.label[idxP(g, g.nx - 1, j)] = Cell.Air;
  // Row 0 leaves, row 1 reverses — one call has to treat them differently.
  f.u[idxU(g, g.nx - 1, 0)] = 2;
  f.u[idxU(g, g.nx - 1, 1)] = -2;

  applyOutflow(g, f.u, f.v, f.label);

  assert.equal(f.u[idxU(g, g.nx - 1, 0)], 2, 'outgoing flow must pass through untouched');
  assert.equal(f.u[idxU(g, g.nx, 0)], 2, 'and be extrapolated onto the ghost face');
  assert.equal(f.u[idxU(g, g.nx - 1, 1)], 0, 'inflow through the outlet must be clamped away');
  assert.equal(f.u[idxU(g, g.nx, 1)], 0, 'the ghost face may never disagree with the outlet');
});

test('applyOutflow clamps toward each edge OWN outward normal', () => {
  // A low-edge outlet leaves in -x, so there the sign convention flips and
  // POSITIVE u is the backflow. Getting this wrong turns the clamp into a
  // wall that blocks the outlet entirely.
  const f = createFields(g, Float64Array);
  for (let j = 0; j < g.ny; j++) f.label[idxP(g, 0, j)] = Cell.Air;
  f.u[idxU(g, 1, 0)] = -2; // leaving through the left edge
  f.u[idxU(g, 1, 1)] = 2; // entering through it

  applyOutflow(g, f.u, f.v, f.label);

  assert.equal(f.u[idxU(g, 1, 0)], -2, 'outgoing flow must pass through untouched');
  assert.equal(f.u[idxU(g, 0, 0)], -2);
  assert.equal(f.u[idxU(g, 1, 1)], 0, 'inflow through a left-edge outlet must be clamped');
  assert.equal(f.u[idxU(g, 0, 1)], 0);
});
