import { test } from 'node:test';
import assert from 'node:assert/strict';
import { computeRange } from './heatmap.ts';

test('computeRange: auto finds actual min/max', () => {
  const { lo, hi } = computeRange(new Float64Array([-3, 1, 2]), {
    kind: 'auto',
  });
  assert.equal(lo, -3);
  assert.equal(hi, 2);
});

test('computeRange: symmetric is centered on the true max magnitude', () => {
  // regression case: a negative value earlier in the array must not get
  // overwritten by a smaller positive value later on.
  const { lo, hi } = computeRange(new Float64Array([-10, 1]), {
    kind: 'symmetric',
  });
  assert.equal(lo, -10);
  assert.equal(hi, 10);
});

test('computeRange: fixed passes bounds through untouched', () => {
  const { lo, hi } = computeRange(new Float64Array([0, 0, 0]), {
    kind: 'fixed',
    min: -5,
    max: 5,
  });
  assert.equal(lo, -5);
  assert.equal(hi, 5);
});

test('computeRange: percentile ignores a sparse extreme', () => {
  // 1000 values at magnitude 1, one at 100 — the shape of a vorticity field
  // whose peak is a handful of cells on an obstacle surface.
  const field = new Float64Array(1000).fill(1);
  field[0] = 100;

  const max = computeRange(field, { kind: 'symmetric' });
  assert.equal(max.hi, 100, 'symmetric is dominated by the single outlier');

  const p99 = computeRange(field, { kind: 'percentile', p: 0.99 });
  assert.ok(p99.hi < 5, `p99 = ${p99.hi}, should sit near the bulk at 1`);
  assert.equal(p99.lo, -p99.hi, 'must stay symmetric so zero lands mid-ramp');
});

test('computeRange: percentile of a flat field is not zero-width', () => {
  // A bin index landing in the first bucket must still return a usable range,
  // or the colormap gets a zero-width interval and paints everything mid-ramp.
  const field = new Float64Array(100).fill(3);
  const r = computeRange(field, { kind: 'percentile', p: 0.5 });
  assert.ok(r.hi > 0, `expected a positive half-width, got ${r.hi}`);

  const zero = computeRange(new Float64Array(10), { kind: 'percentile', p: 0.99 });
  assert.deepEqual(zero, { lo: 0, hi: 0 }, 'an all-zero field has no scale');
});
