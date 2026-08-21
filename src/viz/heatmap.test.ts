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
