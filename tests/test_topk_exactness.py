# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for the exact checker used by the GPU top-k fixture."""
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent/'fixtures'))
from topk_exactness import assert_topk_exact


class TopkExactness(unittest.TestCase):
    def check(self, scores, actual, expected, start=0, end=None):
        logits = torch.tensor(scores, dtype=torch.float32)
        assert_topk_exact(logits, torch.tensor(actual, dtype=torch.int32),
                          torch.tensor(expected, dtype=torch.int32), start,
                          len(scores) if end is None else end)

    def test_cutoff_tie_swaps_and_order_pass(self):
        self.check([9, 8, 4, 4, 4, 1], [4, 0, 1], [0, 1, 2])
        self.check([0]*6, [5, 3, 1], [0, 2, 4])
        self.check([999, 9, 4, 4, 1, 999], [2, 0], [0, 1], 1, 5)

    def test_strict_score_ids_and_below_cutoff_rejected(self):
        for actual, expected in (([0, 2, 3], [0, 1, 2]),
                                 ([0, 1, 5], [0, 1, 2]),
                                 ([0, 2, 3], [0, 2, 4])):
            with self.subTest(actual=actual, expected=expected), self.assertRaises(AssertionError):
                self.check([9, 8, 4, 4, 4, 1], actual, expected)

    def test_score_bits_and_noncutoff_tie_swaps_rejected(self):
        # Equal numeric zeros do not permit replacing +0 with -0.
        with self.assertRaisesRegex(AssertionError, 'selected score bits'):
            self.check([9, 0., -0.], [0, 2], [0, 1])
        # Both selections have the same score-bit multiset; crossed zero
        # swaps still violate the requirement that changed IDs share bits.
        with self.assertRaisesRegex(AssertionError, 'cutoff swap score bits'):
            self.check([9, 0., -0., 0., -0.], [0, 3, 4], [0, 1, 2])
        with self.assertRaisesRegex(AssertionError, 'strict-score IDs'):
            self.check([9, 9, 4, 4], [1, 2], [0, 2])
        # One-ULP score differences cannot be treated as ties.
        above = torch.nextafter(torch.tensor(4.), torch.tensor(5.)).item()
        with self.assertRaises(AssertionError):
            self.check([9, above, 4], [0, 2], [0, 1])

    def test_padding_bounds_duplicates_and_empty_rows(self):
        self.check([3, 2], [1, -1, 0, -1], [0, 1, -1, -1])
        self.check([999], [-1, -1], [-1, -1], 1, 1)
        for actual in ([0, 0, -1, -1], [0, -1, -1, -1],
                       [0, 2, -1, -1], [0, 1, -2, -1]):
            with self.subTest(actual=actual), self.assertRaises(AssertionError):
                self.check([3, 2], actual, [0, 1, -1, -1])


if __name__ == '__main__':
    unittest.main()
