# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import fp4_kv_admission as A
sys.path.insert(0,str(ROOT/'overlay/bringup'))
from glm_fp4_prefill import scratch_bytes, LOGITS_BYTES


class Admission(unittest.TestCase):
    def test_calibration_reproduces_both_reported_losses_and_rank0_extra(self):
        self.assertAlmostEqual(A.long_prefill_bytes('fp8',61440)[0]/A.GiB,.605)
        self.assertAlmostEqual(A.long_prefill_bytes('fp4x',50000)[0]/A.GiB,1.013)
        self.assertGreater(A.long_prefill_bytes('fp4x',50000)[0],A.long_prefill_bytes('fp8',50000)[0])
        self.assertEqual(A.long_prefill_bytes('fp4x',0),[0]*4)

    def test_largest_grid_point_and_next_point_under_both_rules(self):
        sim=os.environ['FP4_SIM_DIR']
        safe=A.safe_length(sim,'fp4x')
        here=A.forecast(sim,'fp4x',safe); next_=A.forecast(sim,'fp4x',safe+64)
        self.assertTrue(here['passes_8_5']);self.assertTrue(here['passes_live_8_0'])
        self.assertFalse(next_['passes_8_5']);self.assertLess(safe,80000)
        self.assertFalse(A.forecast(sim,'fp4x',100288)['passes_live_8_0'])
        self.assertEqual(here['per_lever_bytes']['index_gather_growth'],0)
        self.assertEqual(here['per_lever_bytes']['tiled_prefill_bank'],scratch_bytes()-LOGITS_BYTES)
        self.assertEqual(A.forecast(sim,'fp4x',100288)['per_lever_bytes']['tiled_prefill_bank'],scratch_bytes()-LOGITS_BYTES)
        live=A.safe_length(sim,'fp4x',live_only=True)
        self.assertTrue(A.forecast(sim,'fp4x',live)['passes_live_8_0'])
        self.assertFalse(A.forecast(sim,'fp4x',live+64)['passes_live_8_0'])
        self.assertTrue(A.forecast(sim,'fp8',66112)['passes_8_5'])


if __name__=='__main__':unittest.main()
