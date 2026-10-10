# SPDX-License-Identifier: Apache-2.0
"""W4 default-vector isolation and exact gate regression tests."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT/'overlay/bringup'),str(ROOT/'overlay/kstop')]
import glm_rowselect_qual as Q
import glm_mtp_rowselect_config as RC
import glm_fp4_kv as KV
from test_mtp_rowselect import ENV

class Integration(unittest.TestCase):
    def test_detalign_required_and_off_cold(self):
        self.assertFalse(RC.options({}))
        self.assertTrue(RC.options(ENV))
        for value in ('0','ab',''):
            with self.assertRaises(ValueError):RC.options(ENV|{'GLM_MOE_DET_ALIGN':value})

    def test_no_float_relief_even_for_one_bit_or_signed_zero(self):
        base=[torch.zeros((1,3),dtype=torch.long),torch.ones((1,128),dtype=torch.bfloat16),
              torch.zeros(1),torch.zeros(1,11),torch.zeros(64,4,dtype=torch.bfloat16),
              torch.zeros(1,dtype=torch.long),torch.zeros(1,dtype=torch.long),
              torch.zeros(1,2,dtype=torch.long),torch.zeros(2,dtype=torch.long)]
        self.assertTrue(Q.bounded_pair(base,base,False,True)['bit_exact'])
        for i in range(len(base)):
            right=[t.clone() for t in base]
            if right[i].is_floating_point():right[i].flatten()[0]=-0.0 if i!=1 else 1.0078125
            else:right[i].flatten()[0]=1
            result=Q.bounded_pair(base,right,False,True,near_zero_relief=True)
            self.assertFalse(result['bit_exact'])
            self.assertFalse(result['drift_bounded'])

    def test_optional_features_reach_every_rank_and_default_keys_are_absent(self):
        from test_launcher import C
        values={key:'v-'+key for key in C.PROFILE_KEYS}
        values.update(dict(GLM_MOE_CANON_ALIGN='0',GLM_MTP_KSTOP_UNIFORM_BATCH='k2',GLM_MTP_KSTOP='1',GLM_MTP_FIX='1',GLM_KV_FORMAT='fp8',GLM_PAD_HYGIENE='0',
            GLM_SPEC_SAMPLE='0',GLM_INDEXER_SHORTCUT='0',VLLM_USE_V2_MODEL_RUNNER='1',
            GLM_MTP_KSTOP_CAPTURE_LAYOUT='reuse',GLM_DRAFT_HEAD='nvfp4',GLM_DRAFT_HEAD_INIT='1'))
        keys=('GLM_MTP_ROWSELECT','GLM_DRAFT_EHPROJ','GLM_DRAFT_EHPROJ_INIT')
        with patch.dict(C.ENV,dict(values,GLM_MTP_ROWSELECT='0',GLM_DRAFT_EHPROJ='0',GLM_DRAFT_EHPROJ_INIT='0')):
            self.assertFalse(any(k in C.optional_env() for k in keys))
        with patch.dict(C.ENV,dict(values,GLM_MOE_DET_ALIGN='1',GLM_MTP_ROWSELECT='1',GLM_DRAFT_EHPROJ='fp8',GLM_DRAFT_EHPROJ_INIT='1')):
            for rank in range(4):
                env=C.rank_env(rank)
                self.assertEqual([env[k] for k in keys],['1','fp8','1'])
        with patch.dict(C.ENV,dict(values,GLM_MOE_DET_ALIGN='0',GLM_MTP_ROWSELECT='1')):
            with self.assertRaises(ValueError):C.optional_env()

    def test_fp4x_f3_refused(self):
        env=dict(GLM_KV_FORMAT='fp4x', GLM_FULL_MLA='triton',
                 VLLM_USE_V2_MODEL_RUNNER='1', GLM_MLA_SPLIT_K='32')
        with self.assertRaises(ValueError):
            KV.kv_format(env|{'GLM_GLUE_DSA_IDX_CACHE':'1'})

if __name__=='__main__':unittest.main()
