# SPDX-License-Identifier: Apache-2.0
"""CPU image tests for actual vLLM loader contract validation; no GPU."""
import copy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import boot_preflight as B
B.overlay_paths(ROOT)
if not sys.platform.startswith('linux'):raise unittest.SkipTest('run CPU loader contracts in the serving image')
import torch
from vllm.model_executor.models.utils import AutoWeightsLoader


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.model=torch.nn.Module();self.model.embed_tokens=torch.nn.Module()
        self.model.embed_tokens.register_parameter('weight',torch.nn.Parameter(torch.empty((16,8),dtype=torch.bfloat16,device='meta')))
    def load_weights(self,weights):return AutoWeightsLoader(self).load_weights(weights)


class RealLoaderContracts(unittest.TestCase):
    def data(self):
        return dict(tensors={'model.embed_tokens.weight':dict(dtype='BF16',shape=[16,8])},cache=dict(sidecars=[]))
    def check(self,data):
        with patch.dict(os.environ,{'GLM_ATTN_WEIGHTS':'int8'}):
            return B.check_parameters(dict(target=Model(),draft=Model()),data,{},B.Report())
    def test_stock_loader_accepts_matching_metadata(self):
        self.assertEqual(self.check(self.data()),dict(target=1,draft=1))
    def test_weight_shape_header_is_checked_before_value_fake(self):
        data=self.data()
        data['tensors']['model.foo.weight_shape']=dict(dtype='I64',shape=[3])
        with self.assertRaisesRegex(ValueError,'weight_shape header must be a pair'):
            self.check(data)
    def test_stock_loader_refuses_wrong_dtype_shape_name_missing(self):
        for mutation in ('dtype','shape','name','missing'):
            data=self.data()
            if mutation=='dtype':data['tensors']['model.embed_tokens.weight']['dtype']='F32'
            if mutation=='shape':data['tensors']['model.embed_tokens.weight']['shape']=[16,9]
            if mutation=='name':data['tensors']['model.wrong.weight']=data['tensors'].pop('model.embed_tokens.weight')
            if mutation=='missing':data['tensors'].clear()
            with self.subTest(mutation=mutation),self.assertRaises((ValueError,KeyError,AssertionError)):
                self.check(data)

if __name__=='__main__':unittest.main()
