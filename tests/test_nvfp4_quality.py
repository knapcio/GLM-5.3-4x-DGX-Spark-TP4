# SPDX-License-Identifier: Apache-2.0
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('nq',ROOT/'bench/nvfp4_quality.py')
nq=importlib.util.module_from_spec(spec);spec.loader.exec_module(nq)

class QualityTests(unittest.TestCase):
    def test_actual_next_token_teacher_forcing(self):
        with tempfile.TemporaryDirectory() as d:
            panel=Path(d)/'panel.json';out=Path(d)/'out.json'
            with patch.object(nq,'PROMPTS',[('p','sample')]):
                def request(base,path,body):
                    if path=='/tokenize':return {'tokens':[7,9]}
                    if 'prompt_logprobs' not in body:return {'choices':[{'token_ids':[11]}]}
                    self.assertEqual(body['prompt'],[7,9,11])
                    return {'choices':[{'prompt_token_ids':[7,9,11],'prompt_logprobs':[None,{'9':{'logprob':-.1}},{'11':{'logprob':-.4}}]}]}
                with patch.object(nq,'request',side_effect=request):
                    nq.freeze('unused',panel);nq.collect('unused',panel,out)
                data=json.loads(out.read_text());other=copy.deepcopy(data)
                other['rows'][0]['logprob']=-.699
                self.assertTrue(nq.drift(data,other)['passed'])
                other['rows'][0]['logprob']=-.701
                self.assertFalse(nq.drift(data,other)['passed'])
                other['rows'][0]['tokens']=[7,9,12]
                with self.assertRaises(ValueError):nq.drift(data,other)

    def test_qeval_mean_drop_not_median(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'a';b=Path(d)/'b';a.mkdir();b.mkdir()
            def write(folder,scores):
                for i,score in enumerate(scores):
                    rows=[dict(id=str(j),category='code',**{'pass':j<score}) for j in range(75)]
                    (folder/f'qeval-{i}.json').write_text(json.dumps(dict(results=rows)))
            write(a,[71,72,73]);write(b,[71,71,72])
            self.assertTrue(nq.qgate(a,b)['passed']) # drop 2/3 tasks
            write(b,[71,71,71]);self.assertFalse(nq.qgate(a,b)['passed']) # drop 1
            (b/'qeval-2.json').unlink()
            with self.assertRaises(ValueError):nq.qgate(a,b)

    def test_needle_gate_requires_complete_identical_panel(self):
        rows=[dict(length=n,seed=s,passed=True,finish_reason='stop',content='code',expected='code',tokens_sha256=str((n,s))) for n in (16384,63488,96000) for s in range(20)]
        self.assertTrue(nq.ngate(rows,copy.deepcopy(rows))['passed'])
        other=copy.deepcopy(rows);other[0]['passed']=False
        self.assertFalse(nq.ngate(rows,other)['passed'])
        with self.assertRaises(ValueError):nq.ngate(rows,rows[:-1])
        other=copy.deepcopy(rows);other[0]['tokens_sha256']='changed'
        with self.assertRaises(ValueError):nq.ngate(rows,other)

if __name__=='__main__':unittest.main()
