# SPDX-License-Identifier: Apache-2.0
"""Host-memory safety gates with no /proc writes, network, or fleet actions."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import page_cache_policy as P
import stress_step as S
from fp4_kv_layout import layout, aligned_bytes, CARVE_BYTES


def memory(**kw):
    return dict(MemAvailable=9*1048576,MemFree=2*1048576,Cached=10*1048576,
                Shmem=1048576,Mapped=1048576,Dirty=0,Writeback=0,**kw)


class Safety(unittest.TestCase):
    def test_page_cache_guards_and_double_check_after_sync(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'lock').touch()
            dropped=[];sync=[]
            def run():return P.perform('flush',p,read=memory,sync=lambda:sync.append(1),drop=lambda:dropped.append(1))
            self.assertEqual(run()['reason'],'disabled')
            (p/'enabled').touch()
            self.assertEqual(run()['reason'],'no-fresh-serving-lease')
            P.perform('begin',p)
            self.assertEqual(run()['reason'],'checkpoint-loading')
            P.perform('ready',p)
            self.assertEqual(run()['action'],'drop_caches=1')
            self.assertEqual((len(sync),len(dropped)),(1,1))
            # Disable during sync: no write to drop_caches follows.
            r=P.perform('flush',p,read=memory,sync=lambda:(p/'enabled').unlink(),drop=lambda:dropped.append(1))
            self.assertEqual(r['action'],'skip-after-sync');self.assertEqual(len(dropped),1)
            (p/'enabled').touch();os.utime(p/'ready',(time.time()-100,time.time()-100))
            self.assertEqual(run()['reason'],'no-fresh-serving-lease')

    def test_busy_writeback_and_low_floor_skip_without_sync(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'lock').touch();(p/'enabled').touch();P.perform('ready',p)
            for key,value,why in [('Dirty',65537,'writeback-busy'),('MemAvailable',7*1048576,'below-serving-floor'),
                                  ('Cached',3*1048576,'cache-below-4GiB'),('MemFree',5*1048576,'enough-free-pages')]:
                m=memory();m[key]=value
                sync=[];r=P.perform('flush',p,read=lambda:m,sync=lambda:sync.append(1),drop=lambda:self.fail('drop'))
                self.assertEqual(r['reason'],why);self.assertEqual(sync,[])

    def test_exact_capacity_and_literal_c4_refusal(self):
        for n,blocks,length in [(0,1573,98176),(1,2097,131712),(2,2622,165312),(3,3147,198912)]:
            g=S.geometry(n)
            self.assertEqual((g['layout']['blocks'],g['max_model_len']),(blocks,length))
            self.assertFalse(g['c4_resident_possible'])
            q=S.geometry(n,'pool-quarter')
            self.assertTrue(q['c4_resident_possible']);self.assertFalse(q['qualifies_requested_worst_case'])
            self.assertLessEqual(aligned_bytes('fp4x',blocks),g['ordinary_bytes']+CARVE_BYTES)
            self.assertGreater(aligned_bytes('fp4x',blocks+1),g['ordinary_bytes']+CARVE_BYTES)
        for head in (0,-1,1073741825,'2147483648'):
            with self.assertRaises(ValueError):layout('fp4x',head)

    def test_broken_cache_policy_cannot_prevent_container_stop(self):
        e=dict(RECIPE_ROOT=str(ROOT),RECIPE_HOSTS='s1 s2 s3 s4',RECIPE_IPS='1 2 3 4',IMAGE='image')
        with patch.dict(os.environ,e):
            spec=importlib.util.spec_from_file_location('stop_headroom_cluster',ROOT/'scripts/cluster.py')
            c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
            calls=[]
            def remote(rank,command,timeout=80):
                calls.append(command)
                return 'false' if command.startswith('docker inspect') else ''
            with patch.object(c,'page_cache_state',side_effect=RuntimeError('helper missing')), \
                 patch.object(c,'remote',side_effect=remote),patch.object(c,'dispram',return_value=None), \
                 patch.object(c,'unlock') as unlock:
                with self.assertRaisesRegex(RuntimeError,'Containers stopped'):
                    c.stop({'ctn':'glm53full-kvhr-test','token':'owner'})
                self.assertEqual(sum(x.startswith('docker stop') for x in calls),4)
                unlock.assert_not_called()

    def test_expanded_prefill_arena_covers_the_physical_pool(self):
        sys.path.insert(0,str(ROOT/'overlay/bringup'))
        from glm_fp4_prefill import capacities, scratch_bytes
        for n in range(4):
            g=S.geometry(n)
            keys,rows,workspace=capacities(g['layout']['tokens'],g['max_model_len'])
            self.assertEqual(rows,g['layout']['tokens'])
            self.assertGreaterEqual(workspace,rows*1152)
            self.assertGreaterEqual(keys,4*g['max_model_len'])
            self.assertGreaterEqual(scratch_bytes(pool_tokens=rows,max_model_len=g['max_model_len']),workspace+(256<<20))
        self.assertEqual(capacities(),(524288,100672,115974144))

    def test_missing_metrics_are_not_silently_treated_as_zero(self):
        with self.assertRaisesRegex(RuntimeError,'required metric missing'):
            S.metric('# no preemption counter', 'vllm:num_preemptions_total')
        self.assertEqual(S.metric('vllm:num_preemptions_total{engine="0"} 3\n','vllm:num_preemptions_total'),3)

    def test_mac_execution_refuses_before_any_subprocess(self):
        with patch.object(S.platform,'system',return_value='Darwin'),patch.object(S.subprocess,'run') as run:
            with self.assertRaisesRegex(ValueError,'rank0'):S.execute(argparse.Namespace(execute=True))
            run.assert_not_called()

    def test_guard_monitor_interrupts_before_blocked_http_wait(self):
        with tempfile.TemporaryDirectory() as d:
            mon=S.Monitor(Path(d),'glm53full-kvhr-test')
            for rank in range(4):mon.latest[rank]=(time.monotonic()-10,{})
            calls=[]
            mon.on_trip=lambda:(calls.append(1),mon.closed.set())
            mon.watch()
            self.assertEqual(calls,[1]);self.assertIn('stale',mon.problem)

    def test_block_receipts_raise_on_wrong_rank_head(self):
        with tempfile.TemporaryDirectory() as d:
            g=S.geometry(1,'pool-quarter');m={'boot':'glm53full-kvhr-test','geometry':g}
            wrong='glm-dispram-kv: '+json.dumps({'event':'pool','head':1<<30,'num_blocks':2097})
            with patch.dict(os.environ,RECIPE_HOSTS='rank0 rank1 rank2 rank3'), \
                 patch.object(S.subprocess,'run',return_value=argparse.Namespace(stdout=wrong)):
                with self.assertRaisesRegex(RuntimeError,'receipt mismatch'):S.receipts(None,m,Path(d))

    def test_launcher_larger_head_reaches_all_ranks_and_refuses_wrong_modes(self):
        e=dict(RECIPE_ROOT=str(ROOT),RECIPE_HOSTS='s1 s2 s3 s4',RECIPE_IPS='1 2 3 4',IMAGE='image',
               GLM_KV_FORMAT='fp4x',GLM_FULL_MLA='triton',VLLM_USE_V2_MODEL_RUNNER='1',GLM_MLA_SPLIT_K='32',
               GLM_MTP_KSTOP='1',RECIPE_DISPRAM='require',RECIPE_KV_HEAD_BYTES=str(2<<30),RECIPE_MAX_MODEL_LEN='131712')
        with patch.dict(os.environ,e):
            spec=importlib.util.spec_from_file_location('headroom_cluster',ROOT/'scripts/cluster.py')
            c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
            args=json.loads((ROOT/'profiles/serve-args.json').read_text())
            with patch.object(c,'dispram',return_value=object()):
                for rank in range(4):
                    shaped=c.launch_shape(args)
                    self.assertEqual(shaped[shaped.index('--kv-cache-memory-bytes')+1],str(2<<30))
                    self.assertEqual(c.kstop_layout(c.launch_switches())['blocks'],2097)
                for changes in ({'GLM_KV_FORMAT':'fp8'},{'RECIPE_KV_HEAD_BYTES':str((2<<30)+1)},
                                {'RECIPE_MAX_MODEL_LEN':'200000'},{'RECIPE_DISPRAM':'auto'}):
                    with patch.dict(os.environ,changes),self.assertRaises(ValueError):c.launch_switches()


if __name__=='__main__':unittest.main()
