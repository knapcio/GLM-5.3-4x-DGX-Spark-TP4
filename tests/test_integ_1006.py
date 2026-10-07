# SPDX-License-Identifier: Apache-2.0
"""Offline composition, exact geometry, and coordinator refusal/recovery gates."""
import argparse
from contextlib import ExitStack
import json
import os
import types
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import integ_1006 as I
import stress_step as S


class Integration(unittest.TestCase):
    def test_half_gib_ladder_is_common_exact_aligned_capacity(self):
        for extra,head,blocks,length,c4 in [(1,2,2097,131712,33472),(1.5,2.5,2360,148544,37632),
                   (2,3,2622,165312,41856),(2.5,3.5,2884,182080,46016)]:
            g=S.geometry(extra,'pool-quarter')
            self.assertEqual((g['ordinary_bytes'],g['layout']['blocks'],g['max_model_len'],g['c4_total_tokens_per_request']),
                             (int(head*I.GiB),blocks,length,c4))
            self.assertLessEqual(g['c4_required_blocks'],blocks)
            self.assertFalse(S.geometry(extra,'literal')['c4_resident_possible'])
        for extra in (.1,1.25,float('nan'),float('inf'),-1,'1'):
            with self.assertRaises(ValueError):S.geometry(extra)

    def test_preflight_advisory_does_not_block_without_cache_or_on_failure(self):
        import boot_preflight as B
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            step=Path(d);out=step/'report.json';headers=step/'headers.json'
            with patch.object(B.subprocess,'run') as run:
                I.boot_preflight(step,out)
                run.assert_not_called();self.assertEqual(json.loads(out.read_text())['status'],'SKIPPED')
            headers.write_text('{}')
            def fake(command,**kw):
                self.assertIn('--headers',command);self.assertEqual(kw['timeout'],125)
                out.write_text('{"ok":false}')
                return SimpleNamespace(returncode=1)
            with patch.object(B.subprocess,'run',side_effect=fake):
                I.boot_preflight(step,out,headers=headers,image='local-v11')
            self.assertEqual(json.loads(out.read_text())['status'],'WARNING')
            with patch.object(B.subprocess,'run',side_effect=RuntimeError('docker unavailable')):
                I.boot_preflight(step,out,headers=headers)
            self.assertEqual(json.loads(out.read_text())['status'],'WARNING')
            with patch.object(B.subprocess,'run') as run:
                I.boot_preflight(step,out,headers=headers,enabled=False)
                run.assert_not_called()

    def test_coordinator_refuses_mac_before_any_operation(self):
        with patch.object(I.platform,'system',return_value='Darwin'),patch.object(I.subprocess,'run') as run,patch.object(I,'remote') as remote:
            with self.assertRaisesRegex(ValueError,'rank0'):I.run(argparse.Namespace(execute=True))
            run.assert_not_called();remote.assert_not_called()

    def test_boolean_flag_cannot_override_sim_lowers_or_source_binding(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);g=S.geometry(1,'pool-quarter');m=dict(geometry=g,package_sha256='pkg',dry_sha256='dry')
            ad=dict(package_sha256='pkg',dry_sha256='dry',ordinary_bytes=g['ordinary_bytes'],max_model_len=g['max_model_len'],
                    blocks=g['layout']['blocks'],stress_mode='pool-quarter',per_rank_lower_GiB=[8.5]*4,
                    passes_8_5=True,reasons=[],page_cache_credit_GiB=0,sim_source_sha256='source')
            I.save(p/'admission.json',ad)
            with self.assertRaisesRegex(ValueError,'recalibration'):I.check_admission(p,m)
            (p/'model.py').write_text('# exact local simulator\n');digest=S.sha(p/'model.py')
            ad.update(sim_source_sha256=digest,calibration_sources={'model.py':digest},calibration_model='explicit per-lever costs and observed stress lower')
            I.save(p/'admission.json',ad);self.assertEqual(I.check_admission(p,m),ad)
            ad['per_rank_lower_GiB'][0]=8.499;I.save(p/'admission.json',ad)
            with self.assertRaisesRegex(ValueError,'>=8.5'):I.check_admission(p,m)
            ad['per_rank_lower_GiB'][0]=8.5;ad['page_cache_credit_GiB']=1;I.save(p/'admission.json',ad)
            with self.assertRaisesRegex(ValueError,'zero cache'):I.check_admission(p,m)

    def test_unknown_retained_lock_blocks_restore(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'fleet_busy').write_text('other owner');out=p/'out';out.mkdir()
            with patch.object(I.Path,'home',return_value=p),patch.object(I.subprocess,'run') as run:
                with self.assertRaisesRegex(RuntimeError,'recovery required'):I.halt_launcher(None,p,out)
                run.assert_not_called()
            with self.assertRaisesRegex(RuntimeError,'another coordinator'):I.acquire_window_lock(p/'fleet_busy',dict(token='serving'))
            (p/'fleet_busy').unlink();token,owned=I.acquire_window_lock(p/'fleet_busy',dict(token='serving'))
            self.assertTrue(owned);self.assertEqual((p/'fleet_busy').read_text(),token)
            I.release_window_lock(p/'fleet_busy',token);self.assertFalse((p/'fleet_busy').exists())

    def test_missing_coalesced_receipts_reject_boot(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for r in range(4):(p/f'rank{r}-boot.log').write_text('Model loading took 95.1 GiB and 210 seconds\n')
            with self.assertRaisesRegex(RuntimeError,'receipts absent'):I.boot_ledger(p,0,300,360,True)

    def test_boot_ledger_uses_exclusive_model_phases_and_separate_admission(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for r in range(4):(p/f'rank{r}-boot.log').write_text('Loading weights took 190 seconds\nLoading weights took 7 seconds\nModel loading took 95.1 GiB and 215 seconds\nGraph capturing finished in 9 secs\nglm-nvfp4-attn: loaded 385 logical attention matrices\nglm-fast-load: mtp-only 1 check PASS\n')
            I.boot_ledger(p,10,310,370)
            j=json.loads((p/'boot-breakdown.json').read_text());self.assertEqual(j['health_seconds'],300)
            self.assertEqual(j['admission_seconds'],60);self.assertEqual(j['ranks']['0']['phases']['other_construction_repack_s'],18)


    def test_failed_candidate_restores_b_only_after_successful_stop(self):
        for stop_fails in (False,True):
            with self.subTest(stop_fails=stop_fails),tempfile.TemporaryDirectory() as d:
                home=Path(d);window=home/'window';step=window/'step0';restore=window/'restore0';bcl=home/'b-clone'
                for p in (step/'clone',restore/'clone',bcl/'state',home/'.config/systemd/user/glm-serving-watch.service.d',home/'glm-control'):
                    p.mkdir(parents=True)
                g=S.geometry(0,'pool-quarter');m=dict(geometry=g,boot='control',package_sha256='digest',dry_sha256='dry',helper_sha256={})
                I.save(step/'step.json',m);(restore/'memwatch.py').write_text('# existing serving guard\n')
                I.save(restore/'restore.json',dict(boot='restored-b',package_sha256='digest',memwatch_sha256=S.sha(restore/'memwatch.py')))
                (restore/'dry.txt').write_text('dry')
                I.save(window/'window.json',dict(baseline_ctn='current-b',baseline_clone=str(bcl),last_passed_index=-1))
                I.save(bcl/'state/deployment.json',dict(ctn='current-b',token='owner'))
                (home/'fleet_busy').write_text('owner')
                (home/'.config/systemd/user/glm-serving-watch.service.d/fp4x-serving.conf').write_text('original watch')
                class Mon:
                    problem=None;on_trip=None
                    def __init__(self,*args):pass
                    def start(self):pass
                    def ready(self):return True
                    def check(self):pass
                    def close(self):pass
                class Fast(Mon):pass
                process=types.SimpleNamespace(poll=lambda:None)
                def run(cmd,**kw):
                    if cmd==['bash','./start.sh','stop'] and (home/'fleet_busy').read_text()=='owner':(home/'fleet_busy').unlink()
                    return types.SimpleNamespace(stdout='dry',returncode=0)
                def handed(step,m,p,mon,out):I.save(out/'RESTORED-B.json',dict(boot=m['boot']))
                args=argparse.Namespace(execute=True,window=window,index=0,endpoint='http://local',dash_endpoint='http://local')
                contexts=[patch.dict(os.environ,INVOCATION_ID='offline-fixture'),patch.object(I.platform,'system',return_value='Linux'),
                     patch.object(I.platform,'node',return_value='rank0'),patch.object(I.Path,'home',return_value=home),
                     patch.object(I,'verify_step'),patch.object(S,'package_hash',return_value='digest'),
                     patch.object(I,'remote',side_effect=lambda r,cmd,timeout=80: 'image' if cmd.startswith('docker image inspect') else '[{"Image":"image","State":{"Running":true}}]'),patch.object(I,'rendered_images',return_value={r:'image' for r in range(4)}),patch.object(I.subprocess,'run',side_effect=run),
                     patch.object(I.subprocess,'Popen',return_value=process),patch.object(I.signal,'signal'),
                     patch.object(S,'Monitor',Mon),patch.object(I,'LoaderSamples',Fast),patch.object(I,'wait_admission',return_value=(1,2)),
                     patch.object(S,'receipts'),patch.object(I,'boot_ledger')]
                with ExitStack() as stack:
                    for ctx in contexts:stack.enter_context(ctx)
                    smoke=stack.enter_context(patch.object(I,'smoke',side_effect=[RuntimeError('candidate failed'),'model']))
                    halt=stack.enter_context(patch.object(I,'halt_launcher',side_effect=RuntimeError('stop incomplete') if stop_fails else None))
                    handoff=stack.enter_context(patch.object(I,'handoff_restore',side_effect=handed))
                    with patch.object(I,'boot_preflight'),self.assertRaises(RuntimeError):I.run(args)
                result=json.loads((step/'run/RESULT.json').read_text())
                self.assertEqual(result['restored_B'],not stop_fails)
                self.assertEqual(handoff.call_count,0 if stop_fails else 1)
                self.assertEqual(smoke.call_count,1 if stop_fails else 2)
                self.assertEqual(result['outcome'],'RECOVERY_REQUIRED' if stop_fails else 'FAILED')


if __name__=='__main__':unittest.main()
