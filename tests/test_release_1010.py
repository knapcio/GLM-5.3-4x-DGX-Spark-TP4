"""Frozen release selectors through shell profile loading and every container rank."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from release_table import tables
from public_export_audit import audit


class FrozenRelease(unittest.TestCase):
    def test_readme_keeps_full_rigmark_table(self):
        readme = (ROOT/'README.md').read_text()
        self.assertIn('## RigMark 1.0.0 (thinking on, effort low)', readme)
        self.assertIn('| prose | **30.051** | 29.057–30.674 |', readme)
        self.assertIn('| code | 34.89 [37.37] | 46.99 [25.20] | 67.90 [18.33] |', readme)
        self.assertIn('| warm prefix-cache replay | 16,114.3 |', readme)

    def test_public_tree_passes_export_audit(self):
        self.assertTrue(audit(ROOT)['passed'])

    def test_public_operational_helpers_are_portable_and_match_docs(self):
        dispram = (ROOT/'scripts/dispram.sh').read_text()
        start = (ROOT/'start.sh').read_text()
        coordinator = (ROOT/'tests/run_pad_hygiene_v2_coordinator.sh').read_text()
        self.assertIn('HOME_D=${DISPRAM_HOME:-/srv/glm-dispram}', dispram)
        self.assertIn('service|monitor)', dispram)
        self.assertIn('if [ "$1" = service ]; then "$0" start; else "$0" watch >/dev/null; fi', dispram)
        self.assertIn("case \"$hard\" in none) ;; *[!0-9]*|'')", dispram)
        self.assertIn('[ "$hard" != none ]', dispram)
        self.assertIn('stop|service|monitor|service-teardown', dispram)
        self.assertIn('DISPRAM_HOME=${DISPRAM_HOME:-/srv/glm-dispram}', start)
        self.assertIn('GLM_CAMPAIGN_DAY', coordinator)
        self.assertIn('task_repo=$(cd "$(dirname "$0")/.." && pwd)', coordinator)
        self.assertNotIn('/srv/projects/', coordinator)

    def test_profile_reaches_all_ranks_without_experimental_controls(self):
        code = '''import json,runpy
mod=runpy.run_path('scripts/cluster.py')
mod['dispram'].__globals__['dispram_guard_present']=lambda:True
print(json.dumps([mod['rank_env'](i) for i in range(4)]))
'''
        env = dict(os.environ)
        for key in list(env):
            if key.startswith(('GLM_', 'RECIPE_')):
                del env[key]
        env.update(RECIPE_ROOT=str(ROOT), RECIPE_HOSTS='rank0 rank1 rank2 rank3',
                   RECIPE_IPS='192.0.2.1 192.0.2.2 192.0.2.3 192.0.2.4', IMAGE='image:local',
                   MODEL_DIR='/models/target', DRAFT_DIR='/models/draft', NCCL_HOST_DIR='/nccl',
                   OVERLAY_REMOTE='/runtime', FABRIC_IFACE='fabric0', IB_HCA='hca0,hca1', DRY='1')
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py') as script:
            script.write(code); script.flush()
            command = 'source profiles/current.env; source .env.example; '
            command += 'export MODEL_DIR=/models/target NCCL_HOST_DIR=/nccl OVERLAY_REMOTE=/runtime '
            command += 'GLM_ATTN_NVFP4_DIR=/sidecars/attn GLM_NVFP4_MORE_DIR=/sidecars/more; exec "$1" -B "$2"'
            ranks = json.loads(subprocess.check_output(
                ['bash', '-c', command, 'release-check', sys.executable, script.name],
                cwd=ROOT, env=env, text=True))
        expected = dict(GLM_MOE_DET_ALIGN='1', GLM_DRAFT_HEAD='nvfp4', GLM_DRAFT_HEAD_INIT='1',
                        GLM_DRAFT_EHPROJ='fp8', GLM_DRAFT_EHPROJ_INIT='1', GLM_GLUE_ROUTER_BF16='1',
                        GLM_GLUE_MOE_WS='1', GLM_GLUE_DSA_IDX_CACHE='0', GLM_MLA_SPLIT_K='32',
                        GLM_MTP_KSTOP_UNIFORM_BATCH='k2')
        for rank in ranks:
            self.assertEqual({key: rank[key] for key in expected}, expected)
            self.assertNotIn('GLM_MTP_ROWSELECT', rank)
            self.assertFalse(any(any(term in key for term in ('TAU', 'K4', 'CACHE_TRIM', 'ASYNC_V2',
                                                             'NOSPLIT', 'ROPE_I8')) for key in rank))
        profile = subprocess.check_output(['bash', '-c', 'source profiles/current.env; '
            'printf "%s %s %s %s %s" "$RECIPE_LIVE_FLOOR_GIB" "$RECIPE_STRESS_FLOOR_GIB" '
            '"$RECIPE_CAPTURE_HEADROOM_GIB" "$RECIPE_ADMISSION_FLOOR_GIB" "$GLM_PRECAPTURE_FLOOR_GIB"'],
            cwd=ROOT, env=env, text=True)
        self.assertEqual(profile, '4.5 4.5 6.0 6.5 7.5')

    @staticmethod
    def receipt():
        rows = []
        for kind in ('prose', 'code', 'structured', 'json'):
            for concurrency, count in ((1, 5), (2, 3), (4, 3), (8, 2)):
                for run in range(count):
                    rows.append(dict(phase='scored', kind=kind, concurrency=concurrency,
                        result=dict(results=[dict(streamsOk=concurrency, streamsFailed=0,
                            aggregateDecodeTps=30+run, meanDecodeTps=15+run)])))
        for _ in range(3):
            rows.append(dict(phase='prefill-scored', result=dict(results=[
                dict(targetTokens=size, prefillTps=900, ttftMs=4200)
                for size in (4096, 8192, 16384, 32768)])))
        rows.append(dict(mode='full', sweep=dict(max_model_len=262144)))
        return rows

    def test_table_medians_and_incomplete_or_failed_receipts(self):
        rows = self.receipt()
        self.assertIn('| prose | 32.00 | 31.00 [16.00]', tables(rows))
        self.assertIn('| time to first token (s) | 4.20 | 4.20 | 4.20 | 4.20 |', tables(rows))
        with self.assertRaises(ValueError):
            tables(rows[1:])
        rows[0]['result']['results'][0]['streamsFailed'] = 1
        with self.assertRaises(ValueError):
            tables(rows)
        rows = self.receipt()
        rows[0]['result']['results'][0]['aggregateDecodeTps'] = float('nan')
        with self.assertRaises(ValueError):
            tables(rows)


if __name__ == '__main__':
    unittest.main(verbosity=2)
