"""Public start.sh defaults against the frozen config.env, without feature overrides."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'overlay/bringup'))
import glm_decode_fair

FIXTURE = ROOT / 'tests/fixtures/release-1010-gate-env.json'
FAIR_FIXTURE = ROOT / 'tests/fixtures/release-1010-decode-fair.json'
FIXTURE_SHA256 = '5d11ea6f5b375778388bfe72660e3f0d6ac25e042c65cb9da23cc6807355a19e'

# Every exclusion is deliberate; feature equivalents are checked below and documented
# in docs/release-1010-gate.md. Never fill missing vector keys from the gate fixture.
EXCLUSIONS = {
    'GLM_PARAM_HASH': 'Gate diagnostic hashing, not a serving feature default.',
    'VLLM_SERVER_DEV_MODE': 'Private diagnostic RPC API; public draft INIT enables it independently.',
    'RECIPE_LIVE_FLOOR_GIB': 'Host launcher floor, never a container env key.',
    'GATE_CTN': 'Host gate container identity, not a feature (absent in this pinned config).',
    'GLM_DECODE_FAIR_CONTROL': 'Optional runtime sidecar; public boot policy is the same 4096/40.',
    'GLM_SKIP_MLA_PLAN': 'Gate A/B harness starts off with AB_INIT=0; public skip defaults off.',
    'GLM_SKIP_MLA_PLAN_AB_INIT': 'Initial state of that optional A/B harness; public skip is off.',
    'GLM_PREFILL_SOLO_CAPACITY': 'Gate launcher validation-only key; public constructor capacity is 4096.',
    'GLM_PREFILL_SOLO_CHUNK': 'Gate launcher validation-only key; 0 retains ordinary adaptive prefill.',
    'GLM_KSTOP_ASYNC_PAYLOAD': 'Gate value 0 disables async v2, which is not shipped publicly.',
    'GLM_MTP_TAU_AB': 'Runtime tau trial harness; public fixed control has the same boot tau 0.74.',
    'GLM_MTP_TAU_INIT': 'Trial harness boot override; public control.json already selects 0.74.',
    'GLM_MTP_K4_CAPTURE': 'Trial graph reservation is 0; public speculative width remains K3.',
    'GLM_MTP_K4_GATE': 'Runtime K4 trial is 0; public runtime has no K4 implementation.',
}


class PublicGateDefaults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text())
        cls.gate = cls.fixture['env']
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            site = tmp / 'site.env'
            site.write_text('''HOSTS=(rank0 rank1 rank2 rank3)
IPS=(192.0.2.1 192.0.2.2 192.0.2.3 192.0.2.4)
FABRIC_IFACE=fabric0
IB_HCA=hca0,hca1
MODEL_DIR=/models/target
DRAFT_DIR=/models/draft
NCCL_HOST_DIR=/nccl
OVERLAY_REMOTE=/runtime
IMAGE=image:local
NCCL_SHA256=''' + '0' * 64 + '''
GLM_ATTN_NVFP4_DIR=/sidecars/attn
GLM_NVFP4_MORE_DIR=/sidecars/more
''')
            # Exercise the real shell entry point and DRY main(). Only the presence
            # check for the separately built GPU copy guard is substituted, as in
            # test_release_1010.py; no profile/env/command generation is mocked.
            python = tmp / 'python3'
            python.write_text('#!' + sys.executable + '\n' + '''import runpy, sys
from unittest.mock import patch
path = sys.argv.pop(1)
mod = runpy.run_path(path)
with patch.dict(mod['main'].__globals__, dispram_guard_present=lambda: True):
    mod['main']()
''')
            python.chmod(0o755)
            for name in ('ssh', 'docker', 'curl', 'scp', 'rsync'):
                forbidden = tmp / name
                forbidden.write_text('#!/bin/sh\necho "forbidden transport" >&2\nexit 99\n')
                forbidden.chmod(0o755)
            # Allowlist OS essentials only: inherited GLM/RECIPE/VLLM settings cannot
            # hide a profile mismatch. Site .env contains paths/hosts, no selectors.
            env = {key: os.environ[key] for key in ('HOME', 'TMPDIR', 'SYSTEMROOT') if key in os.environ}
            env.update(PATH=str(tmp) + os.pathsep + os.environ.get('PATH', os.defpath),
                       DRY='1', RECIPE_CONFIG=str(site), PYTHONDONTWRITEBYTECODE='1')
            cls.rendered = subprocess.check_output([str(ROOT / 'start.sh'), 'serve'],
                                                   cwd=ROOT, env=env, text=True, timeout=30)
        cls.vectors = []
        for line in cls.rendered.splitlines():
            if line.startswith('docker run '):
                args = shlex.split(line)
                values = [args[i + 1].split('=', 1) for i, token in enumerate(args) if token == '-e']
                if len(values) != len(dict(values)):
                    raise AssertionError('Duplicate docker environment key')
                cls.vectors.append((args, dict(values)))

    def test_gate_fixture_is_pinned(self):
        self.assertEqual(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(), FIXTURE_SHA256)
        self.assertEqual(self.fixture['source_sha256'],
                         'f9bd4b1e31d6dda93702842125374ab872d3b5707e93101c2be99f36b97129e3')

    def test_every_docker_vector_matches_gate_feature_env(self):
        self.assertEqual(len(self.vectors), 8)  # Four JIT prep and four serving vectors.
        self.assertEqual(sum('--rm' in args for args, _ in self.vectors), 4)
        names = [args[args.index('--name') + 1] for args, _ in self.vectors]
        self.assertEqual(len(set(names)), 8)
        for args, env in self.vectors:
            for key, value in self.gate.items():
                if key not in EXCLUSIONS:
                    with self.subTest(container=args[args.index('--name') + 1], key=key):
                        self.assertEqual(env.get(key), value)
            self.assertEqual(env['GLM_GLUE_ROUTER_EXPECT'], 'target:75,mtp:1')

    def test_effective_decode_fair_boot_matches_gate_sidecar(self):
        sidecar = FAIR_FIXTURE.read_text()
        self.assertEqual(json.loads(sidecar),
                         dict(schema=2, chunk=4096, decode_steps=40, sequence=0))

        def resolve(env):
            # Same resolver calls as glm_adaptive_chunk.install's scheduler factory.
            enabled, chunk, path = glm_decode_fair.settings(env)
            chunk, steps = glm_decode_fair.control_policy(
                SimpleNamespace(), path, chunk, glm_decode_fair.boot_decode_steps(env))
            return dict(enabled=enabled, chunk=chunk, decode_steps=steps)

        expected = dict(enabled=True, chunk=4096, decode_steps=40)
        for args, env in self.vectors:
            name = args[args.index('--name') + 1]
            # The pinned config.env holds gate overrides to the base launch vector.
            # Preserve the literal /cache path; substitute only its file contents.
            gate_env = dict(env, **self.gate)
            with self.subTest(container=name, boot='gate'):
                with patch.object(Path, 'read_text', autospec=True, return_value=sidecar) as read:
                    gate = resolve(gate_env)
                read.assert_called_once_with(Path('/cache/decode-fair.json'))
                self.assertEqual(gate, expected)
            for path in ('', None):
                with self.subTest(container=name, public_sidecar=path):
                    public_env = dict(env)
                    if path is None:
                        public_env.pop('GLM_DECODE_FAIR_CONTROL')
                    else:
                        public_env['GLM_DECODE_FAIR_CONTROL'] = path
                    with patch.object(Path, 'read_text', side_effect=AssertionError('sidecar read')):
                        public = resolve(public_env)
                    self.assertEqual(public, expected)
                    self.assertEqual(public, gate)

    def test_readme_documents_optional_decode_fair_sidecar(self):
        readme = (ROOT / 'README.md').read_text()
        self.assertIn('optional `GLM_DECODE_FAIR_CONTROL` sidecar', readme)
        self.assertIn('default boot needs no sidecar', readme)

    def test_excluded_controls_have_the_documented_boot_equivalents(self):
        expected = dict(GLM_DECODE_FAIR_CONTROL='/cache/decode-fair.json', GLM_SKIP_MLA_PLAN='ab',
                        GLM_SKIP_MLA_PLAN_AB_INIT='0', GLM_PREFILL_SOLO_CAPACITY='4096',
                        GLM_PREFILL_SOLO_CHUNK='0', GLM_KSTOP_ASYNC_PAYLOAD='0', GLM_MTP_TAU_AB='1',
                        GLM_MTP_TAU_INIT='0.74', GLM_MTP_K4_CAPTURE='0', GLM_MTP_K4_GATE='0')
        self.assertEqual({key: self.gate[key] for key in expected}, expected)
        control = json.loads((ROOT / 'overlay/kstop/control.json').read_text())
        self.assertEqual(control['tau'], float(self.gate['GLM_MTP_TAU_INIT']))
        self.assertEqual(control['mode'], 'k3-stop')
        for args, env in self.vectors:
            self.assertEqual(env['GLM_DECODE_FAIR_CONTROL'], '')
            self.assertEqual(env['GLM_DECODE_FAIR'], '1')
            self.assertEqual(env['GLM_DECODE_FAIR_CHUNK'], '4096')
            self.assertEqual(env['GLM_DECODE_FAIR_DECODE_STEPS'], '40')
            self.assertEqual(env['GLM_MTP_KSTOP'], '1')
            self.assertEqual(env['GLM_MTP_KSTOP_UNIFORM_BATCH'], 'k2')
            for key in expected.keys() - {'GLM_DECODE_FAIR_CONTROL'}:
                self.assertNotIn(key, env)
            if '--rm' not in args:
                self.assertEqual(args[args.index('--max-num-batched-tokens') + 1], '4096')
                spec = json.loads(args[args.index('--speculative-config') + 1])
                self.assertEqual(spec['num_speculative_tokens'], 3)
        # Trial implementations must remain absent, rather than becoming active
        # behind omitted env keys. Existing CPU policy tests exercise their replacements.
        runtime = (ROOT / 'overlay/kstop/kstop_runtime.py').read_text()
        for key in ('GLM_MTP_TAU_AB', 'GLM_MTP_TAU_INIT', 'GLM_MTP_K4_CAPTURE',
                    'GLM_MTP_K4_GATE', 'GLM_KSTOP_ASYNC_PAYLOAD'):
            self.assertNotIn(key, runtime)


if __name__ == '__main__':
    unittest.main(verbosity=2)
