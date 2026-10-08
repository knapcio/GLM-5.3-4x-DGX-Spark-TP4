#!/usr/bin/env python3
"""Workstation launcher for four TP ranks; dry rendering has no fleet access."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(os.environ['RECIPE_ROOT'])
sys.path.insert(0, str(ROOT/'scripts'))
ENV = os.environ
HOSTS = ENV['RECIPE_HOSTS'].split()
IPS = ENV['RECIPE_IPS'].split()
IMAGES = ENV.get('RECIPE_IMAGES', '').split() or [ENV['IMAGE']] * 4
STATE = ROOT / 'state/deployment.json'
SERVE_ARGS = Path(ENV.get('RECIPE_SERVE_ARGS', ROOT / 'profiles/serve-args.json'))
# RECIPE_* lines in current.env are launch-shape switches read here; they never enter a container.
PROFILE_KEYS = [k for k in re.findall(r'^export (\w+)=', (ROOT / 'profiles/current.env').read_text(), re.M)
                if not k.startswith('RECIPE_') and k not in ('GLM_ATTN_WEIGHTS', 'GLM_NVFP4_GROUPS',
                    'GLM_FP4_RECENT_WINDOW', 'GLM_FP4_RECENT_AB', 'GLM_LOADER', 'GLM_KV_FORMAT',
                    'GLM_DECODE_FAIR',
                    'GLM_DECODE_FAIR_CHUNK', 'GLM_DECODE_FAIR_DECODE_STEPS', 'GLM_DECODE_FAIR_CONTROL')]
GiB = 1 << 30
HEALTH_POLL_S = 1          # loopback /health while booting
SAMPLE_BOOT_S = 5          # rank samples (memory floors, admission window) while booting
SAMPLE_STEADY_S = 10       # rank samples once admitted
# MemAvailable floors in GiB (profiles/current.env sets the release values; these code defaults are the earlier ones).
FLOOR_DEFAULTS = dict(RECIPE_LIVE_FLOOR_GIB='6.0', RECIPE_CAPTURE_HEADROOM_GIB='8.0', RECIPE_ADMISSION_FLOOR_GIB='8.0')


def floor_gib(key):
    text = ENV.get(key, '').strip() or FLOOR_DEFAULTS[key]
    value = float(text)
    if not 2.5 <= value <= 32.0:   # 2.5: lowest guard used for a ballast test under the serving watch
        raise ValueError(key + ' must be 2.5..32 GiB')
    return value


def floors_note():
    """Validated memory floors as one line (DRY and serve log)."""
    pre = float(ENV.get('GLM_PRECAPTURE_FLOOR_GIB', '').strip() or '10')
    if not 3.0 <= pre <= 32.0:
        raise ValueError('GLM_PRECAPTURE_FLOOR_GIB must be 3..32 GiB')
    return ('Memory floors (GiB MemAvailable per rank): live %g, capture headroom %g, admission %g for 60 s, '
            'in-process pre-capture %g' % (floor_gib('RECIPE_LIVE_FLOOR_GIB'), floor_gib('RECIPE_CAPTURE_HEADROOM_GIB'),
                                           floor_gib('RECIPE_ADMISSION_FLOOR_GIB'), pre))

# Launch-shape switches and their released defaults (profiles/current.env). Every other value is refused.
LAUNCH_DEFAULTS = {'RECIPE_C4_PASS2_GRAPH': '0', 'RECIPE_MAX_MODEL_LEN': '32768', 'RECIPE_KV_PIN_L1': '0',
                   'RECIPE_NCCL_NO_LL128': '0', 'RECIPE_KV_HEAD_BYTES': str(1 << 30),
                   'RECIPE_PAGE_CACHE_POLICY': '0'}
# Context lengths with a recorded reason: the released 32,768, the full 2 GiB pool (693 blocks, one null) for
# native K2, and 44,224 for the K-stop full-pool layout (one max-length request + K3 lookahead + null = 693 blocks).
MAX_MODEL_LENS = ('32768', '44224', '44288', '66112', '100288')
# Pinned target KV per rank: the released 2 GiB (693 blocks) or the mem-ledger L1 pin of 821 blocks (2.369 GiB).
KV_PIN_BYTES = {'0': 2147483648, '1': 2543549952}
# FP8 MLA KV of this checkpoint: 79 MLA layers x 576 B + 22 indexer layers x 132 B per token, 64-token blocks.
KV_BLOCK_BYTES, KV_BLOCK_TOKENS = 3098112, 64
# GLM_MTP_KSTOP=1 launches only these layouts, keyed by RECIPE_MAX_MODEL_LEN (docs/results/kstop-memory-admission.json).
# Both capture FULL_DECODE_ONLY graphs at [1, 4, 16] with four slots: every c1 shape (M2/M3/M4, later M1) and c4 q4
# (M16, later M4) replay a graph; 14 descriptors instead of the 28 of widths 1..16. The 1..16 / 693-block layout
# (predicted rank-0 7.82 GiB, under the 8.0 floor) cannot be launched.
KSTOP_CAPTURE_SIZES = [1, 4, 16]
# Size 4 rounds to M6 q3: reuse selects it at c2, m12 pads c2 to M12.
# c3 pads to M12; c4 q3 replays M12 exactly. Retain c1 and warmup K3 graphs.
KSTOP_K2_CAPTURE_SIZES = [1, 4, 12, 16]
KSTOP_LAYOUTS = {
    # Default: sim-admitted at 8.78 [8.68, 8.85] GiB rank 0 with LL128 on; never booted.
    '32768': dict(kv_bytes=544 * KV_BLOCK_BYTES, blocks=544, requires_no_ll128=False,
                  status='sim-admitted, not booted'),
    # Optional full pool: 8.70 [8.60, 8.77] GiB only if NCCL without LL128 frees the ledger's 0.44 GiB (unmeasured).
    '44224': dict(kv_bytes=KV_PIN_BYTES['0'], blocks=693, requires_no_ll128=True,
                  status='UNMEASURED: needs the NCCL-without-LL128 memory receipt before a boot'),
}
KSTOP_LOOKAHEAD = 3
# With RECIPE_DISPRAM=require, K-stop launches only this layout: a 1 GiB ordinary KV head per rank plus the
# 2,046 MiB display carveout behind it (2,145,386,496 B; one VMM range, glm_dispram_kv) = 1,039 blocks, which
# holds one 66,112-token request (64K prompt + 512 output + 64) + K3 lookahead + the null block (1,035 blocks).
# Rank 0 after warm predicted 9.04 [8.93] GiB, pre-capture about 10.9 GiB (release-stack/stack-1002b
# admission, with only the short graphs K-stop can dispatch); never booted.
DISPRAM_CARVE_BYTES = 2145386496
KSTOP_DISPRAM_LAYOUT = dict(kv_bytes=1073741824, blocks=(1073741824 + DISPRAM_CARVE_BYTES) // KV_BLOCK_BYTES,
                            requires_no_ll128=False, max_model_len='66112',
                            status='dispram require: 1 GiB head + 2046 MiB carveout, sim-admitted, not booted')
DISPRAM_GUARD = '/overlay/guard/libdispram_copy_guard.so'

# Display-carveout KV hook (dispram by kindlingai), default off: RECIPE_DISPRAM 0 | 1/auto | require.
# The integration (scripts/dispram_recipe.py with scripts/dispram.sh, overlay/bringup/glm_dispram_kv.py and
# service/) is not on this branch; until it is added, any value other than 0 is refused before anything starts.
# Its calls below are the only places it touches the launch: extra container mounts/env, the lender check before
# compaction, the lease postcheck after stop, and the dispram-setup command. With K-stop the only admitted mode
# is require (KSTOP_DISPRAM_LAYOUT: the ordinary head alone cannot hold the context, so there is no fallback).
DISPRAM_MODES = {'0': '0', '1': 'auto', 'auto': 'auto', 'require': 'require'}
DISPRAM_HEAD_KV_BYTES = KSTOP_DISPRAM_LAYOUT['kv_bytes']


def dispram_guard_present():
    # Not in git: copied from dispram-a2 guard/build into the clone's overlay before a dispram boot.
    return (ROOT / DISPRAM_GUARD.replace('/overlay/', 'overlay/', 1)).is_file()


def dispram():
    """The dispram integration module when RECIPE_DISPRAM is on, else None."""
    mode = DISPRAM_MODES.get(ENV.get('RECIPE_DISPRAM', '0').strip() or '0')
    if mode is None:
        raise ValueError('RECIPE_DISPRAM must be 0, 1/auto or require')
    if mode == '0':
        return None
    try:
        import dispram_recipe
    except ImportError:
        raise ValueError('RECIPE_DISPRAM=%s needs the dispram integration (scripts/dispram_recipe.py), '
                         'which this release does not include; set RECIPE_DISPRAM=0' % mode) from None
    if not dispram_guard_present():
        raise ValueError('RECIPE_DISPRAM needs the copy guard at overlay/guard/libdispram_copy_guard.so '
                         '(dispram-a2 guard/build, sha256 in its SHA256SUMS)')
    return dispram_recipe


def kv_format():
    mode = ENV.get('GLM_KV_FORMAT', 'fp8')
    if mode not in ('fp8', 'fp4x'):
        raise ValueError('GLM_KV_FORMAT must be fp8 or fp4x')
    if mode == 'fp4x':
        if ENV.get('GLM_FULL_MLA') != 'triton' or ENV.get('VLLM_USE_V2_MODEL_RUNNER') != '1':
            raise ValueError('FP4x requires Triton MLA and V2 runner')
        if ENV.get('RECIPE_PROFILE', 'native-mtp-k2') != 'native-mtp-k2':
            raise ValueError('FP4x uses the one native-mtp-k2 profile')
        if ENV.get('GLM_MLA_SPLIT_K') != '32' or ENV.get('GLM_MLA_SPLIT_MAX_ROWS', '36') != '36':
            raise ValueError('FP4x release requires split32 and the 36-row dispatch cap')
        for key in ('GLM_FP4_PROBE_SIM', 'GLM_FP4_PROBE_CAPTURE', 'GLM_DSA_SWA_POOL',
                    'GLM_GLUE_DSA_IDX_CACHE', 'GLM_SPEC_SAMPLE', 'GLM_TRITON_MLA_PREFILL'):
            if ENV.get(key, '0') not in ('', '0'):
                raise ValueError('FP4x does not compose with ' + key)
    return mode


def launch_switches():
    mode = kv_format()
    values = {key: ENV.get(key, default).strip() for key, default in LAUNCH_DEFAULTS.items()}
    head = values['RECIPE_KV_HEAD_BYTES']
    if not head.isdigit() or int(head) < GiB or int(head) % (2 << 20):
        raise ValueError('RECIPE_KV_HEAD_BYTES must be >=1 GiB, aligned to 2 MiB')
    if int(head) != GiB and (mode != 'fp4x' or ENV.get('RECIPE_DISPRAM') != 'require'):
        raise ValueError('larger ordinary KV heads require FP4x and dispram require')
    if values['RECIPE_PAGE_CACHE_POLICY'] not in ('0', '1'):
        raise ValueError('RECIPE_PAGE_CACHE_POLICY must be 0 or 1')
    if mode == 'fp4x' and (ENV.get('GLM_MTP_KSTOP') != '1' or ENV.get('RECIPE_DISPRAM') != 'require'):
        raise ValueError('FP4x requires K-stop and dispram require')
    if mode == 'fp8' and values['RECIPE_MAX_MODEL_LEN'] == '100288':
        raise ValueError('100288 requires GLM_KV_FORMAT=fp4x')
    for key in ('RECIPE_C4_PASS2_GRAPH', 'RECIPE_KV_PIN_L1', 'RECIPE_NCCL_NO_LL128'):
        if values[key] not in ('0', '1'):
            raise ValueError(key + ' must be 0 or 1')
    packed_cap = 0
    if mode == 'fp4x':
        from fp4_kv_layout import layout as packed_layout
        packed_cap = int(packed_layout(mode, int(head))['max_model_len'])
    packed_length = (mode == 'fp4x' and values['RECIPE_MAX_MODEL_LEN'].isdigit()
                     and 64 <= int(values['RECIPE_MAX_MODEL_LEN']) <= packed_cap
                     and int(values['RECIPE_MAX_MODEL_LEN']) % 64 == 0)
    if values['RECIPE_MAX_MODEL_LEN'] not in MAX_MODEL_LENS and not packed_length:
        raise ValueError('RECIPE_MAX_MODEL_LEN must be one of ' + ', '.join(MAX_MODEL_LENS))
    values['GLM_MTP_KSTOP'] = ENV.get('GLM_MTP_KSTOP', '0').strip()
    if values['GLM_MTP_KSTOP'] not in ('0', '1'):
        raise ValueError('GLM_MTP_KSTOP must be 0 or 1')
    pad = ENV.get('GLM_PAD_HYGIENE', '0').strip()
    if pad not in ('0', '1'):
        raise ValueError('GLM_PAD_HYGIENE must be 0 or 1')
    if pad == '1' and values['GLM_MTP_KSTOP'] != '1':
        raise ValueError('GLM_PAD_HYGIENE=1 requires GLM_MTP_KSTOP=1')
    capture_layout = ENV.get('GLM_MTP_KSTOP_CAPTURE_LAYOUT', 'm12').strip()
    if values['GLM_MTP_KSTOP'] == '1' and capture_layout not in ('m12', 'reuse'):
        raise ValueError('GLM_MTP_KSTOP_CAPTURE_LAYOUT must be m12 or reuse')
    uniform = ENV.get('GLM_MTP_KSTOP_UNIFORM_BATCH', '0').strip()
    if uniform not in ('0', '1', 'k2'):
        raise ValueError('GLM_MTP_KSTOP_UNIFORM_BATCH must be 0, 1 or k2')
    if uniform != '0' and values['GLM_MTP_KSTOP'] != '1':
        raise ValueError('GLM_MTP_KSTOP_UNIFORM_BATCH='+uniform+' requires GLM_MTP_KSTOP=1')
    if values['GLM_MTP_KSTOP'] == '1':
        layout = kstop_layout(values)
        kv_blocks, lookahead = layout['blocks'], KSTOP_LOOKAHEAD
    else:
        if values['RECIPE_MAX_MODEL_LEN'] in ('44224', '66112'):
            raise ValueError('RECIPE_MAX_MODEL_LEN=%s is a K-stop layout; it requires GLM_MTP_KSTOP=1' % values['RECIPE_MAX_MODEL_LEN'])
        if dispram() is not None:
            raise ValueError('RECIPE_DISPRAM is admitted with the K-stop layout only (GLM_MTP_KSTOP=1)')
        kv_blocks, lookahead = KV_PIN_BYTES[values['RECIPE_KV_PIN_L1']] // KV_BLOCK_BYTES, 0
    # vLLM keeps one block of the pool as its null block; one max-length request must fit in the rest
    # (K-stop layouts also count the K3 lookahead, the campaign's pre-boot rule).
    if -(-(int(values['RECIPE_MAX_MODEL_LEN']) + lookahead) // KV_BLOCK_TOKENS) > kv_blocks - 1:
        raise ValueError('RECIPE_MAX_MODEL_LEN does not fit in the pinned KV pool')
    changed = [key for key, value in values.items() if key in LAUNCH_DEFAULTS and value != LAUNCH_DEFAULTS[key]]
    if values['GLM_MTP_KSTOP'] == '1':
        changed.append('GLM_MTP_KSTOP')
    if changed and ENV.get('RECIPE_PROFILE', 'native-mtp-k2') != 'native-mtp-k2':
        raise ValueError('launch switches apply to the native-mtp-k2 profile only: ' + ', '.join(changed))
    dispram()
    optional_env()
    return values


def kstop_layout(values):
    """The admitted K-stop layout for these switches; every other K-stop combination is refused."""
    if values['RECIPE_KV_PIN_L1'] != '0':
        raise ValueError('GLM_MTP_KSTOP=1 pins its own KV pool; RECIPE_KV_PIN_L1 must be 0')
    mode = DISPRAM_MODES.get(ENV.get('RECIPE_DISPRAM', '0').strip() or '0')
    if mode not in ('0', 'require'):
        raise ValueError('GLM_MTP_KSTOP=1 with dispram requires RECIPE_DISPRAM=require (no plain-pool fallback)')
    if mode == 'require':
        if kv_format() == 'fp4x':
            from fp4_kv_layout import layout
            packed = layout('fp4x', int(values['RECIPE_KV_HEAD_BYTES']))
            length = values['RECIPE_MAX_MODEL_LEN']
            if not length.isdigit() or not 64 <= int(length) <= int(packed['max_model_len']) or int(length)%64:
                raise ValueError('FP4x requires a 64-token context grid within packed capacity')
            return dict(packed, max_model_len=values['RECIPE_MAX_MODEL_LEN'])
        if values['RECIPE_MAX_MODEL_LEN'] != KSTOP_DISPRAM_LAYOUT['max_model_len']:
            raise ValueError('GLM_MTP_KSTOP=1 with RECIPE_DISPRAM=require launches RECIPE_MAX_MODEL_LEN=66112 only')
        return KSTOP_DISPRAM_LAYOUT
    layout = KSTOP_LAYOUTS.get(values['RECIPE_MAX_MODEL_LEN'])
    if layout is None:
        raise ValueError('GLM_MTP_KSTOP=1 requires RECIPE_MAX_MODEL_LEN 32768 (544-block pool) or 44224 '
                         '(693-block pool, RECIPE_NCCL_NO_LL128=1)')
    if layout['requires_no_ll128'] and values['RECIPE_NCCL_NO_LL128'] != '1':
        raise ValueError('the K-stop full-pool layout (693 blocks, 44224) requires RECIPE_NCCL_NO_LL128=1')
    return layout


def kstop_capture_sizes():
    return KSTOP_K2_CAPTURE_SIZES if ENV.get('GLM_MTP_KSTOP_UNIFORM_BATCH', '0').strip() == 'k2' else KSTOP_CAPTURE_SIZES


def launch_shape(args):
    """Apply the launch-shape switches to a copy of the saved native K2 vector."""
    switches = launch_switches()
    args = list(args)
    if ENV.get('RECIPE_PROFILE', 'native-mtp-k2') != 'native-mtp-k2':
        return args                     # other profiles keep their saved vector (switches are at defaults)
    if switches['RECIPE_C4_PASS2_GRAPH'] == '1':
        # Width 4 gives the later MTP pass (one token per request) a FULL graph at c4; the target and
        # first MTP pass keep their graphs. Same JSON spelling as the saved vector.
        i = args.index('--compilation-config') + 1
        config = json.loads(args[i])
        config['cudagraph_capture_sizes'] = sorted(set(config['cudagraph_capture_sizes']) | {4})
        args[i] = json.dumps(config)
    kv_bytes = KV_PIN_BYTES[switches['RECIPE_KV_PIN_L1']]
    if switches['GLM_MTP_KSTOP'] == '1':
        # K-stop drafts up to three tokens; target verify widths are 2-4 rows per request (M2-M4 at c1,
        # M16 at c4 when every request drafted three) and the later MTP passes stay M1-M4. [1, 4, 16]
        # keeps every c1 shape and c4 q4 on a graph; c2/c3 q4 pad to M16 and later passes to M4.
        i = args.index('--speculative-config') + 1
        spec = json.loads(args[i])
        spec['num_speculative_tokens'] = 3
        args[i] = json.dumps(spec)
        i = args.index('--compilation-config') + 1
        config = json.loads(args[i])
        if config.get('cudagraph_mode') != 'FULL_DECODE_ONLY' or args[args.index('--max-num-seqs') + 1] != '4':
            raise ValueError('K-stop layouts are admitted for FULL_DECODE_ONLY graphs and four slots only')
        config['cudagraph_capture_sizes'] = list(kstop_capture_sizes())
        config['max_cudagraph_capture_size'] = max(kstop_capture_sizes())
        args[i] = json.dumps(config)
        kv_bytes = kstop_layout(switches)['kv_bytes']
    args[args.index('--max-model-len') + 1] = switches['RECIPE_MAX_MODEL_LEN']
    args[args.index('--kv-cache-memory-bytes') + 1] = str(kv_bytes)
    return args


def kstop_note():
    """One line naming the K-stop layout (None with the switch off)."""
    switches = launch_switches()
    if switches['GLM_MTP_KSTOP'] != '1':
        return None
    layout = kstop_layout(switches)
    return ('K-stop layout: K3, capture sizes %s, %d KV blocks, max_model_len %s, NCCL LL128 %s, uniform batch K %s, capture layout %s; %s'
            % (kstop_capture_sizes(), layout['blocks'], switches['RECIPE_MAX_MODEL_LEN'],
               'off' if switches['RECIPE_NCCL_NO_LL128'] == '1' else 'on',
               {'1': 'on', 'k2': 'k2'}.get(ENV.get('GLM_MTP_KSTOP_UNIFORM_BATCH', '0').strip(), 'off'), ENV.get('GLM_MTP_KSTOP_CAPTURE_LAYOUT', 'm12'), layout['status']))


def validate():
    if len(HOSTS) != 4 or len(IPS) != 4 or len(IMAGES) != 4:
        raise ValueError('Exactly four hosts, fabric addresses and images required')
    if len(set(HOSTS)) != 4 or len(set(IPS)) != 4:
        raise ValueError('Hostnames and fabric addresses must be unique')
    for host in HOSTS:
        if not re.fullmatch(r'[A-Za-z0-9_.@-]+', host):
            raise ValueError('Invalid SSH host')
    for key in ('MODEL_DIR', 'DRAFT_DIR', 'NCCL_HOST_DIR', 'OVERLAY_REMOTE'):
        if not re.fullmatch(r'/[A-Za-z0-9_./-]+', ENV[key]):
            raise ValueError(key + ' requires an absolute path without shell syntax')
    if len({ENV[k] for k in ('MODEL_DIR', 'DRAFT_DIR', 'NCCL_HOST_DIR', 'OVERLAY_REMOTE')}) != 4:
        raise ValueError('Model, draft, NCCL and runtime paths must be distinct')
    nccl_hashes()
    if persistent_cache_enabled():
        path = ENV.get('PERSISTENT_CACHE_DIR', '')
        if not re.fullmatch(r'/[A-Za-z0-9_./-]+', path) or '..' in path.split('/'):
            raise ValueError('PERSISTENT_CACHE_DIR requires an absolute path without shell syntax')
        for key in ('MODEL_DIR', 'DRAFT_DIR', 'NCCL_HOST_DIR', 'OVERLAY_REMOTE'):
            a, b = path.rstrip('/') + '/', ENV[key].rstrip('/') + '/'
            if a.startswith(b) or b.startswith(a):
                raise ValueError('PERSISTENT_CACHE_DIR must lie outside ' + key)


def nccl_hashes():
    # One hash for identical binaries, or four comma-separated hashes in HOSTS order.
    values = ENV['NCCL_SHA256'].split(',')
    values = values * 4 if len(values) == 1 else values
    if len(values) != 4 or any(not re.fullmatch(r'[0-9a-f]{64}', v) for v in values):
        raise ValueError('NCCL_SHA256 must be one SHA256 or four comma-separated SHA256 values in host order')
    return values


def remote(rank, command, timeout=45):
    return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                                    HOSTS[rank], command], text=True, timeout=timeout).strip()


# Opt-in keys that reach the containers only when set to a value other than 0, so the default vector carries
# none of them: the MLA plan skip (1 = static, ab = in-boot A/B through worker RPCs) and vLLM's dev API, which
# the A/B mode needs on the private loopback port.
OPTIONAL_KEYS = ('GLM_SKIP_MLA_PLAN', 'GLM_SKIP_MLA_PLAN_AB_INIT', 'VLLM_SERVER_DEV_MODE', 'GLM_PARAM_HASH')


def optional_env():
    env = {k: ENV[k].strip() for k in OPTIONAL_KEYS if ENV.get(k, '').strip() not in ('', '0')}
    sys.path.insert(0, str(ROOT / 'overlay/bringup'))
    from glm_recent_kv import options as recent_options
    recent_window, _, recent_ab = recent_options(ENV)
    if recent_window:
        env.update({k: ENV[k] for k in ('GLM_FP4_RECENT_WINDOW','GLM_FP4_RECENT_AB','GLM_FP4_RECENT_INIT') if k in ENV})
        if recent_ab:
            env['VLLM_SERVER_DEV_MODE']='1'
    mode = env.get('GLM_SKIP_MLA_PLAN', '0')
    if mode not in ('0', '1', 'ab'):
        raise ValueError('GLM_SKIP_MLA_PLAN must be 0, 1 or ab')
    if env.get('VLLM_SERVER_DEV_MODE', '1') != '1' or env.get('GLM_SKIP_MLA_PLAN_AB_INIT', '1') != '1':
        raise ValueError('VLLM_SERVER_DEV_MODE and GLM_SKIP_MLA_PLAN_AB_INIT must be 0 or 1')
    if mode != '0' and ENV.get('GLM_FULL_MLA') != 'triton':
        raise ValueError('GLM_SKIP_MLA_PLAN needs GLM_FULL_MLA=triton')
    hashing = env.get('GLM_PARAM_HASH', '0')
    if hashing not in ('0', '1'):
        raise ValueError('GLM_PARAM_HASH must be 0 or 1')
    if (mode == 'ab' or hashing == '1' or recent_ab) != ('VLLM_SERVER_DEV_MODE' in env):
        raise ValueError('VLLM_SERVER_DEV_MODE=1 needs GLM_SKIP_MLA_PLAN=ab or GLM_PARAM_HASH=1 or recent KV AB')
    if 'GLM_SKIP_MLA_PLAN_AB_INIT' in env and mode != 'ab':
        raise ValueError('GLM_SKIP_MLA_PLAN_AB_INIT applies to GLM_SKIP_MLA_PLAN=ab only')
    # Include zero explicitly so FP4x's default-on scheduler can be disabled.
    adaptive = {k: ENV[k] for k in ('GLM_PREFILL_CHUNK_ADAPTIVE', 'GLM_PREFILL_CHUNK_THRESHOLD') if k in ENV}
    if adaptive.get('GLM_PREFILL_CHUNK_ADAPTIVE', '0') not in ('0', '1'):
        raise ValueError('GLM_PREFILL_CHUNK_ADAPTIVE must be 0 or 1')
    if int(adaptive.get('GLM_PREFILL_CHUNK_THRESHOLD', '16384')) <= 0:
        raise ValueError('GLM_PREFILL_CHUNK_THRESHOLD must be positive')
    env.update(adaptive)
    fair = {k: ENV[k] for k in ('GLM_DECODE_FAIR', 'GLM_DECODE_FAIR_CHUNK', 'GLM_DECODE_FAIR_DECODE_STEPS', 'GLM_DECODE_FAIR_CONTROL') if k in ENV}
    sys.path.insert(0, str(ROOT / 'overlay' / 'bringup'))
    import glm_decode_fair
    glm_decode_fair.settings(dict(ENV, **env, **fair))
    if fair.get('GLM_DECODE_FAIR', '0') != '0':
        env.update(fair)
    weights = ENV.get('GLM_ATTN_WEIGHTS', 'int8')
    if weights not in ('int8', 'nvfp4'):
        raise ValueError('GLM_ATTN_WEIGHTS must be int8 or nvfp4')
    if weights == 'nvfp4':
        sys.path.insert(0, str(ROOT / 'overlay' / 'overlay'))
        import glm_nvfp4_groups as nvfp4_groups
        selected_groups = nvfp4_groups.groups(ENV)
        sidecar = ENV.get('GLM_ATTN_NVFP4_DIR', '')
        if not sidecar.startswith('/') or sidecar == ENV['MODEL_DIR']:
            raise ValueError('NVFP4 needs an absolute, separate GLM_ATTN_NVFP4_DIR')
        if ENV.get('GLM_NVFP4_WSIM', '') not in ('', '0'):
            raise ValueError('real NVFP4 cannot be combined with weight simulator')
        env.update(GLM_ATTN_WEIGHTS='nvfp4', GLM_ATTN_NVFP4_DIR='/attn-nvfp4')
        # Omit the default selector entirely: serving DRY vectors remain byte-identical.
        if selected_groups != ('attn',):
            more = ENV.get('GLM_NVFP4_MORE_DIR', '')
            if not more.startswith('/') or more == ENV['MODEL_DIR'] or more == sidecar:
                raise ValueError('extra groups require a separate absolute GLM_NVFP4_MORE_DIR')
            if 'mtp' in selected_groups and (ENV.get('GLM_MTP_ONLY_LOAD') != '1' or ENV.get('GLM_TARGET_SKIP_MTP') != '1'):
                raise ValueError('MTP NVFP4 requires native selected target/draft loading')
            env.update(GLM_NVFP4_GROUPS=','.join(selected_groups), GLM_NVFP4_MORE_DIR='/more-nvfp4')
    elif ENV.get('GLM_NVFP4_GROUPS', 'attn') != 'attn':
        raise ValueError('extra real groups require GLM_ATTN_WEIGHTS=nvfp4')
    # Explicit loader opt-in reaches every rank without changing the profile default.
    loader = ENV.get('GLM_LOADER', '').strip()
    if loader not in ('', 'fast', 'coalesced'):
        raise ValueError('GLM_LOADER must be fast or coalesced')
    if loader:
        env['GLM_LOADER'] = loader
    loader_keys = ('GLM_COALESCED_BATCH_MB', 'GLM_COALESCED_OWNED_MB',
                   'GLM_COALESCED_THREADS', 'GLM_COALESCED_DIRECT',
                   'GLM_COALESCED_EMERGENCY_MB', 'GLM_COALESCED_MARGIN_MB')
    settings = {k: ENV[k] for k in loader_keys if k in ENV}
    if settings and loader != 'coalesced':
        raise ValueError('GLM_COALESCED settings need GLM_LOADER=coalesced')
    if loader == 'coalesced':
        sys.path.insert(0, str(ROOT / 'overlay/overlay'))
        from glm_coalesced_load import options
        options(ENV)
        env.update(settings)
    return env


def rank_env(rank):
    env = {k: ENV[k] for k in PROFILE_KEYS}
    env.update(optional_env())
    if kv_format() == 'fp4x':
        env['GLM_KV_FORMAT'] = 'fp4x'
        switches = launch_switches()
        if int(switches['RECIPE_KV_HEAD_BYTES']) != GiB:
            env['GLM_FP4_POOL_BLOCKS'] = str(kstop_layout(switches)['blocks'])
            env['GLM_FP4_MAX_MODEL_LEN'] = switches['RECIPE_MAX_MODEL_LEN']
    if 'NCCL_PROTO' in env:
        raise ValueError('set NCCL_PROTO through RECIPE_NCCL_NO_LL128 only')
    if launch_switches()['RECIPE_NCCL_NO_LL128'] == '1':
        # No LL128 protocol: NCCL then allocates no LL128 buffers (about 4.7 MiB per connection).
        env['NCCL_PROTO'] = '^LL128'
    env.update(VLLM_HOST_IP=IPS[rank], NCCL_SOCKET_IFNAME='=' + ENV['FABRIC_IFACE'],
               GLOO_SOCKET_IFNAME=ENV['FABRIC_IFACE'], NCCL_IB_HCA='=' + ENV['IB_HCA'],
               B12X_ROCE_HCA=ENV['IB_HCA'])
    return env


PREFIX_FLAGS = ('--enable-prefix-caching', '--no-enable-prefix-caching')


def prefix_flag(args):
    """The one prefix-caching flag of a saved vector (on in the native profile, off in DSpark K3)."""
    found = [flag for flag in PREFIX_FLAGS for item in args if item == flag]
    if len(found) != 1:
        raise ValueError('the saved vector must carry exactly one prefix-caching flag')
    return found[0]


def rank_args(rank):
    args = launch_shape(json.loads(SERVE_ARGS.read_text()))
    args[args.index('--node-rank') + 1] = str(rank)
    args[args.index('--master-addr') + 1] = IPS[0]
    if rank:
        # Preserve the saved worker vector's order: --headless goes right before the prefix-caching flag.
        args.insert(args.index(prefix_flag(args)), '--headless')
    return args



def prefill_control(chunk, sequence):
    path = ENV.get('GLM_W2_PREFILL_CONTROL')
    if not path:
        return
    if path != '/cache/d2w2-prefill-control.json' or chunk not in (512, 2048, 4096):
        raise ValueError('Unsupported prefill control path/cap')
    control = dict(schema=1, chunk=chunk, sequence=sequence)
    code = 'import json,os,pathlib; p=pathlib.Path(PATH); q=p.with_name(p.name+".new"); q.write_text(PAYLOAD); os.replace(q,p); print(p.read_text())'
    code = code.replace('PATH', repr(ENV['OVERLAY_REMOTE'] + '/cache/d2w2-prefill-control.json')).replace('PAYLOAD', repr(json.dumps(control)))
    for rank in range(4):
        if json.loads(remote(rank, 'python3 -c ' + shlex.quote(code))) != control:
            raise RuntimeError('Prefill readback disagreement')


def select_serving_prefill():
    if not ENV.get('GLM_W2_PREFILL_CONTROL'):
        return
    cap = int(ENV.get('GLM_W2_PREFILL_CHUNK', '512'))
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        metrics = remote(0, 'curl -fsS -m5 http://127.0.0.1:8095/metrics')
        counts = []
        for key in ('vllm:num_requests_running', 'vllm:num_requests_waiting'):
            vals = [float(line.split()[-1]) for line in metrics.splitlines() if line.startswith(key+'{') or line.startswith(key+' ')]
            if not vals:
                raise RuntimeError('Missing drain metric')
            counts.append(sum(vals))
        if counts == [0, 0]:
            prefill_control(cap, time.time_ns())
            print('Serving prefill cap %d; constructor capacity4096' % cap, flush=True)
            return
        time.sleep(.25)
    raise RuntimeError('Prefill cap selection did not drain in30s')


def uses_external_draft():
    args = json.loads(SERVE_ARGS.read_text())
    return json.loads(args[args.index('--speculative-config')+1]).get('model') == '/draft'


def docker_command(rank, ctn):
    cmd = ['docker', 'run', '-d', '--restart', 'no', '--name', f'{ctn}-r{rank}',
           '--gpus', 'all', '--network', 'host', '--ipc', 'host', '--device', '/dev/infiniband',
           '--cap-add', 'IPC_LOCK', '--ulimit', 'memlock=-1', '--ulimit', 'stack=67108864',
           '--ulimit', 'nofile=1048576:1048576', '--memory', '112g', '--memory-swap', '112g',
           '--entrypoint', 'vllm']
    mounts = [(ENV['MODEL_DIR'], '/model', True),
                              (ENV['OVERLAY_REMOTE'], '/overlay', True),
                              (ENV['OVERLAY_REMOTE'] + '/cache', '/cache', False),
                              (ENV['NCCL_HOST_DIR'], '/opt/nccl', True)]
    if ENV.get('GLM_ATTN_WEIGHTS', 'int8') == 'nvfp4':
        mounts.append((ENV['GLM_ATTN_NVFP4_DIR'], '/attn-nvfp4', True))
        if ENV.get('GLM_NVFP4_GROUPS', 'attn') != 'attn':
            mounts.append((ENV['GLM_NVFP4_MORE_DIR'], '/more-nvfp4', True))
    if uses_external_draft():
        mounts.append((ENV['DRAFT_DIR'], '/draft', True))
    for source, dest, ro in mounts:
        cmd += ['-v', source + ':' + dest + (':ro' if ro else '')]
    env = rank_env(rank)
    hook = dispram()
    if hook is not None:
        extra_mounts, extra_env = hook.container_args(ENV)
        cmd += extra_mounts
        env.update(extra_env)
        # The copy guard (as qualified in the dispram TP4 window) belongs in the engine containers only.
        env['LD_PRELOAD'] = env['LD_PRELOAD'] + ':' + DISPRAM_GUARD
        env['DISPRAM_GUARD_REQUIRED'] = '1'
    for key, value in sorted(env.items()):
        cmd += ['-e', key + '=' + value]
    return cmd + [IMAGES[rank]] + rank_args(rank)


# Cold FlashInfer JIT modules that the serving process would otherwise compile with nvcc right after
# graph capture (sampling, about 2-2.7 GiB of host memory for about 37 s) and during the first FULL graph
# (batch MLA). They are built into the fresh per-deployment cache from the same image and environment
# before the model containers start; the serving build then finds them up to date and runs no nvcc.
JIT_PREP_TIMEOUT_S = 600
JIT_PREP_MLA = ('batch_mla_attention_dtype_q_bf16_dtype_kv_e4m3_dtype_o_bf16_dtype_idx_i32'
                '_head_dim_ckv_512_head_dim_kpe_64_profiler_False')
# One line, so DRY output stays one command per line.
JIT_PREP_CODE = '; '.join([
    'import torch',
    'from flashinfer.jit.attention import gen_batch_mla_module',
    'from flashinfer.jit.sampling import gen_sampling_module',
    "specs = [gen_sampling_module(), gen_batch_mla_module('fa2', torch.bfloat16, torch.float8_e4m3fn, "
    "torch.bfloat16, torch.int32, 512, 64, False)]",
    f"assert [s.name for s in specs] == ['sampling', {JIT_PREP_MLA!r}], [s.name for s in specs]",
    '[s.build(need_lock=True) for s in specs]',
    'assert all(s.is_compiled for s in specs), [s.name for s in specs if not s.is_compiled]',
    "print(chr(10).join('JIT PREP %s %s' % (s.name, s.get_library_path()) for s in specs), flush=True)"])


def jit_prep_command(rank, ctn):
    # Same image, environment and /cache mount as the rank's model container. PYTHONPATH is
    # emptied so the overlay hooks do not run; no GPU, no network. The in-container timeout
    # ends the container (and its compilers) itself; --rm leaves nothing behind.
    env = rank_env(rank)
    env['PYTHONPATH'] = ''
    cmd = ['docker', 'run', '--rm', '--name', f'{ctn}-jitprep-r{rank}', '--network', 'none',
           '--memory', '16g', '--memory-swap', '16g',
           '-v', ENV['OVERLAY_REMOTE'] + '/cache:/cache', '-v', ENV['NCCL_HOST_DIR'] + ':/opt/nccl:ro']
    for key, value in sorted(env.items()):
        cmd += ['-e', key + '=' + value]
    return cmd + ['--entrypoint', 'timeout', IMAGES[rank], '-k', '10', str(JIT_PREP_TIMEOUT_S),
                  'python3', '-c', JIT_PREP_CODE]


def jit_prep(ctn):
    def prep_rank(rank):
        out = remote(rank, shlex.join(jit_prep_command(rank, ctn)), timeout=JIT_PREP_TIMEOUT_S + 60)
        built = [line for line in out.splitlines() if line.startswith('JIT PREP ')]
        if len(built) != 2:
            raise RuntimeError('JIT prep incomplete: ' + out[-400:])
        return built
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        # Fail closed: every rank must finish before compaction and launch.
        futures = [pool.submit(prep_rank, rank) for rank in range(4)]
        errors = []
        for rank, future in enumerate(futures):
            try:
                for line in future.result():
                    print(HOSTS[rank], line, flush=True)
            except Exception as exc:
                errors.append((rank, str(exc)[-400:]))
    if errors:
        raise RuntimeError('JIT prep failed: ' + repr(errors))
    print('JIT PREP PASS', flush=True)


# Persistent compile cache (RECIPE_PERSISTENT_CACHE=1, opt-in). A fresh deployment starts from
# an empty $OVERLAY_REMOTE/cache. With the flag, each node keeps one generation per key under
# PERSISTENT_CACHE_DIR, outside every release tree, so a fresh clone reuses the FlashInfer JIT,
# Triton, b12x and CUDA caches. Key = image ID + FlashInfer version + CUDA arch lists + overlay
# tree hash + cache layout. A different key is a different directory: a miss builds fresh and
# nothing is reused across keys. The cache files are root-owned, so they are copied only inside a
# container of the keyed image (as root, cp -a, overlay/tools/glm_persistent_cache.py), never by
# the SSH user on the host. A generation is published by one atomic rename after admission and is
# never overwritten.
PERSIST_CACHE_ENV = ('CUDA_CACHE_PATH', 'FLASHINFER_WORKSPACE_BASE', 'HF_HOME', 'TORCHINDUCTOR_CACHE_DIR',
                     'TRITON_CACHE_DIR', 'VLLM_CACHE_ROOT', 'XDG_CACHE_HOME')
PERSIST_EXCLUDE = ('d2w2-prefill-control.json', 'd2w2-prefill-control.json.new')   # deployment state
PERSIST_MAX_BYTES = 16 * GiB
PERSIST_TIMEOUT_S = 600     # in-container limit of one seed or save helper
PERSIST_SAVE_DEADLINE_S = PERSIST_TIMEOUT_S + 30   # the watchdog stops save helpers still running after this


def persistent_cache_enabled():
    value = ENV.get('RECIPE_PERSISTENT_CACHE', '0')
    if value not in ('0', '1'):
        raise ValueError('RECIPE_PERSISTENT_CACHE must be 0 or 1')
    return value == '1'


def overlay_hash():
    # The tree rsync copies to $OVERLAY_REMOTE (everything but __pycache__).
    digest = hashlib.sha256()
    for path in sorted(p for p in (ROOT / 'overlay').rglob('*') if p.is_file() and '__pycache__' not in p.parts):
        digest.update(path.relative_to(ROOT / 'overlay').as_posix().encode() + b'\0' +
                      hashlib.sha256(path.read_bytes()).hexdigest().encode() + b'\n')
    return digest.hexdigest()


def persist_parts(image_id):
    env = rank_env(0)
    return dict(schema=1, image=image_id, overlay=overlay_hash(),
                kv_format=kv_format(), kv_abi=('fp4x_v1_e2m1_s16_e4m3_pow2_r64' if kv_format() == 'fp4x' else 'fp8_e4m3'),
                arch={k: env.get(k, '') for k in ('FLASHINFER_CUDA_ARCH_LIST', 'TORCH_CUDA_ARCH_LIST')},
                cache_env={k: env.get(k, '') for k in PERSIST_CACHE_ENV})


def persist_command(rank, ctn, image_id, phase, key=''):
    # The model container's /cache mount (read-only when saving), the rsynced overlay for the
    # helper, and the keyed image by ID; no GPU, no network, no overlay hooks (PYTHONPATH empty).
    if phase not in ('seed', 'save'):
        raise ValueError('phase must be seed or save')
    cache = ENV['OVERLAY_REMOTE'] + '/cache:/cache' + (':ro' if phase == 'save' else '')
    name = persist_save_name(ctn, rank) if phase == 'save' else f'{ctn}-pcache-seed-r{rank}'
    cmd = ['docker', 'run', '--rm', '--name', name, '--network', 'none',
           '--memory', '4g', '--memory-swap', '4g', '-v', ENV['OVERLAY_REMOTE'] + ':/overlay:ro', '-v', cache,
           '-v', ENV['PERSISTENT_CACHE_DIR'] + ':/persist',
           '-e', 'PYTHONPATH=', '-e', 'PCACHE_PARTS=' + json.dumps(persist_parts(image_id), sort_keys=True)]
    if phase == 'save':
        cmd += ['-e', 'PCACHE_KEY=' + key, '-e', 'PCACHE_MAX=%d' % PERSIST_MAX_BYTES,
                '-e', 'PCACHE_EXCLUDE=' + json.dumps(PERSIST_EXCLUDE)]
    return cmd + ['--entrypoint', 'timeout', image_id, '-k', '10', str(PERSIST_TIMEOUT_S),
                  'python3', '/overlay/tools/glm_persistent_cache.py', phase]


def persist_result(out, phase):
    words = {'seed': ('HIT', 'MISS', 'MISMATCH'), 'save': ('SAVED', 'KEEP', 'SKIP')}[phase]
    lines = [line.split() for line in out.splitlines() if line.startswith('PERSISTENT CACHE ')]
    if len(lines) != 1 or lines[0][2] not in words or not re.fullmatch(r'[0-9a-f]{32}', lines[0][3]):
        raise RuntimeError('persistent cache %s: unexpected output %r' % (phase, out[-400:]))
    return lines[0][2], lines[0][3]


def persistent_cache_seed(ctn):
    # Before JIT prep: on a hit JIT prep and the boot's compiles find their outputs up to date.
    def seed_rank(rank):
        image_id = remote(rank, 'docker image inspect -f {{.Id}} ' + shlex.quote(IMAGES[rank]))
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', image_id):
            raise RuntimeError('image ID unresolved on ' + HOSTS[rank])
        remote(rank, 'mkdir -p ' + shlex.quote(ENV['PERSISTENT_CACHE_DIR']))
        try:
            out = remote(rank, shlex.join(persist_command(rank, ctn, image_id, 'seed')), timeout=PERSIST_TIMEOUT_S + 60)
        except subprocess.CalledProcessError as exc:
            # Exit 3 = CORRUPT: the generation for this key failed verification; nothing was seeded.
            raise RuntimeError('seed helper exit %s: %s' % (exc.returncode, (exc.output or '')[-400:]))
        status, key = persist_result(out, 'seed')
        return dict(image=image_id, key=key, seed=status)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(seed_rank, rank) for rank in range(4)]
        result, errors = {}, []
        for rank, future in enumerate(futures):
            try:
                result[str(rank)] = future.result()
                print(HOSTS[rank], 'PERSISTENT CACHE', result[str(rank)]['seed'], result[str(rank)]['key'], flush=True)
            except Exception as exc:
                errors.append((rank, str(exc)[-400:]))
    if errors:
        raise RuntimeError('Persistent cache seed failed: ' + repr(errors))
    return result


def persist_save_name(ctn, rank):
    return f'{ctn}-pcache-save-r{rank}'


class PersistentCacheSave:
    """Save helpers started after admission; they run beside the watchdog, never inside it.

    One daemon thread per rank runs the save container over SSH. ``poll`` (called from the
    watchdog loop, which keeps sampling memory, swap, errors and progress meanwhile) prints the
    results and stops helpers still running after ``PERSIST_SAVE_DEADLINE_S``. ``cancel`` stops
    them on any abort; a helper killed mid-copy leaves no published generation. A save failure
    never stops serving.
    """

    def __init__(self, deployment, now):
        self.deployment, self.started, self.cancelled = deployment, now, False
        self.results, self.reported = {}, set()
        self.lock = threading.Lock()
        self.threads = []
        for rank in range(4):
            t = threading.Thread(target=self._run, args=(rank,), name=f'pcache-save-r{rank}', daemon=True)
            t.start()
            self.threads.append(t)

    def _run(self, rank):
        try:
            result = self._save(rank)
        except Exception as exc:
            result = 'SAVE FAILED (serving continues): ' + str(exc)[-300:]
        with self.lock:
            self.results[rank] = result

    def _save(self, rank):
        info = self.deployment['persistent_cache'][str(rank)]
        ctn = self.deployment['ctn']
        running = remote(rank, 'docker inspect -f {{.Image}} ' + shlex.quote(ctn + f'-r{rank}'))
        if running != info['image']:
            return 'SKIP image changed since seed'
        if self.cancelled:
            return 'SKIP cancelled'
        out = remote(rank, shlex.join(persist_command(rank, ctn, info['image'], 'save', info['key'])),
                     timeout=PERSIST_SAVE_DEADLINE_S)
        return ' '.join(persist_result(out, 'save'))

    def done(self):
        return all(not t.is_alive() for t in self.threads)

    def poll(self, now):
        with self.lock:
            fresh = [(r, v) for r, v in sorted(self.results.items()) if r not in self.reported]
            self.reported.update(r for r, _ in fresh)
        for rank, result in fresh:
            print(HOSTS[rank], 'PERSISTENT CACHE', result, flush=True)
        if not self.done() and not self.cancelled and now - self.started > PERSIST_SAVE_DEADLINE_S:
            print('PERSISTENT CACHE SAVE deadline %d s passed; stopping the helpers (serving continues)'
                  % PERSIST_SAVE_DEADLINE_S, flush=True)
            self.cancel()

    def cancel(self):
        # docker stop sends TERM: the helper removes its temporary and publishes nothing.
        if self.done() or self.cancelled:
            self.cancelled = True
            return
        self.cancelled = True
        ctn = self.deployment['ctn']

        def stop_rank(rank):
            remote(rank, 'docker stop -t 10 ' + shlex.quote(persist_save_name(ctn, rank)) + ' >/dev/null 2>&1 || true',
                   timeout=40)
        # A helper may start just after a stop (the container did not exist yet): repeat the stop
        # for ranks whose thread is still alive, a bounded number of times.
        for _ in range(3):
            alive = [rank for rank, t in enumerate(self.threads) if t.is_alive()]
            if not alive:
                return
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = {rank: pool.submit(stop_rank, rank) for rank in alive}
                for rank, future in futures.items():
                    try:
                        future.result()
                    except Exception as exc:
                        print(HOSTS[rank], 'PERSISTENT CACHE helper stop failed:', str(exc)[-200:], flush=True)
            for rank in alive:
                self.threads[rank].join(15)
        alive = [HOSTS[rank] for rank, t in enumerate(self.threads) if t.is_alive()]
        if alive:
            print('PERSISTENT CACHE helpers still running on', alive, '(in-container limit %d s)' % PERSIST_TIMEOUT_S, flush=True)


ARG_MAX_STRLEN = 131072   # Linux MAX_ARG_STRLEN: one argv string, here the remote shell's -c program


def verify_command(path, manifest, mode='cached'):
    # Full CPU checksum verification streams each file, never initializes CUDA. The manifest
    # is embedded once and bound to a name, so the single remote argument stays small.
    if mode not in ('cached', 'full'):
        raise ValueError('VERIFY_WEIGHTS must be cached or full')
    verify = (ROOT / 'scripts/weights.py').read_text().split("if __name__ == '__main__':")[0]
    receipt = 'None' if mode == 'full' else 'receipt_path(ROOT_DIR, MANIFEST)'
    call = f'\nMANIFEST = {manifest!r}\nROOT_DIR = Path({path!r})\nverify(ROOT_DIR, MANIFEST, {receipt})\n'
    command = 'nice -n 10 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -c ' + shlex.quote(verify + call)
    if len(command.encode()) >= ARG_MAX_STRLEN - 4096:
        raise RuntimeError('weight verification program too large for one remote argument')
    return command


def preflight(ctn):
    # Two independent single-thread checksum streams; preserve every rank receipt.
    def check_rank(rank):
        code = '''import json, pathlib, subprocess
p=pathlib.Path
image=json.loads(subprocess.check_output(['docker','image','inspect',IMAGE]))[0]
assert image['Architecture']=='arm64', 'linux/arm64 image required'
assert not (HEAD and p.home().joinpath('fleet_busy').exists()), 'fleet is owned'
assert not subprocess.check_output(['docker','ps','-a','--filter','name=^'+NAME+'$','--format','{{.Names}}']).strip(), 'name already used'
running=json.loads(subprocess.check_output(['docker','ps','-q']).decode() and subprocess.check_output(['docker','inspect']+subprocess.check_output(['docker','ps','-q']).decode().split()) or '[]')
for c in running:
 req=c['HostConfig'].get('DeviceRequests') or []
 assert not any('gpu' in caps for d in req for caps in d.get('Capabilities',[])), 'GPU container is running: '+c['Name']
 assert not any(m['Source']==OVERLAY for m in c.get('Mounts',[])), 'runtime path in use'
assert not p(OVERLAY).exists(), 'runtime path already exists; use a fresh path'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']).strip(), 'GPU process is running'
assert p('/dev/infiniband').is_dir(), 'RDMA device missing'
for d in ([MODEL,DRAFT] if EXTERNAL_DRAFT else [MODEL]):
 assert p(d).joinpath('config.json').is_file(), 'weights missing'
 assert not list(p(d).rglob('*.part')), 'partial downloads present'
assert len(list(p(MODEL).glob('*.safetensors')))==282, 'expected 282 target shards'
mem=dict((l.split(':')[0],int(l.split()[1])) for l in p('/proc/meminfo').read_text().splitlines())
assert mem['MemAvailable']>=110*1048576, 'preboot headroom below 110 GiB'
assert p('/usr/local/sbin/spark-compact-mem.sh').is_file(), 'compaction helper missing'
assert p(NCCL).joinpath('libnccl.so.2.30.7').is_file(), 'NCCL missing'
print(json.dumps(dict(architecture=image['Architecture'],image=image['Id'],MemAvailable_kB=mem['MemAvailable'])))
'''
        setup = '\n'.join(k + '=' + repr(v) for k, v in dict(IMAGE=IMAGES[rank], NAME=f'{ctn}-r{rank}',
                      OVERLAY=ENV['OVERLAY_REMOTE'], MODEL=ENV['MODEL_DIR'], DRAFT=ENV['DRAFT_DIR'],
                      NCCL=ENV['NCCL_HOST_DIR'], EXTERNAL_DRAFT=uses_external_draft(), HEAD=rank == 0).items())
        print(HOSTS[rank], remote(rank, 'python3 -c ' + shlex.quote(setup + '\n' + code)))
        checkpoints = [('target', ENV['MODEL_DIR'])]
        if uses_external_draft():
            checkpoints.append(('drafter', ENV['DRAFT_DIR']))
        for kind, path in checkpoints:
            manifest = json.loads((ROOT / f'manifests/{kind}.json').read_text())
            result = remote(rank, verify_command(path, manifest, ENV.get('VERIFY_WEIGHTS', 'cached')), timeout=1200)
            print(HOSTS[rank], result, flush=True)
        sha = remote(rank, 'sha256sum ' + shlex.quote(ENV['NCCL_HOST_DIR'] + '/libnccl.so.2.30.7')).split()[0]
        if sha != nccl_hashes()[rank]:
            raise RuntimeError('NCCL binary hash mismatch on ' + HOSTS[rank])
        addresses = remote(rank, 'ip -4 -o addr show dev ' + shlex.quote(ENV['FABRIC_IFACE']))
        if IPS[rank] + '/' not in addresses:
            raise RuntimeError('Fabric IP/interface mismatch on ' + HOSTS[rank])
        remote(rank, 'sudo -n -l /usr/local/sbin/spark-compact-mem.sh >/dev/null')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        # Await every task, including failures, before admitting any container.
        results = [pool.submit(check_rank, rank) for rank in range(4)]
        errors = []
        for rank, future in enumerate(results):
            try:
                future.result()
            except Exception as exc:
                errors.append((rank, str(exc)))
        if errors:
            raise RuntimeError('Preflight failed: ' + repr(errors))
    print('PREFLIGHT PASS', flush=True)


def lock(token):
    remote(0, 'set -C; printf %s ' + shlex.quote(token) + ' > "$HOME/fleet_busy"')


def unlock(token):
    # Only unlink the exact lock owned by this deployment.
    remote(0, 'python3 -c ' + shlex.quote('from pathlib import Path; p=Path.home()/"fleet_busy"; '
                    f'\np.unlink() if p.exists() and p.read_text()=={token!r} else None'))


def page_cache_state(action):
    if ENV.get('RECIPE_PAGE_CACHE_POLICY', '0') == '1':
        for rank in range(4):
            remote(rank, 'python3 /usr/local/sbin/glm-page-cache-policy.py ' + action, timeout=45)


def stop(deployment):
    policy_error = None
    try:
        page_cache_state('end')
    except Exception as exc:
        policy_error = exc  # a broken optional timer must never prevent docker stop
    def stop_rank(rank):
        remote(rank, 'docker stop -t 60 ' + shlex.quote(deployment['ctn'] + f'-r{rank}'), timeout=80)
        status = remote(rank, 'docker inspect -f ' + shlex.quote('{{.State.Running}}') + ' ' + shlex.quote(deployment['ctn'] + f'-r{rank}'))
        if status != 'false':
            raise RuntimeError('container still running')
    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(stop_rank, rank) for rank in range(4)]
        for rank, future in enumerate(futures):
            try:
                future.result()
            except Exception as exc:
                failed.append((rank, str(exc)))
    if failed:
        raise RuntimeError('Stop incomplete; lock retained: ' + repr(failed))
    hook = dispram()
    if hook is not None:
        hook.postcheck(HOSTS, ENV)      # raises (lock retained) if a carveout lease outlived its engine
    if policy_error is not None:
        raise RuntimeError('Containers stopped; cache policy end failed; lock retained: ' + str(policy_error))
    unlock(deployment['token'])
    print('Four containers stopped and preserved; owned lock released', flush=True)


SAMPLE_CODE = '''import json,pathlib,subprocess,re
p=pathlib.Path
m={l.split(':')[0]:int(l.split()[1]) for l in p('/proc/meminfo').read_text().splitlines()}
v={l.split()[0]:int(l.split()[1]) for l in p('/proc/vmstat').read_text().splitlines()}
d=json.loads(subprocess.check_output(['docker','inspect',NAME]))[0]
logs=subprocess.check_output(['docker','logs','--tail','150',NAME],stderr=subprocess.STDOUT).decode(errors='replace')
# On a failure only, recover the beginning of the traceback from complete logs.
if re.search(r'Traceback|EngineDead|CUDA error|illegal memory access|device-side assert|Worker.*(?:ERROR|failed|Failed)',logs):
 logs=subprocess.check_output(['docker','logs',NAME],stderr=subprocess.STDOUT).decode(errors='replace')
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits']).decode()
print(json.dumps(dict(mem=m,vm=v,state=d['State'],restarts=d['RestartCount'],logs=logs,gpu=gpu)))
'''


def health():
    return remote(0, "curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8095/health || true") == '200'


def fatal_line(line):
    # Log-level ERROR is matched case-sensitively: status JSON such as RoCE's
    # "error_hca": 0 counters in the readiness line is not an error.
    if 'import_utils' in line and 'WARNING' in line:
        return False
    return bool(re.search(r'CUDA error|illegal memory access|device-side assert|Traceback|EngineDead|\bnan\b|\binf\b', line, re.I)
                or re.search(r'Worker.*(\bERROR\b|\bfailed\b|\bFailed\b)', line))


def failure_summary(logs):
    """First traceback/exception and first terminal Error, never a caret/frame."""
    # The routine import_utils WARNING (duplicate NCCL runtime probe) prints its own traceback at every start;
    # it is never the failure (fatal_line excludes it too).
    lines = [l for l in logs.splitlines() if not ('import_utils' in l and 'WARNING' in l)]
    exception = next((l for l in lines if 'Traceback (most recent call last)' in l), None)
    error = next((l for l in lines if re.search(r'\b(?:[A-Za-z_][\w.]*Error|[A-Za-z_][\w.]*Exception|Exception):', l)), None)
    if exception is None:exception = error
    if error is None:
        error = next((l for l in lines if fatal_line(l) and not re.search(r'(?:\^+$|File "|Traceback|\]\s+(?:return |raise |self\.|[a-z_]+ =))', l)), None)
    return ' | '.join(dict.fromkeys(l for l in (exception, error) if l))


def inspect_samples(samples, swap_base, growing):
    available = []
    failures = []
    for rank, sample in enumerate(samples):
        mem = sample['mem']; available.append(mem['MemAvailable'])
        if sample.get('restarts', 0):
            raise RuntimeError(f'rank {rank} container restarted')
        if not sample['state']['Running'] or sample['state']['OOMKilled']:
            summary=failure_summary(sample['logs'])
            failures.append(f'rank {rank} exited or was OOM killed'+(': '+summary if summary else ''))
            continue
        live = floor_gib('RECIPE_LIVE_FLOOR_GIB')
        if mem['MemAvailable'] < live * 1048576:
            raise RuntimeError(f'rank {rank} MemAvailable below {live:g} GiB')
        used = mem['SwapTotal'] - mem['SwapFree']
        swap_base.setdefault(rank, used)
        growing[rank] = growing.get(rank, 0) + 1 if used - swap_base[rank] >= 64 * 1024 else 0
        if growing[rank] >= 3:
            raise RuntimeError(f'rank {rank} sustained swap growth >=64 MiB')
        errors = [line for line in sample['logs'].splitlines() if fatal_line(line)]
        if errors:
            failures.append(f'rank {rank} runtime error: {failure_summary(sample["logs"]) or errors[0]}')
    if failures:
        raise RuntimeError('\n'.join(failures))
    return available


def capture_headroom(samples, avail):
    # The in-process hook refuses capture below GLM_PRECAPTURE_FLOOR_GIB before it starts. The capture
    # progress lines stay in the log tail until health 200, so this floor (RECIPE_CAPTURE_HEADROOM_GIB) covers
    # capture, the post-capture warm-up and API start. The cold FlashInfer JIT builds
    # that dipped below it are made before launch by jit_prep().
    capturing = any('Capturing CUDA graph' in s['logs'] or 'Capturing cudagraph' in s['logs'] for s in samples)
    floor = floor_gib('RECIPE_CAPTURE_HEADROOM_GIB')
    if capturing and min(avail) < floor * 1048576:
        raise RuntimeError('capture headroom below %g GiB' % floor)


def progress_fingerprint(samples, metrics=None):
    if metrics is not None:
        return '\n'.join(l for l in metrics.splitlines() if l.startswith('vllm:generation_tokens_total') or l.startswith('vllm:prompt_tokens_total'))
    # Health/access logs cannot keep a stalled cold boot alive.
    return '\n'.join(line for sample in samples for line in sample['logs'].splitlines()
                     if not re.search(r'\"(GET|POST) |/health|/metrics|Avg prompt throughput|Avg generation throughput', line))


def monitor(deployment, boot=False):
    started = time.monotonic()
    steady = None
    swap_base = {int(k): v for k, v in deployment.get('swap_base', {}).items()}
    growing = {}
    progress = {}
    ready = False
    saver = None
    next_sample = next_health = started
    directory = ROOT / 'logs' / deployment['ctn']
    directory.mkdir(parents=True, exist_ok=True)
    try:
        while True:
            now = time.monotonic()
            if boot and not ready and now >= next_health:
                next_health = now + HEALTH_POLL_S
                ready = health()
                if ready:
                    print('health 200 after %.1f s' % (now-started), flush=True)
                    next_sample = now      # the admission window starts at the first ready sample
            if now < next_sample:
                time.sleep((min(next_sample, next_health) if boot and not ready else next_sample) - now)
                continue
            next_sample = now + (SAMPLE_BOOT_S if boot else SAMPLE_STEADY_S)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(remote, rank, 'python3 -c ' + shlex.quote(
                    'NAME=' + repr(deployment['ctn'] + f'-r{rank}') + '\n' + SAMPLE_CODE)) for rank in range(4)]
                samples = [json.loads(f.result()) for f in futures]
            with (directory / 'watchdog.jsonl').open('a') as out:
                out.write(json.dumps(dict(elapsed_s=now-started, samples=samples)) + '\n')
            avail = inspect_samples(samples, swap_base, growing)
            if boot and not ready:
                capture_headroom(samples, avail)
            # Serving-clone watchdog (nvfp4w/nvfp4a): sidecar hash + Marlin repack margin.
            if boot and now-started > 1800:
                raise RuntimeError('boot exceeded 1800 seconds')
            if ready and boot:
                admission = floor_gib('RECIPE_ADMISSION_FLOOR_GIB')
                steady = (steady or now) if min(avail) >= admission * 1048576 else None
                if steady is not None and now-steady >= 60:
                    print('Admission PASS: >=%g GiB on all ranks for 60 s (%.1f s after start)' % (admission, now-started), flush=True)
                    select_serving_prefill()
                    if deployment.get('persistent_cache'):
                        saver = PersistentCacheSave(deployment, now)
                    boot = False
                    next_sample = now + SAMPLE_STEADY_S
            if not boot:
                page_cache_state('ready')
            if not boot:
                metrics = remote(0, 'curl -fsS -m 5 http://127.0.0.1:8095/metrics')
                fingerprint = progress_fingerprint(samples, metrics)
            else:
                fingerprint = progress_fingerprint(samples)
            busy = any(float(v.strip()) > 20 for sample in samples for v in sample['gpu'].splitlines())
            # Idle time must not consume the next request's stall allowance (as the serving watch).
            if progress.get('value') != fingerprint or not busy:
                progress = dict(value=fingerprint, time=now)
            # A cold long prefill completes before /metrics counts it (serving watch allowance).
            if busy and now-progress.get('time', now) > 600:
                raise RuntimeError('GPU busy without progress for 600 seconds')
            if saver is not None:
                saver.poll(now)
    except BaseException:
        # SSH loss is fatal; no automatic restart or power operation.
        if saver is not None:
            saver.cancel()
        stop(deployment)
        raise


def serve(ctn):
    if STATE.exists():
        raise RuntimeError('Local deployment state exists; use status/stop before another serve')
    # Cached-header diagnostic runs locally before fleet operations and never vetoes boot.
    import tempfile
    from boot_preflight import advisory
    try:
        with tempfile.TemporaryDirectory(prefix='glm-preflight-') as tmp:
            dry = Path(tmp)/'dry.txt'
            headers = ENV.get('RECIPE_BOOT_PREFLIGHT_HEADERS')
            if headers and Path(headers).is_file():
                dry.write_text('\n'.join(shlex.join(docker_command(rank, ctn)) for rank in range(4))+'\n')
            result = advisory(ROOT, dry, Path(tmp)/'report.json', headers,
                       ENV.get('RECIPE_BOOT_PREFLIGHT_IMAGE'), ENV.get('RECIPE_BOOT_PREFLIGHT', '1') == '1')
            logdir = ROOT/'logs'/ctn
            logdir.mkdir(parents=True, exist_ok=True)
            (logdir/'cpu-boot-preflight.json').write_text(json.dumps(result, indent=2)+'\n')
    except Exception as exc:
        print('WARNING: CPU boot preflight unavailable: '+str(exc)+'; continuing', file=sys.stderr)
    page_cache_state('begin')  # includes checkpoint reads in preflight verification
    preflight(ctn)
    token = ctn + ':' + uuid.uuid4().hex
    lock(token)
    deployment = dict(ctn=ctn, token=token, hosts=HOSTS, ips=IPS, images=IMAGES,
                      runtime=ENV['OVERLAY_REMOTE'], profile=ENV.get('RECIPE_PROFILE', 'native-mtp-k2'), created=time.time())
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(deployment, indent=2) + '\n')
    launched = []
    try:
        for rank in range(4):
            remote(rank, 'mkdir ' + shlex.quote(ENV['OVERLAY_REMOTE']))
            subprocess.run(['rsync', '-a', '--exclude=__pycache__', str(ROOT / 'overlay') + '/',
                            HOSTS[rank] + ':' + ENV['OVERLAY_REMOTE'] + '/'], check=True)
            remote(rank, 'mkdir ' + shlex.quote(ENV['OVERLAY_REMOTE'] + '/cache'))
        if persistent_cache_enabled():
            deployment['persistent_cache'] = persistent_cache_seed(ctn)
            STATE.write_text(json.dumps(deployment, indent=2) + '\n')
        # Before compaction, so the compilers' memory is gone before the preboot floor check.
        jit_prep(ctn)
        hook = dispram()
        if hook is not None:
            for rank in range(4):
                subprocess.run(['rsync', '-a', str(ROOT / 'scripts/dispram.sh'),
                                HOSTS[rank] + ':' + hook.home(ENV) + '/dispram.sh'], check=True)
            hook.ensure(HOSTS, ENV)     # require: raises before any engine starts
        for rank in range(4):
            remote(rank, 'sudo -n /usr/local/sbin/spark-compact-mem.sh', timeout=180)
            # Compaction must succeed and must preserve the 110 GiB preboot floor.
            available = int(remote(rank, "awk '/MemAvailable/{print $2}' /proc/meminfo"))
            if available < 110 * 1048576:
                raise RuntimeError('Preboot memory floor after compaction')
        prefill_control(512, 0)
        deployment['swap_base'] = {rank: int(remote(rank, "awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print t-f}' /proc/meminfo")) for rank in range(4)}
        STATE.write_text(json.dumps(deployment, indent=2) + '\n')
        for rank in (3, 2, 1, 0):
            print(HOSTS[rank], remote(rank, shlex.join(docker_command(rank, ctn))), flush=True)
            launched.append(rank)
    except BaseException:
        stopped = True
        for rank in launched:
            try:
                remote(rank, 'docker stop -t 60 ' + shlex.quote(f'{ctn}-r{rank}'), timeout=80)
            except Exception:
                stopped = False
        if stopped:
            unlock(token)
        raise
    print('Watchdog active in foreground; use another terminal for status and benchmarks. Ctrl-C stops the deployment.', flush=True)
    monitor(deployment, boot=True)


def main():
    validate()
    launch_switches()
    floors = floors_note()
    command = sys.argv[1] if len(sys.argv) > 1 else 'status'
    ctn = ENV.get('CTN', 'glm53full-' + time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])
    if not re.fullmatch(r'glm53full-[A-Za-z0-9_-]+', ctn):
        raise ValueError('CTN must begin glm53full- and contain only letters, digits, _ or -')
    if ENV.get('DRY') == '1':
        if command != 'serve':
            raise ValueError('DRY=1 supports serve only')
        if persistent_cache_enabled():
            for rank in range(4):
                print(f'# persistent-cache-seed rank={rank} host={HOSTS[rank]} (image ID resolved at serve)')
                print(shlex.join(persist_command(rank, ctn, 'sha256:' + '0' * 64, 'seed')))
        if kstop_note():
            print('# ' + kstop_note())
        print('# ' + floors)
        for rank in range(4):
            print(f'# jit-prep rank={rank} host={HOSTS[rank]}')
            print(shlex.join(jit_prep_command(rank, ctn)))
        for rank in (3, 2, 1, 0):
            print(f'# rank={rank} host={HOSTS[rank]}')
            print(shlex.join(docker_command(rank, ctn)))
        if persistent_cache_enabled():
            for rank in range(4):
                print(f'# persistent-cache-save rank={rank} host={HOSTS[rank]} (after admission)')
                print(shlex.join(persist_command(rank, ctn, 'sha256:' + '0' * 64, 'save', '0' * 32)))
        return
    if command == 'dispram-setup':
        hook = dispram()
        if hook is None:
            raise ValueError('dispram-setup needs RECIPE_DISPRAM=1/auto or require')
        hook.setup(HOSTS, ENV, ROOT)
        return
    if command == 'preflight':
        preflight(ctn)
        return
    if command == 'serve':
        if kstop_note():
            print(kstop_note(), flush=True)
        print(floors, flush=True)
        serve(ctn)
        return
    deployment = json.loads(STATE.read_text())
    if deployment['hosts'] != HOSTS or deployment['runtime'] != ENV['OVERLAY_REMOTE']:
        raise RuntimeError('Saved deployment differs from .env; restore its host/runtime settings')
    if command == 'stop':
        stop(deployment)
        STATE.rename(STATE.with_name('stopped-' + deployment['ctn'] + '.json'))
    elif command == 'watch':
        monitor(deployment)
    elif command == 'status':
        for rank in range(4):
            print(HOSTS[rank], remote(rank, 'docker inspect -f ' + shlex.quote('{{.State.Running}} {{.State.OOMKilled}}') + ' ' + shlex.quote(deployment['ctn'] + f'-r{rank}')))
        print('health', health())
    elif command == 'logs':
        rank = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        if rank not in range(4):
            raise ValueError('rank must be 0..3')
        print(remote(rank, 'docker logs --tail 100 ' + shlex.quote(deployment['ctn'] + f'-r{rank}') + ' 2>&1'))
    else:
        raise ValueError('usage: start.sh preflight|serve|status|logs [rank]|watch|stop|dispram-setup')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
