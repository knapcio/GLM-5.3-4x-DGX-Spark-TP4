# SPDX-License-Identifier: Apache-2.0
"""Strict logical inventories shared by the offline converter and real loader."""
import os
import re
import glm_nvfp4_format as fmt

ORDER = ('attn', 'shared', 'indexer', 'dense', 'mtp')
COUNTS = dict(attn=385, shared=225, indexer=63, dense=6, mtp=776)


def groups(env=None):
    value = (os.environ if env is None else env).get('GLM_NVFP4_GROUPS', 'attn')
    parts = value.split(',')
    if not parts or any(p not in ORDER for p in parts) or len(set(parts)) != len(parts):
        raise ValueError('GLM_NVFP4_GROUPS requires unique attn,shared,indexer,dense,mtp')
    return tuple(g for g in ORDER if g in parts)


def classify(prefix):
    m = re.fullmatch(r'(?:.*\.)?layers\.(\d+)\.(?:mtp_block\.)?(.+)', prefix)
    if not m:
        return None
    i, site = int(m[1]), m[2]
    projections = (*fmt.PROJECTIONS, 'fused_qkv_a_proj')
    if i == 78:
        if (site in tuple('self_attn.'+p for p in projections) or
            re.fullmatch(r'mlp\.(?:shared_experts|experts\.\d+)\.(?:gate_proj|up_proj|down_proj|gate_up_proj)', site) or
            site == 'mlp.experts'):
            return 'mtp'
        return None
    if not 0 <= i < 78:
        return None
    if re.fullmatch(r'self_attn\.indexer\.(?:wq_b|wk|weights_proj|wk_weights_proj)', site):
        return 'indexer'
    if i == 0:
        return None
    if site in tuple('self_attn.'+p for p in projections):
        return 'attn'
    if 3 <= i <= 77 and re.fullmatch(r'mlp\.shared_experts\.(?:gate_proj|up_proj|down_proj|gate_up_proj)', site):
        return 'shared'
    if i in (1, 2) and site in tuple('mlp.'+p for p in ('gate_proj', 'up_proj', 'down_proj', 'gate_up_proj')):
        return 'dense'
    return None


def inventory(index, selected, strict=True):
    result = {}
    for name in sorted(index):
        prefix, _, leaf = name.rpartition('.')
        group = classify(prefix)
        if group not in selected or leaf != ('weight' if group == 'indexer' else 'weight_packed'):
            continue
        leaves = ('weight',) if group == 'indexer' else ('weight_packed', 'weight_scale', 'weight_shape')
        if any(prefix+'.'+l not in index for l in leaves):
            raise ValueError('missing source companions: '+prefix)
        result[prefix] = dict(group=group, leaves=list(leaves), group_size=-1 if group == 'mtp' else 128)
    if strict:
        counts = {g:sum(e['group'] == g for e in result.values()) for g in selected}
        if counts != {g:COUNTS[g] for g in selected}:
            raise ValueError('incomplete full GLM-5.3 group inventory: '+str(counts))
    return result
