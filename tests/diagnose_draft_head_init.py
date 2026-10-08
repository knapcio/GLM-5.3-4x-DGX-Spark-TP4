# SPDX-License-Identifier: Apache-2.0
"""Extract saved per-rank refusal, without contacting the serving engine."""
import hashlib, json, re, sys
from pathlib import Path
boot, toggle, out = map(Path, sys.argv[1:])
x = json.loads(boot.read_text())['results']
rows = []
for rank in range(4):
    status = next(s for s in x if s['rank'] == rank)
    terms = next(v['terms'] for v in status['init_failure']['votes'] if v['rank'] == rank)
    failures = [t for t in terms if t['check'] == 'replay_compare' and not all(t['conditions'].values())]
    assert len(failures) == 1
    failed = failures[0]
    desc = failed['measured']['descriptor']
    match = re.search(r'num_tokens=(\d+), num_reqs=(\d+), uniform_token_count=(\d+).*short_context=(True|False)', desc)
    assert match and tuple(match.groups()) == ('16', '4', '4', 'False')
    comparisons = [t for t in terms if t['check'] == 'replay_compare']
    assert [t['measured']['value'] for t in comparisons] == [0., .125]
    output = failed['measured']['outputs']
    assert output[0]['bit_exact'] and output[0]['mismatches'] == 0
    assert output[1]['mismatches'] == 4 and output[1]['max_abs'] == .00390625
    assert all(o['eager_nonfinite'] == o['graph_nonfinite'] == 0 for o in output)
    toggle_status = next(s for s in json.loads(toggle.read_text()) if s['rank'] == rank)
    assert all(desc not in keys for keys in toggle_status['graphs'])
    rows.append(dict(rank=rank, qualification_rows=len(terms), manager=0,
        num_tokens=16, num_reqs=4, uniform_token_count=4, short_context=False,
        input_index=1, value=.125, failed_conditions=['bit_exact'],
        outputs=output, absent_from_saved_toggle_graph_bank=True,
        gate_passed=status['dh_gate_passed'], on=status['on'], ready=status['ready']))
boot_log, toggle_log = boot.parent/'rank0-boot.log', toggle.parent/'rank0-boot.log'
configs = {}
for name, path in (('candidate', boot_log), ('toggle', toggle_log)):
    text = path.read_text()
    configs[name] = dict(num_speculative_tokens=int(re.search(r"'num_speculative_tokens': (\d+)", text)[1]),
        capture_sizes=json.loads(re.search(r"'cudagraph_capture_sizes': (\[[^\]]*\])", text)[1]))
assert configs['candidate']['num_speculative_tokens'] == 3
assert configs['toggle']['num_speculative_tokens'] == 4
assert configs['candidate']['capture_sizes'] == configs['toggle']['capture_sizes'] == [1, 4, 12, 16]
result = dict(ranks=rows, boot_configs=configs, sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (boot, toggle, boot_log, toggle_log)},
    verified_cause='eager/graph draft floating equality rejects regular 4x4 prefill; proposals identical',
    unproven=['specific CUDA kernel/algorithm', 'first-use warmup', 'repeat determinism', 'null-block contamination'],
    note='Boot refuses on its first descriptor at input1; later descriptors and input2 were not reached. Saved toggle bank is different; this is not proof all historical toggle banks omit it.')
out.write_text(json.dumps(result, indent=2)+'\n')
print('DIAGNOSIS PASS: four ranks, regular4x4 prefill, input1=.125, identical proposals; saved toggle descriptor absent')
