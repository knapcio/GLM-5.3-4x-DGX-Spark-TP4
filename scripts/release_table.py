#!/usr/bin/env python3
"""Format a complete frozen-config sparkDash receipt as README Markdown, offline."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

KINDS = ('prose', 'code', 'structured', 'json')
COUNTS = {1: 5, 2: 3, 4: 3, 8: 2}


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('finite positive measured metric required')
    return value


def tables(rows):
    summaries = [r for r in rows if r.get('mode') == 'full' and r.get('sweep')]
    if len(summaries) != 1 or summaries[0]['sweep'].get('max_model_len') != 262144:
        raise ValueError('one full 262144-context sweep required')
    if any(r.get('error') for r in rows):
        raise ValueError('receipt contains a failed job')
    out = ['| Prompt type | c1 | c2 | c4 | c8* |', '|---|---:|---:|---:|---:|']
    for kind in KINDS:
        cells = []
        for concurrency, count in COUNTS.items():
            selected = [r for r in rows if r.get('phase') == 'scored'
                        and r.get('kind') == kind and r.get('concurrency') == concurrency]
            if len(selected) != count:
                raise ValueError(f'{kind} c{concurrency}: expected {count} scored runs')
            aggregate, streams = [], []
            for row in selected:
                result = row['result']
                if result.get('status') not in (None, 'completed') or len(result['results']) != 1:
                    raise ValueError('incomplete decode job')
                metric = result['results'][0]
                if metric.get('streamsOk') != concurrency or metric.get('streamsFailed', 0) or metric.get('error'):
                    raise ValueError('failed decode stream')
                aggregate.append(positive(metric['aggregateDecodeTps']))
                streams.append(positive(metric['meanDecodeTps']))
            value = f'{statistics.median(aggregate):.2f}'
            if concurrency > 1:
                value += f' [{statistics.median(streams):.2f}]'
            cells.append(value)
        out.append('| ' + kind + ' | ' + ' | '.join(cells) + ' |')
    tps, ttft = [], []
    for size in (4096, 8192, 16384, 32768):
        selected = []
        for row in rows:
            if row.get('phase') != 'prefill-scored':
                continue
            result = row['result']
            if result.get('status') not in (None, 'completed'):
                raise ValueError('incomplete prefill job')
            selected.extend(x for x in result['results']
                            if int(x.get('targetTokens') or x.get('contextSize') or 0) == size)
        if len(selected) != 3:
            raise ValueError(f'prefill {size}: expected three scored runs')
        if any(x.get('error') for x in selected):
            raise ValueError('failed prefill result')
        tps.append(f"{statistics.median(positive(x['prefillTps']) for x in selected):.0f}")
        ttft.append(f"{statistics.median(positive(x['ttftMs']) for x in selected)/1000:.2f}")
    out += ['', '| Input | 4K | 8K | 16K | 32K |', '|---|---:|---:|---:|---:|',
            '| cold prefill tok/s | ' + ' | '.join(tps) + ' |',
            '| time to first token (s) | ' + ' | '.join(ttft) + ' |']
    return '\n'.join(out) + '\n'


def qualification(run):
    """Render the reduced gate without upgrading carried or absent measurements."""
    sources = {}

    def read(name):
        path = run / name
        sources[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    complete = read('COMPLETE.json')
    frozen = read('release-run.json')
    if not complete['all_pass'] or not complete['reduced'] or not frozen['reduced']:
        raise ValueError('a completed reduced release gate is required')
    if (complete['config'] != frozen['config'] or complete['phases'] != frozen['phases']
            or complete['carried'] != frozen['carried']):
        raise ValueError('completion/config binding differs')
    boot = frozen['config']['ctn']
    carried = frozen['carried']['groups']
    for group in carried.values():
        for name, digest in group['files'].items():
            path = Path(name)
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != digest:
                raise ValueError(f'carried receipt hash differs: {path.name}')
            sources[path] = actual

    def carry(group, suffix):
        names = [n for n in carried[group]['files'] if n.endswith(suffix)]
        if len(names) != 1:
            raise ValueError(f'one carried {group}/{suffix} required')
        return json.loads(Path(names[0]).read_text())

    def ranks(values):
        return '/'.join(str(values[str(i)]) for i in range(4))

    cells = []

    def cell(label, value, paths, fields):
        cells.append((label, value, paths, fields))

    q = carry('qpanel_B', '/RESULT.json')
    decision = carry('qpanel_B', '/DECISION.json')
    qeval = []
    for group in ('qeval_g5', 'qeval_g6'):
        name = next(n for n in carried[group]['files'] if '/qeval-' in n and n.endswith('.json')
                    and not n.endswith('item-diff.json'))
        data = json.loads(Path(name).read_text())
        diff = carry(group, '/qeval-screen-item-diff.json')
        rows = data['results']
        primary = [r for r in rows if r['category'] in ('code', 'reason', 'math')]
        qeval.append(f"carried w4-ehproj-{group.removeprefix('qeval_')}: {sum(r['pass'] for r in rows)}/{len(rows)}, "
                     f"primary {sum(r['pass'] for r in primary)}/{len(primary)}, "
                     f"{data['truncated']} truncation, {sum(not r['pass'] for r in rows)} failed tasks; "
                     f"decision {diff['decision']}" +
                     (f" (release_decision {diff['release_decision']})" if 'release_decision' in diff else ''))
    cell('Quality admission; qeval status',
         f"Carried qpanel/B ({q['boot']}): {q['passed']}/{q['items']}, "
         f"{q['truncated']} truncations; decision {decision['decision']}. "
         + '; '.join(qeval) + '. Qeval x3 not run on release boot.',
         ['qpanel_B', 'qeval_g5', 'qeval_g6', 'release-run.json'],
         'passed/items/truncated; results[].pass/category; decision/release_decision; phases')
    needles = carry('needles_g6', '/summary.json')
    small = [r for r in needles['runs'] if r['target_input_tokens'] == 16384]
    cell('Needle 16K, carried twice',
         f"Carried w4-ehproj-g6: {', '.join(r['grade']['score_over_n'] for r in small)}; "
         'not rerun on release boot.', ['needles_g6', 'release-run.json'], 'runs[].grade.score_over_n; phases')
    needle = read('needle/summary.json')
    if needle['boot_label'] != boot or len(needle['runs']) != 1:
        raise ValueError('one release-boot needle run required')
    n = needle['runs'][0]
    missed = [key for key, field in n['grade']['fields'].items() if not field['correct']]
    cell('Needle 128K, one run',
         f"{len(needle['runs'])} run, {n['grade']['score_over_n']}; missed `{', '.join(missed)}`; "
         f"TTFT {n['ttft_seconds']} s; {needle['verdict']['status']} under the reduced rule.",
         ['needle/summary.json'], 'runs; grade.score_over_n/fields; ttft_seconds; verdict.status')
    cell('Needle 250K', 'Not run; no carried 250K receipt in the reduced scope.',
         ['release-run.json', 'COMPLETE.json'], 'phases; carried.groups (no 250K measurement)')
    stress = read('done-stress.json')['result']
    memory = read('stress/memory-summary.json')
    trace = read('stress/c4/c4-trace.json')
    single = read('stress/single.json')
    preempt = max(r['preemptions'] for r in trace['trace'])
    swap = {}
    for rank in range(4):
        path = run / f'stress/rank{rank}-memory.jsonl'
        sources[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        samples = [json.loads(line) for line in path.read_text().splitlines()]
        swap[str(rank)] = max(s['mem_kB']['SwapTotal'] - s['mem_kB']['SwapFree'] for s in samples)
    if stress['status'] != 'PASS' or memory['problem'] is not None or preempt or any(swap.values()):
        raise ValueError('stress must pass with zero observed swap/preemptions')
    cell('c4 shared-pool and single maximum-context stress',
         f"{stress['status']}; c4 {stress['geometry']['c4_prompt_tokens']}+{stress['output_tokens']} each, "
         f"single {stress['single_prompt']}+{single['usage']['completion_tokens']}; "
         f"rank 0/1/2/3 minima {ranks(stress['min_GiB'])} GiB; "
         f"swap maxima {ranks(swap)} KiB; preemptions {preempt}.",
         ['done-stress.json', 'stress/memory-summary.json', 'stress/c4/c4-trace.json', 'stress/single.json']
         + [f'stress/rank{i}-memory.jsonl' for i in range(4)],
         'result.geometry/single_prompt/output_tokens/min_GiB; problem; trace[].preemptions; usage; max(SwapTotal-SwapFree)')
    apc = read('stress/c4/apc.json')
    mem = read('done-memstress.json')
    rig = read('rigmark-summary.json')
    receipt = read('rigmark-receipt.json')
    if receipt['settings']['extra_body']['chat_template_kwargs']['reasoning_effort'] != 'low':
        raise ValueError('RigMark effort low required')
    cell('Prefix-cache first/repeat and retained-cache stress',
         f"8K cold/replay prefill {rig['rates']['prefill_8k_cold']}/{rig['rates']['prefill_8k_warm_replay']} tok/s; "
         f"stress APC hits/queries {apc['hit_delta']}/{apc['query_delta']} tokens; "
         f"retained-cache quiet minima {'/'.join(map(str, mem['quiet_min_GiB']))} GiB (ranks 0/1/2/3).",
         ['rigmark-summary.json', 'rigmark-receipt.json', 'stress/c4/apc.json', 'done-memstress.json'],
         'rates.prefill_8k_cold/prefill_8k_warm_replay; hit_delta/query_delta; quiet_min_GiB')
    soak = read('done-soak.json')['result']
    if soak['boot'] != boot or soak['verdict'] != 'PASS' or soak['soak_s'] < 900:
        raise ValueError('completed release-boot soak >=900 seconds required')
    cell('Mixed-load soak',
         f"{soak['verdict']}; requested {soak['minutes']} min, observed {soak['soak_s']} s; "
         f"{soak['requests']} requests, {soak['errors']} errors, {soak['preemptions']} preemptions; "
         f"quiet drift {ranks(soak['mem_drift_GiB'])} GiB (ranks 0/1/2/3).",
         ['done-soak.json', 'soak/RESULT.json'], 'result.minutes/soak_s/requests/errors/preemptions/mem_drift_GiB')
    read('soak/RESULT.json')
    init = read('done-initgate1.json')
    for rank in range(4):
        path = run / f'init-qual-rank{rank}.jsonl'
        sources[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if not any(r['check'] == 'ehproj_captured' and r['conditions']['combined_init_qualified'] for r in rows):
            raise ValueError('combined eh_proj qualification missing')
    if not all(r['on'] and r['ready'] and r['target_bit_exact'] for r in init['ranks']):
        raise ValueError('all-rank ready/target equality missing')
    cell('Draft-head + eh_proj INIT',
         f"{len(init['ranks'])}/{len(init['ranks'])} ranks ON/ready, combined FP8 eh_proj qualified; "
         f"{init['ranks'][0]['n_cases']} cases and "
         f"{'/'.join(str(init['terms'][str(i)]['rows']) for i in range(4))} rows per rank; "
         f"{sum(v['failed'] for v in init['terms'].values())} failed; target head bit-exact on every rank.",
         ['done-initgate1.json'] + [f'init-qual-rank{i}.jsonl' for i in range(4)],
         'ranks[].on/ready/target_bit_exact/n_cases; terms; ehproj_captured.conditions.combined_init_qualified')
    t0 = read('done-t0.json')
    equality = read('release-t0-stack2-equality.json')
    report = read('release-t0/result.json')
    read('release-t0-comparison.json')
    if (sources[run/'release-t0-comparison.json'] != t0['report_sha256']
            or sources[run/'release-t0-stack2-equality.json'] != t0['equality_sha256']
            or report['boot_label'] != boot or report['status'] != 'COMPLETE'):
        raise ValueError('T0 receipt binding differs')
    cell('Temperature-0 sequential repeats on this boot',
         f"{t0['determinism']}/{t0['determinism']} prompts identical across {equality['repeats']} repeats; "
         f"{t0['native_equal']}/{t0['determinism']} native-ID equality to carried stack-g2/glue-t0.",
         ['done-t0.json', 'release-t0/result.json', 'release-t0-comparison.json',
          'release-t0-stack2-equality.json', 'glue_stack2'],
         'determinism/native_equal; repeats/reference')
    if rig['rc'] or not all(rig['gates'].values()):
        raise ValueError('RigMark workload gates must pass')
    cell(f"RigMark {receipt['protocol']['version']}, thinking on, effort low",
         '; '.join(f"{kind} {rig['rates'][kind]} tok/s" for kind in ('prose', 'code', 'structured'))
         + f"; code c4 aggregate {rig['rates']['code_c4_aggregate']} tok/s; "
         f"8K cold prefill {rig['rates']['prefill_8k_cold']} tok/s; all workload gates passed.",
         ['rigmark-summary.json', 'rigmark-receipt.json'], 'rates/gates; protocol.version; settings.extra_body')
    sources[run/'sparkdash.jsonl'] = hashlib.sha256((run/'sparkdash.jsonl').read_bytes()).hexdigest()
    dash = read('dash-vs-det-release.json')
    if dash['same_boot'] != boot:
        raise ValueError('sparkDash boot differs')
    for path in sources:
        if path.parent == run and path.name.startswith('done-'):
            identity = json.loads(path.read_text())['identity']
            if any(identity[key] != frozen['config'][key] for key in ('ctn', 'sha', 'env')):
                raise ValueError(f'phase identity differs: {path.name}')
    markdown = '| Check | Release result |\n|---|---|\n'
    markdown += '\n'.join(f'| {label} | {value} |' for label, value, _, _ in cells) + '\n'
    # All published paths are portable relative to the diagnostics directory.
    def portable(path):
        parts = path.parts
        return '/'.join(parts[parts.index('diagnostics'):])
    provenance = ['## Recorded reduced gate and cell provenance', '',
                  f"Reduced gate **PASS**, boot `{boot}`, serving `{frozen['config']['sha']}`.",
                  'The public profile boot equivalences above bind this serving vector to the export.',
                  'Fresh phases: ' + ', '.join(frozen['phases']) + '.',
                  'Qpanel and qeval were carried; the time-slicing scenario and long soak were dropped.',
                  'Fresh soak used the reduced duration; fresh needle used only 128K x1.',
                  '16K x2 is carried from w4-ehproj-g6. 250K was neither run nor carried.',
                  'The qeval regression decisions remain KILL; reduced PASS does not upgrade them.', '',
                  '| README cells | Receipt references | Fields / interpretation |', '|---|---|---|',
                  '| sparkDash: prose/code/structured/json x c1/c2/c4/c8 (16 cells), cold prefill and TTFT x 4K/8K/16K/32K (8 cells) | `sparkdash.jsonl`, `dash-vs-det-release.json` | Every scored run; `tables()` median/format rules. Comparison columns are not published. |']
    provenance += [f"| {label} | " + ', '.join(f'`{p}`' for p in paths) + f' | {fields} |'
                   for label, _, paths, fields in cells]
    provenance += ['', 'Local receipt root: `' + portable(run) + '/`. Each short filename above resolves there.',
                   'Carried group references resolve to the complete file list below; hashes were verified before rendering.', '',
                   '| Carried group / source run | Recorded interpretation |', '|---|---|']
    provenance += [f"| `{group}` | {data['note']} |" for group, data in carried.items()]
    provenance += ['', '| Receipt path | SHA256 |', '|---|---|']
    provenance += [f'| `{portable(path)}` | `{digest}` |' for path, digest in sorted(sources.items())]
    return markdown, '\n'.join(provenance) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--gate-run', type=Path, help='also render reduced-gate quality cells')
    parser.add_argument('--provenance', type=Path, help='write portable receipt paths and SHA256s')
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.receipt.read_text().splitlines() if line.strip()]
    try:
        output = tables(rows)
        if args.gate_run:
            if args.receipt.resolve() != (args.gate_run/'sparkdash.jsonl').resolve():
                raise ValueError('sparkDash receipt must belong to gate run')
            quality, provenance = qualification(args.gate_run)
            output += '\n' + quality
            if args.provenance:
                args.provenance.write_text(provenance)
        elif args.provenance:
            raise ValueError('--provenance requires --gate-run')
        print(output, end='')
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
