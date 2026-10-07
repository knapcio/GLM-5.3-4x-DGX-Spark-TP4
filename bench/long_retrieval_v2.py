#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Seeded public synthetic retrieval probe; no server lifecycle or lock access."""
import argparse
import bisect
import hashlib
import http.client
import json
from pathlib import Path
import random
import re
import time
from datetime import datetime
from urllib.parse import urlsplit

import os
# Targets are configurable with NEEDLE_TARGETS; defaults match the qualified 98K layout.
TARGETS = tuple(int(x) for x in os.environ.get('NEEDLE_TARGETS', '16384,63488,96128').split(','))
REGISTRY_FIELDS = tuple(f"r{i}" for i in range(1, 8))
FIELDS = REGISTRY_FIELDS + ("alias_result", "corrected_operator", "matching_count")
CHAT_KWARGS = {"enable_thinking": False, "reasoning_effort": "low"}
MAX_TOKENS = 1024
PASS_RULE = ("For EACH paired repeat (1 and 2), EACH long length (63K and max-2K): "
             "score >= 16K score - 1 and all seven registry fields correct. "
             "All six responses must complete with finish_reason=stop; "
             "transport/protocol failures produce ERROR, never PASS.")


def compact(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=True)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def panel(rows, seed):
    """Separate RNG streams keep facts and depth strata identical across lengths."""
    if rows < 100:
        raise ValueError("at least 100 rows required")
    rng = random.Random(seed)
    keys = ["LC_" + "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=10))
            for _ in range(9)]
    values = ["V_" + "".join(rng.choices("abcdefghjkmnpqrstuvwxyz23456789", k=6))
              for _ in range(7)]
    region, cutoff = rng.randrange(5), rng.randrange(25, 76)
    revision, result_index = rng.randrange(20, 90), rng.randrange(7)
    predicate = rng.choice(("inside", "outside"))
    corrected = "and" if predicate == "inside" else "or"
    broken = "or" if corrected == "and" else "and"
    row_rng = random.Random(f"{seed}:archive")
    records, count = [], 0
    for i in range(rows):
        r, n, state = row_rng.randrange(5), row_rng.randrange(100), row_rng.choice(("open", "closed"))
        count += int(r == region and n >= cutoff and state == "open")
        records.append(f"Archive row {i:07d}: region {r}; quantity {n}; state {state}.")
    needles = []
    for i, (key, value) in enumerate(zip(keys, values)):
        needles += [f"Registry record: key {key}; revision {revision}; exact value {value}.",
                    f"Registry record: key {key}; revision {revision-1}; exact value old_{i}.",
                    f"Registry record: key {key}X; revision {revision+1}; exact value decoy_{i}."]
    needles += [f"Alias record: key {keys[7]}; revision {revision}; target {keys[8]}.",
                f"Alias record: key {keys[7]}; revision {revision-1}; target {keys[(result_index+1)%7]}.",
                f"Alias record: key {keys[8]}; revision {revision}; target {keys[result_index]}.",
                f"Alias record: key {keys[8]}; revision {revision-1}; target {keys[(result_index+2)%7]}."]
    relation = ("position >= lower {op} position <= upper" if predicate == "inside"
                else "position < lower {op} position > upper")
    needles.append(f"Source audit: {predicate}_window uses {relation.format(op=broken)}. Bounds are inclusive.")
    placement_rng = random.Random(f"{seed}:depths")
    placement_rng.shuffle(needles)
    depths = []
    for j, needle in enumerate(needles):
        # One dependency per stratum; spans early, middle, and late input.
        fraction = (j + placement_rng.uniform(0.2, 0.8)) / len(needles)
        index = min(rows-1, int(fraction * rows))
        records[index] += " " + needle
        depths.append({"row": index, "fraction": fraction, "record": needle})
    schema = {field: "string" for field in FIELDS[:-1]}
    schema["corrected_operator"] = "and|or"
    schema["matching_count"] = "integer"
    task = ("Return ONLY one compact JSON object, without markdown, explanations, or reasoning. "
            "Use exactly these ten scalar fields and types: " + compact(schema) + ". "
            "No arrays or row indices. All string values are at most 8 characters. "
            "Select the greatest numeric revision for each exact key (not the last occurrence). "
            "For alias_result select the greatest revision at EACH alias hop, follow the chain, "
            "then return the latest registry VALUE at the terminal key. "
            'For corrected_operator return only "and" or "or", whichever fixes the Source audit. '
            "Registry output mapping: " + compact(dict(zip(REGISTRY_FIELDS, keys[:7]))) + ". "
            f"Alias to resolve: {keys[7]}. matching_count is COUNT(*) over Archive rows where "
            f"region = {region} AND quantity >= {cutoff} AND state = open. "
            f"Return this COUNT as ONE integer from 0 through {rows}; never enumerate matches. "
            "Total answer must be less than 300 tokens.")
    prompt = task + "\nBEGIN ARCHIVE\n" + "\n".join(records) + "\nEND ARCHIVE\n" + task
    expected = dict(zip(REGISTRY_FIELDS, values))
    expected.update(alias_result=values[result_index], corrected_operator=corrected, matching_count=count)
    return prompt, expected, depths


def oracle(prompt):
    """Recompute ground truth from serialized records, independent of generation count."""
    mapping = json.loads(re.search(r"Registry output mapping: (\{[^\n]+?\})\.", prompt)[1])
    registry, aliases = {}, {}
    for key, revision, value in re.findall(r"Registry record: key (\w+); revision (\d+); exact value (\w+)\.", prompt):
        if key not in registry or int(revision) > registry[key][0]:
            registry[key] = (int(revision), value)
    for key, revision, target in re.findall(r"Alias record: key (\w+); revision (\d+); target (\w+)\.", prompt):
        if key not in aliases or int(revision) > aliases[key][0]:
            aliases[key] = (int(revision), target)
    key = re.search(r"Alias to resolve: (\w+)\.", prompt)[1]
    seen = set()
    while key in aliases:
        if key in seen:
            raise ValueError("alias cycle")
        seen.add(key)
        key = aliases[key][1]
    region, cutoff = map(int, re.search(r"region = (\d+) AND quantity >= (\d+)", prompt).groups())
    count = sum(int(int(r) == region and int(n) >= cutoff and state == "open")
                for r, n, state in re.findall(r"Archive row \d+: region (\d+); quantity (\d+); state (\w+)\.", prompt))
    result = {field: registry[k][1] for field, k in mapping.items()}
    result.update(alias_result=registry[key][1],
                  corrected_operator="and" if "Source audit: inside_window" in prompt else "or",
                  matching_count=count)
    return result


class LocalTokens:
    def __init__(self, directory):
        from tokenizers import Tokenizer
        from jinja2 import Environment, StrictUndefined
        directory = Path(directory)
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.template = Environment(undefined=StrictUndefined, extensions=["jinja2.ext.loopcontrols"]).from_string(
            (directory / "chat_template.jinja").read_text())
        self.pins = {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                     for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")}

    def render(self, prompt):
        return self.template.render(messages=[{"role": "user", "content": prompt}],
                                    add_generation_prompt=True, tools=None, **CHAT_KWARGS)

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)


def generate(target, seed, tokens):
    rows = max(100, target // 24)
    for _ in range(20):
        prompt, expected, depths = panel(rows, seed)
        rendered = tokens.render(prompt)
        encoding = tokens.encode(rendered)
        count = len(encoding.ids)
        if target-256 <= count <= target:
            break
        rows += max(1, (target-128-count)//24) if count < target-256 else min(-1, (target-128-count)//24)
        rows = max(100, rows)
    else:
        raise RuntimeError(f"cannot fit target {target}")
    if oracle(prompt) != expected:
        raise AssertionError("serialized ground truth disagrees")
    answer_tokens = len(tokens.encode(compact(expected)).ids)
    if answer_tokens >= 300:
        raise AssertionError(f"expected answer too long: {answer_tokens}")
    ends = [end for _, end in encoding.offsets]
    for depth in depths:
        start = rendered.index(depth["record"])
        depth["token_start"] = bisect.bisect_right(ends, start)
        depth["token_fraction"] = depth["token_start"] / count
    request = {"model": "GLM-5.3", "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": MAX_TOKENS, "chat_template_kwargs": dict(CHAT_KWARGS),
               "stream": True, "stream_options": {"include_usage": True}, "seed": 0}
    return {"target_input_tokens": target, "input_tokens": count, "rows": rows, "seed": seed,
            "expected": expected, "expected_answer_tokens": answer_tokens, "dependencies": depths,
            "prompt_sha256": sha(prompt), "rendered_chat_sha256": sha(rendered),
            "tokenizer_sha256": tokens.pins, "request": request}


def reject_duplicates(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate key: {key}")
        obj[key] = value
    return obj


def extract(content):
    """Accept fences/prose around ONE object; reject ambiguous or truncated JSON."""
    if not isinstance(content, str):
        raise ValueError("content must be a string")
    decoder = json.JSONDecoder(object_pairs_hook=reject_duplicates)
    objects, position = [], 0
    while True:
        start = content.find("{", position)
        if start < 0:
            break
        # Do not salvage an inner object from malformed/truncated outer JSON.
        obj, end = decoder.raw_decode(content, start)
        objects.append(obj)
        position = end
    if len(objects) != 1:
        raise ValueError("expected exactly one JSON object")
    return objects[0]


def grade(content, expected):
    try:
        got, error = extract(content), None
    except (ValueError, TypeError) as exc:
        got, error = {}, str(exc)
    fields = {key: {"expected": value, "actual": got.get(key),
                    "correct": type(got.get(key)) is type(value) and got.get(key) == value}
              for key, value in expected.items()}
    score = sum(field["correct"] for field in fields.values())
    return {"score": score, "total": len(expected), "score_over_n": f"{score}/{len(expected)}",
            "fields": fields, "registry_all_correct": all(fields[k]["correct"] for k in REGISTRY_FIELDS),
            "schema_valid": not error and set(got) == set(FIELDS) and
                all(type(got.get(k)) is type(expected[k]) for k in FIELDS),
            "parse_error": error}


def verdict(runs):
    expected = {(target,repeat) for target in TARGETS for repeat in (1,2)}
    identities = [(r.get("target_input_tokens"),r.get("repeat")) for r in runs]
    if (len(runs) != len(expected) or set(identities) != expected
            or any(run.get("error") for run in runs)):
        return {"status": "ERROR", "rule": PASS_RULE, "reason": "six distinct successful transports required"}
    pairs = []
    for repeat in (1, 2):
        control = next(r for r in runs if r["target_input_tokens"] == TARGETS[0] and r["repeat"] == repeat)
        for target in TARGETS[1:]:
            long = next(r for r in runs if r["target_input_tokens"] == target and r["repeat"] == repeat)
            passed = (control["finish_reason"] == long["finish_reason"] == "stop" and
                      long["grade"]["score"] >= control["grade"]["score"]-1 and long["grade"]["registry_all_correct"])
            pairs.append({"repeat": repeat, "target_input_tokens": target,
                          "control_score": control["grade"]["score_over_n"],
                          "long_score": long["grade"]["score_over_n"], "passed": passed})
    return {"status": "PASS" if all(p["passed"] for p in pairs) else "FAIL", "rule": PASS_RULE, "pairs": pairs}


def self_check(expected):
    checks = []
    def check(name, content, score):
        result = grade(content, expected)
        if result["score"] != score:
            raise AssertionError(f"{name}: {result['score']} != {score}")
        checks.append({"name": name, "score_over_n": result["score_over_n"]})
    check("ground_truth", compact(expected), 10)
    check("fenced", "```json\n"+compact(expected)+"\n```", 10)
    check("surrounding_prose", "Result:\n"+compact(expected)+"\nDone.", 10)
    for field in FIELDS:
        corrupt = dict(expected)
        corrupt[field] = -1 if field == "matching_count" else "WRONG"
        check("corrupt_"+field, compact(corrupt), 9)
    for name, value in (("count_list", [expected["matching_count"]]), ("count_string", str(expected["matching_count"])),
                        ("count_bool", True), ("count_float", float(expected["matching_count"]))):
        check(name, compact(dict(expected, matching_count=value)), 9)
    missing = dict(expected)
    del missing["alias_result"]
    check("missing_field", compact(missing), 9)
    check("truncated", compact(expected)[:-1], 0)
    check("two_objects", compact(expected)+compact(expected), 0)
    check("duplicate_key", compact(expected)[:-1]+',"r1":"WRONG"}', 0)
    check("non_object", "[]", 0)
    check("invalid", "{broken}", 0)
    return checks


def prefill_timeout(tokens):
    return tokens / 300 * 1.5 + 30  # Conservative long-prefill timeout, including startup overhead.


def execute(endpoint, receipt):
    """Read SSE directly; retain partial content and bound prefill plus decode."""
    url = urlsplit(endpoint)
    if url.scheme not in ("http", "https") or not url.hostname or url.query or url.fragment or url.username:
        raise ValueError("endpoint must be an HTTP(S) base URL without credentials/query/fragment")
    path = url.path.rstrip("/")
    if not path.endswith("/chat/completions"):
        path += "/chat/completions" if path.endswith("/v1") else "/v1/chat/completions"
    prefill = prefill_timeout(receipt["input_tokens"])
    connection_type = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(url.hostname, url.port, timeout=5)
    start, content, first, finish, usage, done = time.monotonic(), "", None, None, {}, False
    deadline = start + prefill + 120
    result = {"target_input_tokens": receipt["target_input_tokens"], "input_tokens": receipt["input_tokens"],
              "prefill_timeout_seconds": prefill, "repeat": receipt["repeat"], "prompt_sha256": receipt["prompt_sha256"]}
    response = None
    try:
        connection.connect()
        sock = connection.sock
        sock.settimeout(prefill)
        connection.request("POST", path, body=compact(receipt["request"]).encode(), headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {response.read(4096).decode(errors='replace')}")
        while True:
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError("request deadline expired")
            sock.settimeout(min(remaining, prefill if first is None else 60))
            line = response.readline()
            if not line:
                break
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            chunk = json.loads(data)
            if "error" in chunk:
                raise RuntimeError(compact(chunk["error"]))
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                value = delta.get("content") or ""
                if value:
                    first = time.monotonic()-start if first is None else first
                    content += value
                finish = choice.get("finish_reason") or finish
            usage = chunk.get("usage") or usage
        if not done or finish is None:
            raise RuntimeError("incomplete SSE protocol (DONE/finish_reason missing)")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if response is not None:
            response.close()
        connection.close()
    result.update(content=content, finish_reason=finish, usage=usage, stream_done=done,
                  seconds=time.monotonic()-start, ttft_seconds=first, grade=grade(content, receipt["expected"]))
    if isinstance(usage.get("prompt_tokens"), int):
        result["server_minus_local_prompt_tokens"] = usage["prompt_tokens"]-receipt["input_tokens"]
    return result


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2)+"\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true")
    mode.add_argument("--execute", action="store_true")
    ap.add_argument("--endpoint", help="existing chat-completions HTTP(S) base URL")
    ap.add_argument("--out", type=Path, help="new artifact directory (default: timestamped beside script)")
    ap.add_argument("--tokenizer-dir", type=Path, required=True,
                    help="checkpoint directory containing tokenizer.json and chat_template.jinja")
    ap.add_argument("--model", default="GLM-5.3")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--boot-label", default="operator-managed same boot (not independently verified)")
    args = ap.parse_args()
    if args.execute and not args.endpoint:
        ap.error("--execute requires --endpoint")
    if args.offline and args.endpoint:
        ap.error("--endpoint is only allowed with --execute")
    out = args.out or Path(__file__).parent / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ("-offline" if args.offline else "-execute"))
    out.mkdir(parents=True, exist_ok=False)
    tokens, receipts, checks = LocalTokens(args.tokenizer_dir), {}, {}
    for target in TARGETS:
        receipt = generate(target, args.seed, tokens)
        receipt["request"]["model"] = args.model
        checks[str(target)] = self_check(receipt["expected"])
        # Repeat regeneration verifies deterministic serialized prompts and truth.
        regenerated = generate(target, args.seed, tokens)
        assert receipt["prompt_sha256"] == regenerated["prompt_sha256"]
        assert receipt["expected"] == regenerated["expected"]
        receipts[target] = receipt
        save(out/f"panel-{target}.json", receipt)
        (out/f"prompt-{target}.txt").write_text(receipt["request"]["messages"][0]["content"]+"\n")
        print(f"target={target} input={receipt['input_tokens']} rows={receipt['rows']} expected_answer={receipt['expected_answer_tokens']} tokens scorer_checks={len(checks[str(target)])} PASS", flush=True)
    # Test the predetermined comparative rule at its boundary, including registry guard.
    perfect = [dict(target_input_tokens=t, repeat=r, finish_reason="stop", grade=grade(compact(receipts[t]["expected"]), receipts[t]["expected"]))
               for t in TARGETS for r in (1, 2)]
    assert verdict(perfect)["status"] == "PASS"
    trial = json.loads(json.dumps(perfect))
    trial[2]["grade"]["score"], trial[2]["grade"]["score_over_n"] = 9, "9/10"
    assert verdict(trial)["status"] == "PASS"
    trial[2]["grade"]["score"] = 8
    assert verdict(trial)["status"] == "FAIL"
    trial[2]["grade"]["score"] = 9
    trial[2]["grade"]["registry_all_correct"] = False
    assert verdict(trial)["status"] == "FAIL"
    trial = json.loads(json.dumps(perfect))
    trial[3]["finish_reason"] = "length"
    assert verdict(trial)["status"] == "FAIL"
    trial[3]["error"] = "transport failure"
    assert verdict(trial)["status"] == "ERROR"
    save(out/"self-check.json", {"status": "PASS", "scorer_checks": checks, "verdict_checks": 6, "rule": PASS_RULE})
    summary = {"mode": "offline" if args.offline else "execute", "rule": PASS_RULE,
               "boot_label": args.boot_label, "panels": [{k: v for k, v in r.items() if k != "request"} for r in receipts.values()]}
    if args.execute:
        runs = []
        # Identical requests twice per length; second run may have APC warmth.
        for target in TARGETS:
            for repeat in (1, 2):
                run = execute(args.endpoint, dict(receipts[target], repeat=repeat))
                runs.append(run)
                save(out/f"run-{target}-{repeat}.json", run)
                print(f"target={target} repeat={repeat} score={run['grade']['score_over_n']} registry={run['grade']['registry_all_correct']} finish={run['finish_reason']} error={run.get('error')}", flush=True)
        summary.update(runs=runs, verdict=verdict(runs))
    else:
        summary["verdict"] = {"status": "OFFLINE-SELF-CHECK-PASS", "quality_verdict": "NOT RUN"}
    save(out/"summary.json", summary)
    print(f"{summary['verdict']['status']} artifacts={out.resolve()}", flush=True)
    return 0 if args.offline or summary["verdict"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
