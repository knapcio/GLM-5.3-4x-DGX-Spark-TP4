#!/usr/bin/env python3
"""Check the extracted image source without importing vLLM or CUDA."""
import argparse
import hashlib
import json
from pathlib import Path


def check_pins(root, pin_file):
    pins = json.loads(pin_file.read_text())
    for name, expected in pins.items():
        path = root / (name.replace('.', '/') + '.py')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Source drift: ' + str(path))
    print('SOURCE PINS PASS: %d files (%s)' % (len(pins), pin_file.name))


def check(root):
    overlay = Path(__file__).resolve().parents[1] / 'overlay'
    for pin_file in (overlay / 'source_pins.json', overlay / 'kstop/compat_source_pins.json'):
        check_pins(root, pin_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='parent directory of vllm/')
    check(parser.parse_args().source)
