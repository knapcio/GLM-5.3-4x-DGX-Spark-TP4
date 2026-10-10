#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit a tracked release snapshot. Findings identify locations, never secret bytes."""
import argparse
import collections
import ipaddress
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    'home_path': r'/(?:Users|home)/[^\s"\x27<>]+',
    # Match both concrete hostnames and formatted prefixes such as ``host_0{rank}``.
    'site_hostname': r'(?i)\b(?:spark[-_]?0|nuc-agent)\b',
    'site_subnet': r'(?<![\w.])10\.100\.(?:96|97)(?:\.(?:[0-9]{1,3}|x))?(?![\w.])',
    'tailnet': r'(?i)[\w.-]+\.ts\.net|tail[0-9a-f]{4,}',
    'email': r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}',
    'authorship_or_session': r'(?i)\b(?:\x63\x6c\x61\x75\x64\x65|\x63\x68\x61\x74\x67\x70\x74|\x67\x70\x74-[0-9][\w.-]*|\x63\x6f\x64\x65\x78|\x61\x73\x74\x72\x61|\x6f\x70\x75\x73)\b|https?://[^\s"\x27]*(?:session|\x63\x68\x61\x74\x67\x70\x74\.com|\x63\x6c\x61\x75\x64\x65\.ai)[^\s"\x27]*',
    'key_material': r'-----BEGIN (?:[A-Z ]+)?PRIVATE KEY-----|\b(?:ghp_|github_pat_|sk-[A-Za-z0-9]|tskey-)[A-Za-z0-9_-]+',
}
IP = re.compile(r'(?<![\w.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![\w.])')


def audit(root):
    tracked = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).decode().split('\0')
    findings=[]; allowed=collections.Counter(); files=0
    for name in tracked:
        path=root/name
        if not path.is_file(): continue
        try: lines=path.read_text().splitlines()
        except UnicodeError: continue
        files+=1
        for number,line in enumerate(lines,1):
            # The scanner's own pattern definitions are not export contents.
            if name == 'scripts/public_export_audit.py': continue
            for kind,pattern in PATTERNS.items():
                for match in re.finditer(pattern,line):
                    if kind=='email' and match[0]=='knapcio@gmail.com':
                        allowed['owner_email']+=1;continue
                    findings.append(dict(file=name,line=number,kind=kind))
            for match in IP.finditer(line):
                try: ip=ipaddress.IPv4Address(match[0])
                except ipaddress.AddressValueError:
                    allowed['invalid_ipv4_test_fixture']+=1;continue
                if ip.is_loopback or ip in ipaddress.IPv4Network('192.0.2.0/24'):
                    allowed['loopback_or_TEST_NET_fixture']+=1
                else: findings.append(dict(file=name,line=number,kind='site_ip'))
    return dict(scope='tracked tip snapshot; git history not rewritten',text_files=files,
                findings=findings,allowed=dict(allowed),passed=not findings)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path)
    a=p.parse_args();result=audit(ROOT)
    text=json.dumps(result,indent=2)+'\n'
    if a.out:a.out.write_text(text)
    print(text,end='')
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
