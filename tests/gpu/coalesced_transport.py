"""Future pinned-image transport gate; run with no serving workers, before one candidate boot."""
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests'))
from test_coalesced_load import C, F, fixtures, keys, manifest, safe_open, stock, torch


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; this gate cannot pass on the Mac')
    os.environ.update(GLM_COALESCED_BATCH_MB='1', GLM_COALESCED_OWNED_MB='8',
                      GLM_COALESCED_THREADS='4', GLM_COALESCED_DIRECT='1')
    fill, prefetch = C.RangeReader.fill, C.prefetch
    pending = [0]
    def delayed(self, batch, output):
        torch.cuda._sleep(50_000_000)
        return fill(self, batch, output)
    def observed(iterator, cuda):
        for batch, event, output in prefetch(iterator, cuda):
            if event is not None and not event.query(): pending[0] += 1
            yield batch, event, output
    with tempfile.TemporaryDirectory() as d:
        files = fixtures(Path(d), count=132)
        reference = manifest(stock(files))
        st = F._Stats()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), patch.object(C.RangeReader, 'fill', delayed), patch.object(C, 'prefetch', observed):
            iterator = C.coalesced_safetensors_iterator(files, keys, lambda p: safe_open(p, framework='pt'), stats=st)
            actual = manifest((name, tensor.cpu()) for name, tensor in iterator)
        stream.synchronize()
        assert reference == actual
        assert pending[0] > 0, 'gate did not exercise a pending upload event'
        assert st.coalesced['cuda'] and st.coalesced['direct'] and st.coalesced['complete']
        F.release(st, 'coalesced-transport-gate', wait_s=0)
        print(f'CUDA transport hash-EQUAL: {len(actual)} tensors; {pending[0]} pending event handoffs')


if __name__ == '__main__': main()
