# SPDX-License-Identifier: Apache-2.0
import base64
import importlib.util
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import boot_preflight as B
import checkpoint_headers as H


class Headers(unittest.TestCase):
    def shard(self,root,header=None,payload=b'\0'*8,name='a.safetensors'):
        h=header or {'x':{'dtype':'F32','shape':[2],'data_offsets':[0,8]}}
        raw=json.dumps(h).encode();p=root/name;p.write_bytes(struct.pack('<Q',len(raw))+raw+payload);return p
    def test_reads_header_only(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.shard(Path(d));opened=p.open;reads=[]
            class Reader:
                def __enter__(self):self.f=opened('rb');return self
                def __exit__(self,*a):self.f.close()
                def read(self,n):reads.append(n);return self.f.read(n)
            with patch.object(Path,'open',lambda *a,**k:Reader()):h=H.read_header(p)
            self.assertEqual(reads,[8,h['header_bytes']])
    def test_corrupt_headers(self):
        for t in [dict(dtype='BF16',shape=[2],data_offsets=[0,8]),dict(dtype='XX',shape=[2],data_offsets=[0,8]),
                  dict(dtype='F32',shape=[-2],data_offsets=[0,8]),dict(dtype='F32',shape=[2],data_offsets=[1,9])]:
            with self.subTest(t=t),self.assertRaises(ValueError):H.validate({'x':t},8)
    def test_overlap_gap(self):
        with self.assertRaises(ValueError):H.validate({'x':dict(dtype='U8',shape=[4],data_offsets=[1,5])},5)
    def test_export_roundtrip_and_index_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.shard(root)
            (root/'config.json').write_text('{}')
            (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'x':'a.safetensors'}}))
            b=H.export(root);self.assertEqual(H.unpack(b)[1]['x']['shape'],[2])
            b['documents']['model.safetensors.index.json']=base64.b64encode(b'{"weight_map":{"y":"a.safetensors"}}').decode()
            with self.assertRaisesRegex(ValueError,'inventory'):H.unpack(b)
    def test_cache_digest_and_duplicate_keys(self):
        with self.assertRaises(ValueError):json.loads('{"x":1,"x":2}',object_pairs_hook=H.unique)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=self.shard(root);h=H.read_header(p);h['sha256']='bad'
            with self.assertRaisesRegex(ValueError,'digest'):H.unpack(dict(documents={},files={'a':h}))
    def test_sidecar_repeated_leaf_names(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.shard(root,name='a.safetensors');self.shard(root,name='b.safetensors')
            (root/'manifest.json').write_text(json.dumps({'tensors':{'a':{'file':'a.safetensors'},'b':{'file':'b.safetensors'}}}))
            self.assertEqual(set(H.unpack(H.export(root,True))[1]),{'a.safetensors/x','b.safetensors/x'})
    def test_unsafe_shard(self):
        with tempfile.TemporaryDirectory() as d,self.assertRaises(ValueError):H.safe_file(Path(d),'../escape')


class SidecarContracts(unittest.TestCase):
    def test_names_shapes_dtypes_source_and_companions(self):
        B.overlay_paths(ROOT)
        import glm_nvfp4_groups as groups,glm_nvfp4_format as fmt
        prefix='model.layers.3.mlp.shared_experts.gate_proj'
        source={'config.json':b'{}','model.safetensors.index.json':b'{"weight_map":{}}'}
        def data():
            hs={};offset=0
            for name,dtype,shape in [('weight_packed','U8',[128,64]),('weight_scale','F8_E4M3',[128,8]),('weight_global_scale','F32',[1])]:
                size=__import__('math').prod(shape)*H.DTYPE_BYTES[dtype]
                hs[name]=dict(dtype=dtype,shape=shape,data_offsets=[offset,offset+size]);offset+=size
            raw=json.dumps(hs).encode()
            manifest=dict(schema=2,algorithm=fmt.ALGORITHM,complete=True,groups=['shared'],
                source={'config_sha256':H.sha(source['config.json']),'index_sha256':H.sha(source['model.safetensors.index.json'])},
                tensors={prefix:dict(group='shared',shape=[128,128],file='w.safetensors',leaves=['weight_packed','weight_scale','weight_shape'])})
            cache=dict(documents={'manifest.json':base64.b64encode(json.dumps(manifest).encode()).decode()},
                files={'w.safetensors':dict(raw_b64=base64.b64encode(raw).decode(),sha256=H.sha(raw),size=8+len(raw)+offset,header_bytes=len(raw))})
            return dict(docs=source.copy(),tensors={prefix+'.'+n:{} for n in manifest['tensors'][prefix]['leaves']},cache=dict(sidecars=[cache]))
        with patch.dict(__import__('os').environ,{'GLM_ATTN_WEIGHTS':'nvfp4','GLM_NVFP4_GROUPS':'shared'}),patch.dict(groups.COUNTS,{'shared':1}):
            result=B.effective_headers(data())
            self.assertEqual(result[prefix+'.weight_global_scale']['dtype'],'F32')
            broken=data();broken['tensors'].pop(prefix+'.weight_shape')
            with self.assertRaisesRegex(ValueError,'companion'):B.effective_headers(broken)
            broken=data();broken['docs']['config.json']=b'{"changed":true}'
            with self.assertRaisesRegex(ValueError,'source'):B.effective_headers(broken)
            broken=data();receipt=broken['cache']['sidecars'][0]['files']['w.safetensors']
            hs=json.loads(base64.b64decode(receipt['raw_b64']));hs['weight_packed']['shape']=[64,128]
            raw=json.dumps(hs).encode();receipt.update(raw_b64=base64.b64encode(raw).decode(),sha256=H.sha(raw),header_bytes=len(raw),size=8+len(raw)+9220)
            with self.assertRaisesRegex(ValueError,'shape/dtype'):B.effective_headers(broken)


class Vectors(unittest.TestCase):
    def test_same_env_and_vector(self):
        v=B.vectors('docker run --rm --entrypoint python img -c foo\n' +
            "docker run -d -v /mac:/model:ro -e GLM_NVFP4_GROUPS=attn,shared,dense,mtp -e GLM_PAD_HYGIENE=1 img serve /model --node-rank 3 --speculative-config '{\"method\":\"mtp\"}'")[0]
        self.assertEqual(v['env']['GLM_PAD_HYGIENE'],'1');self.assertEqual(v['rank'],3)
        self.assertEqual(v['mounts'],{'/model':'/mac'});self.assertEqual(v['args'][-1],'{"method":"mtp"}')
    def test_no_vectors_and_duplicate(self):
        for t in ['','docker run img serve /model\ndocker run img serve /model']:
            with self.assertRaises(ValueError):B.vectors(t)
    def test_report_does_not_hide_failures(self):
        r=B.Report()
        with patch('sys.stdout',io.StringIO()):
            r.check('bad',lambda:1/0);r.check('good',lambda:'ok')
        self.assertFalse(r.result()['ok']);self.assertIn('ZeroDivisionError',r.rows[0]['error'])
    def test_argparse_fail_is_reported(self):
        r=B.Report()
        with patch('sys.stdout',io.StringIO()):r.check('bad',lambda:sys.exit(2))
        self.assertFalse(r.result()['ok'])

if __name__=='__main__':unittest.main()
