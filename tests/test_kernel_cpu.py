"""Execute the transformed prepare-input kernel as NumPy scalar/vector code.

This tests the image kernel's indexing/slot writes. It does not compile Triton
or run attention. Masks, page boundaries, rejection and graph padding use the
actual transformed function body, not a copy of its arithmetic.
"""
import ast
import os
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SRC=Path(os.environ.get('GLM_IMAGE_SRC','/image-source'))
sys.path.insert(0,str(ROOT/'overlay/swa-pool'))
import glm_dsa_swa_pool as G


class T(np.ndarray):
    def to(self,dtype): return self.astype(dtype).view(T)


def tensor(value): return np.asarray(value).view(T)


class Ptr:
    def __init__(self,data,offset=0): self.data,self.offset=data,offset
    def __add__(self,offset): return Ptr(self.data,self.offset+offset)
    def __sub__(self,offset): return Ptr(self.data,self.offset-offset)


class TL:
    int32=np.int32
    int64=np.int64
    def program_id(self,axis): return 0
    def num_programs(self,axis): return 1
    def arange(self,a,b): return tensor(np.arange(a,b))
    def minimum(self,a,b): return tensor(np.minimum(a,b))
    def where(self,c,a,b): return tensor(np.where(c,a,b))
    def load(self,p,mask=True,other=0):
        idx,mask=np.broadcast_arrays(p.offset,mask)
        out=np.full(idx.shape,other,dtype=p.data.dtype)
        out[mask.astype(bool)]=p.data[idx[mask.astype(bool)].astype(int)]
        return tensor(out)
    def store(self,p,value,mask=True):
        idx,value,mask=np.broadcast_arrays(p.offset,value,mask)
        mask=mask.astype(bool)
        p.data[idx[mask].astype(int)]=value[mask]


def kernel():
    name=G.D_NAME
    source=G.transform_source(name,(SRC/(name.replace('.','/')+'.py')).read_text())
    tree=ast.parse(source)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_prepare_dflash_inputs_kernel')
    fn.decorator_list=[]
    for a in fn.args.args: a.annotation=None
    fn.returns=None
    tree=ast.Module(body=[fn],type_ignores=[]); ast.fix_missing_locations(tree)
    namespace={'tl':TL()}
    exec(compile(tree,'<pinned-prepare-kernel>','exec'),namespace)
    return namespace[fn.name],fn


class KernelTests(unittest.TestCase):
    def test_boundary_rejection_null_and_padding(self):
        fn,node=kernel()
        for block_size in (16,64):
            for anchor in (True,False):
                for start,nctx,rejected,maxlen in ((2040,32,7,32768),(31,34,0,32768),(32740,28,1,32768)):
                    nquery=8 if anchor else 9
                    # Deliberately leave the first context page as a null hole.
                    table=np.arange(1,2049,dtype=np.int64)
                    table[start//block_size]=0
                    args={a.arg: Ptr(np.full(512,-999,dtype=np.int64))
                          for a in node.args.args if a.arg.endswith('_ptr')}
                    def port(name,values): args[name]=Ptr(np.asarray(values))
                    port('target_positions_ptr',np.arange(start,start+nctx,dtype=np.int64))
                    port('target_query_start_loc_ptr',[0,nctx])
                    port('idx_mapping_ptr',[0])
                    port('num_rejected_ptr',[rejected])
                    port('num_sampled_ptr',[1])
                    port('last_sampled_ptr',[100])
                    port('next_prefill_tokens_ptr',[101])
                    port('temperature_ptr',[1])
                    port('seeds_ptr',[42])
                    port('block_table_ptr',table)
                    args.update(block_table_stride=len(table),parallel_drafting_token_id=154856,
                        block_size=block_size,num_query_per_req=nquery,num_speculative_steps=8,
                        max_num_reqs=4,max_num_tokens=128,max_model_len=maxlen,
                        SAMPLE_FROM_ANCHOR=anchor,PAD_SLOT_ID=-1,BLOCK_SIZE=128)
                    fn(**args)
                    pos=np.arange(start,start+nctx)
                    expected=np.where((table[pos//block_size]!=0)&(np.arange(nctx)<nctx-rejected),
                        table[pos//block_size]*block_size+pos%block_size,-1)
                    np.testing.assert_array_equal(args['out_context_slot_mapping_ptr'].data[:nctx],expected)
                    qp=np.arange(start+nctx-rejected,start+nctx-rejected+nquery)
                    qids=table[np.minimum(qp//block_size,len(table)-1)]
                    query_expected=np.where((qids!=0)&(qp<maxlen),
                        qids*block_size+qp%block_size,-1)
                    np.testing.assert_array_equal(args['out_query_slot_mapping_ptr'].data[:nquery],query_expected)
                    self.assertTrue(np.all(args['out_query_slot_mapping_ptr'].data[nquery:128]==-1))
                    np.testing.assert_array_equal(args['out_sample_idx_mapping_ptr'].data[8:32],-1)
                    self.assertFalse(any(0<=x<block_size for x in expected))


if __name__=='__main__': unittest.main(verbosity=2)
