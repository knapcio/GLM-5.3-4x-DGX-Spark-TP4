# SPDX-License-Identifier: Apache-2.0
"""One target-step metadata staging launch; stable buffers for graph replay."""
import triton
import triton.language as tl


@triton.jit
def _stage_rows(QSL, IDX, POS, RING, OUTPOS, LOWER, FLAG,
                T: tl.constexpr, R: tl.constexpr, CAP: tl.constexpr,
                BT: tl.constexpr=128):
    t=tl.program_id(0)*BT+tl.arange(0,BT)
    index=tl.full((BT,),-1,tl.int32)
    pos=tl.load(POS+t,mask=t<T,other=-1).to(tl.int32)
    window=tl.load(FLAG)
    for req in range(R):
        begin=tl.load(QSL+req); end=tl.load(QSL+req+1)
        owner=tl.load(IDX+req).to(tl.int32)
        valid_owner=(owner>=0)&(owner<4)
        tl.device_assert((begin==end)|valid_owner,'recent KV request-state index exceeds four slots')
        if tl.program_id(0)==0 and begin<end and valid_owner:
            first=tl.load(POS+begin).to(tl.int32)
            tl.store(LOWER+owner,tl.maximum(0,first-window+1))
        active=(t>=begin)&(t<end)&(t<T)&valid_owner
        index=tl.where(active,owner*CAP+pos%CAP,index)
    tl.store(RING+t,index,mask=t<T)
    tl.store(OUTPOS+t,pos,mask=t<T)


def stage_rows(batch,bank,capacity):
    t=batch.num_tokens_after_padding
    if t:
        _stage_rows[(triton.cdiv(t,128),)](batch.query_start_loc,batch.idx_mapping,
            batch.positions,bank['token_ring'],bank['token_pos'],bank['lower'],bank['flag'],
            t,batch.num_reqs,capacity,num_warps=4,num_stages=1,debug=True)
