# SPDX-License-Identifier: Apache-2.0
"""TP4 native MTP experts, three existing Marlin GEMMs, independent globals.

The stock NVFP4 MoE method silently uses gate's global for up. Split gate/up
instead, exactly like attention's independent A projections. No new kernel.
"""
import glm_nvfp4_format as fmt


def prepare_experts(layer, prepare=None):
    import torch
    if prepare is None:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_nvfp4_moe_layer_for_marlin
        prepare = prepare_nvfp4_moe_layer_for_marlin
    n = layer.intermediate_size_per_partition
    k = layer.hidden_size
    for i in range(layer.num_experts):
        for offset,j in ((0,0),(n,1)):
            fmt.validate_tensors(dict(weight_packed=layer.w13_weight_packed[i,offset:offset+n],
                weight_scale=layer.w13_weight_scale[i,offset:offset+n],
                weight_global_scale=layer.w13_weight_global_scale[i,j].reshape(1)),[n,k])
        fmt.validate_tensors(dict(weight_packed=layer.w2_weight_packed[i],weight_scale=layer.w2_weight_scale[i],
            weight_global_scale=layer.w2_weight_global_scale[i].reshape(1)),[k,n])
    parts=[]
    for j in range(2):
        parts.append(prepare(layer,
            layer.w13_weight_packed[:,j*n:(j+1)*n].contiguous(),
            layer.w13_weight_scale[:,j*n:(j+1)*n].contiguous(),
            layer.w13_weight_global_scale[:,j].contiguous(),
            layer.w2_weight_packed,layer.w2_weight_scale,layer.w2_weight_global_scale,
            is_act_and_mul=False))
    # prepare() is pure with respect to input weights/scales; down is identical in both calls.
    for name,value in zip(('gate','gate_scale','gate_global','down','down_scale','down_global'),parts[0]):
        layer.register_parameter('_glm_'+name,torch.nn.Parameter(value,requires_grad=False))
    for name,value in zip(('up','up_scale','up_global'),parts[1][:3]):
        layer.register_parameter('_glm_'+name,torch.nn.Parameter(value,requires_grad=False))
    for name in ('w13_weight_packed','w13_weight_scale','w13_weight_global_scale',
                 'w2_weight_packed','w2_weight_scale','w2_weight_global_scale',
                 'w13_input_global_scale','w2_input_global_scale'):
        delattr(layer,name)


def apply_experts(layer,x,topk_weights,topk_ids,ops=None,align=None,scalar=None):
    import torch
    if x.dtype != torch.bfloat16 or topk_weights.dtype != torch.float32:
        raise ValueError('MTP requires BF16 activations and FP32 router weights')
    if ops is None:
        from vllm import _custom_ops as ops
    if align is None:
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
        align = moe_align_block_size
    if scalar is None:
        from vllm.scalar_type import scalar_types
        scalar = scalar_types.float4_e2m1f
    m,k=x.shape
    n=layer.intermediate_size_per_partition
    topk=topk_ids.shape[1]
    block=8
    for block in (8,16,32,48,64):
        if m*topk/layer.num_experts/block < .9:
            break
    sorted_ids,expert_ids,padded=align(topk_ids,block,layer.global_num_experts,
                                    layer.expert_map,ignore_invalid_experts=True)
    def gemm(inp,which,size_m,size_n,size_k,num_topk,mul):
        output=torch.empty((m*topk,size_n),device=x.device,dtype=x.dtype)
        return ops.moe_wna16_marlin_gemm(inp,output,getattr(layer,'_glm_'+which),None,
            getattr(layer,'_glm_'+which+'_scale'),None,getattr(layer,'_glm_'+which+'_global'),
            None,None,None,layer.workspace,sorted_ids,expert_ids,padded,topk_weights,
            moe_block_size=block,top_k=num_topk,mul_topk_weights=mul,b_q_type=scalar,
            size_m=size_m,size_n=size_n,size_k=size_k,is_k_full=True,
            use_atomic_add=False,use_fp32_reduce=True,is_zp_float=False)
    gate=gemm(x,'gate',m,n,k,topk,layer.apply_router_weight_on_input)
    up=gemm(x,'up',m,n,k,topk,layer.apply_router_weight_on_input)
    activated=torch.empty_like(gate)
    # Same pinned SiLU-and-mul kernel as the stock Marlin path.
    # The pinned vllm._custom_ops has no silu_and_mul wrapper; the stock path calls the _C op directly.
    silu_and_mul = getattr(ops,'silu_and_mul',None) or torch.ops._C.silu_and_mul
    silu_and_mul(activated,torch.cat((gate,up),dim=-1))
    output=gemm(activated,'down',m*topk,k,n,1,not layer.apply_router_weight_on_input)
    return torch.sum(output.view(m,topk,k),dim=1)


def new_method(moe):
    import torch
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4 import CompressedTensorsW4A4Nvfp4MoEMethod
    class NativeMTPNVFP4(CompressedTensorsW4A4Nvfp4MoEMethod):
        # kstop pad hygiene: topk ids are read after the runner's dead-row remap (apply_experts below).
        glm_marlin_topk_consumer = True
        def __init__(self):
            super().__init__(moe,use_a16=True)
            if self.nvfp4_backend.value != 'MARLIN':
                raise ValueError('MTP requires the pinned Marlin W4A16 backend')
            if moe.tp_size != 4 or moe.ep_size != 1 or not moe.is_act_and_mul:
                raise ValueError('qualified MTP layout requires TP4, EP1, SiLU gate/up')
        def create_weights(self,layer,*args,**kwargs):
            super().create_weights(layer,*args,**kwargs)
            # No activation scales are read or emitted in W4A16 mode.
            layer.w13_input_global_scale.data.fill_(1)
            layer.w2_input_global_scale.data.fill_(1)
        def process_weights_after_loading(self,layer):
            from vllm.model_executor.layers.fused_moe.activation import MoEActivation
            if layer.activation != MoEActivation.SILU or getattr(layer,'swiglu_limit',None) is not None:
                raise ValueError('MTP supports the pinned unclamped SiLU activation only')
            prepare_experts(layer)
        def get_fused_moe_quant_config(self,layer):
            return None
        @property
        def is_monolithic(self):
            return False
        def apply(self,layer,x,topk_weights,topk_ids,shared_experts,shared_experts_input):
            # Runner owns shared-expert execution because supports_internal_mk=False.
            return apply_experts(layer,x,topk_weights,topk_ids)
    return NativeMTPNVFP4()
