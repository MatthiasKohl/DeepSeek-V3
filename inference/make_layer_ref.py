# similar to generate.py but outputs reference values for a specific layer

import os
import json
from argparse import ArgumentParser
from typing import List
from functools import partial
import numpy
import re

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs, RMSNorm, MLA, MLP, Expert, Gate, MoE,\
    Linear, ColumnParallelLinear, RowParallelLinear,\
    apply_rotary_emb, weight_dequant, block_size

print0 = print
def print_seq(group: dist.ProcessGroup, *args, **kwargs):
    for rank in range(dist.get_world_size(group=group)):
        dist.barrier(group=group)
        if dist.get_rank(group=group) == rank:
            print(*args, **kwargs)

# most prompts from Google top100 (https://ahrefs.com/blog/top-google-questions/)
PROMPTS = [
    "The color of the sky is blue but sometimes it can also be",
    """\
apple is pomme,
bannana is banane,
cherry is""",
    "1, 2, 3, 5, 8, 13",
    "ba ba black sheep, have you any wool?",
    "what is my ip?",
    "what to watch?",
    "what dinosaur has 500 teeth?",
    "who won the election 2024?",
    "where is my train?",
    "who won the election?",
    "what the font?",
    "how many weeks in a year?",
    "how to vote lok sabha?",
    "when is mother's day 2024?",
    "que significa?",
    "who called me?",
    "where to watch india national cricket team vs england cricket team?",
    "quando eh o prox carnaval?",
    "cuando cobro?",
    "how to screenshot on mac?",
    "what is my ip address?",
    "who is winning the election?",
    "when is easter?",
    "when the phone rings?",
    "when is father's day 2024?",
    "how to tie a tie?",
    "what time is it?",
    "where to watch australian men's cricket team vs india national cricket team?",
    "how to screenshot on windows?",
    "when is easter 2024?",
    "how many days until christmas?",
    "cuando juega boca?",
    # "how many ounces in a cup?",
    # "who is winning the election 2024?",
    # "how to delete instagram account?",
    # "how to vote in the us?",
    # "what is project 2025?",
    # "why were chainsaws invented?",
    # "what we do in the shadows?",
    # "where to watch india national cricket team vs bangladesh national cricket team?",
    # "when is father's day?",
    # "qué significa?",
    # "when is mothers day?",
    # "how many seconds in a day?",
    # "who won the super bowl 2024?",
    # "how to make money online?",
    # "what is today?",
    # "какой сегодня праздник?",
    # "what if?",
    # "what is love?",
    # "how many ounces in a gallon?",
    # "what is ai?",
    # "where am i?",
    # "where to watch olympics 2024?",
    # "when is the super bowl?",
    # "what time is the eclipse?",
    # "cuando juega river?",
    # "how are you?",
    # "how to lose a guy in 10 days?",
    # "what time is the super bowl?",
    # "when is black friday?",
    # "cuando juega argentina?",
    # "what is the?",
    # "how to delete facebook account?",
    # "what day is it?",
    # "how to draw?",
    # "where to watch zimbabwe national cricket team vs india national cricket team?",
    # "where to watch india national cricket team vs new zealand national cricket team?",
    # "how to train your dragon?",
    # "when is the solar eclipse?",
    # "how many ounces in a pound?",
    # "when is thanksgiving?",
    # "cuando es el dia del padre?",
    # "what beats rock?",
    # "how to pronounce?",
    # "why is my husband yelling at me?",
    # "cual es mi ip?",
    # "how to deactivate facebook?",
    # "how to calculate percentage?",
    # "when is fathers day?",
    # "where to watch south africa national cricket team vs india national cricket team?",
    # "when is the super bowl 2024?",
    # "how to lose weight fast?",
    # "how many days in a year?",
    # "how many days till christmas?",
    # "que hora es?",
    # "how many countries in the world?",
    # "when is the next full moon?",
    # "how it ends?",
    # "quien juega hoy?",
    # "how to register to vote?",
    # "when is black friday 2024?",
    # "quien gano las elecciones en venezuela 2024?",
    # "what is juneteenth?",
    # "how old is taylor swift?",
    # "how many people are in the world?",
    # "who won the tyson paul fight?",
    # "what time is the debate tonight?",
    # "where does kiwi fruit come from?",
    # "how to lose weight?",
    # "que pasa salta?",
    # "how to make money?",
    # "when is thanksgiving 2024?",
    # "when is ramadan 2024?",
    # "When does wicked come out on streaming?",
    # "When did squid game 1 come out?",
    # "Where has jd vance been?",
    # "Why is jim harbaugh limping?",
    # "What does apt mean rosé?",
    # "What Does Pookie Mean?",
    # "How Many Steps In A Mile?",
    # "What Is Gaslighting?",
    # "How To Say Hi In Spanish?",
    # "What Is Temu?",
    # "how many square feet in an acre?",
    # "what is a verb?",
    # "do you want to build a snowman?",
    # "how to cook rice?",
    # "how to get rid of fruit flies?",
    # "how to hard boil eggs?",
    # "how to make cake?",
    # "how many ounces in a pint?",
    # "how to hack wifi passwords?",
    # "how old are you?",
    # "how to start a blog?",
    # "how many ounces are in a gallon?",
    # "how to backup iphone?",
    # "how old is dolly parton?",
]

def sample(logits, temperature: float = 1.0):
    """
    Samples a token from the logits using temperature scaling.

    Args:
        logits (torch.Tensor): The logits tensor for token predictions.
        temperature (float, optional): Temperature for scaling logits. Defaults to 1.0.

    Returns:
        torch.Tensor: The sampled token.
    """
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)

def _gather(input_: torch.Tensor, dim: int = 0, group: dist.ProcessGroup = None) -> torch.Tensor:
    # Bypass the function if we are using only 1 GPU.
    if dist.get_world_size(group=group) == 1:
        return input_

    # Size and dimension.
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)

    input_ct = input_.contiguous()
    tensor_list = [torch.empty_like(input_ct) for _ in range(world_size)]
    tensor_list[rank] = input_ct
    dist.all_gather(tensor_list, input_ct, group=group)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output

def _gather_scale(weight_: torch.Tensor, dim: int = 0, group: dist.ProcessGroup = None):
    if getattr(weight_, "scale", None) is None:
        return None
    return _gather(weight_.scale.detach(), dim=dim, group=group)

def _copy_out(m: torch.nn.Module, base_name: str, out: dict, x: torch.Tensor, is_gate: bool = False, do_gather: bool = True, group: dist.ProcessGroup = None):
    h = m(x.detach())
    if is_gate:
        # gate output is the same for all ranks
        out[base_name + "OutActW"] = h[0]
        out[base_name + "OutActIdx"] = h[1]
    elif do_gather:
        out[base_name + "OutAct"] = _gather(h.unsqueeze(0), group=group)
    else:
        out[base_name + "OutAct"] = h

def _copy_linear(m: Linear, base_name: str, out: dict, x: torch.Tensor = None, gather_out: bool = True, group: dist.ProcessGroup = None):
    if isinstance(m, RowParallelLinear):
        gather_dim = 1
    else:
        gather_dim = 0
    if not isinstance(m, RowParallelLinear) and not isinstance(m, ColumnParallelLinear):
        w = m.weight.unsqueeze(0)
    else:
        w = m.weight
    
    out[base_name + "W"] = _gather(w.detach(), dim=gather_dim, group=group)
    out[base_name + "Wscale"] = _gather_scale(w, dim=gather_dim, group=group)
    if x is not None:
        _copy_out(m, base_name, out, x, do_gather=gather_out, group=group)

def _copy_norm(m: RMSNorm, base_name: str, out: dict, x: torch.Tensor = None, do_gather: bool = False, group: dist.ProcessGroup = None):
    # norm weights are the same for all ranks
    if do_gather:
        out[base_name + "W"] = _gather(m.weight.detach().unsqueeze(0), group=group)
    else:
        out[base_name + "W"] = m.weight.detach()
    if x is not None:
        _copy_out(m, base_name, out, x, do_gather=do_gather, group=group)

def _copy_mla(m: MLA, base_name: str, out: dict, inputs = None, group: dist.ProcessGroup = None):
    # note: assume that q_lora_rank != 0 and attn_impl == "absorb"
    # always save the Kv cache first
    out[base_name + "KvCache"] = _gather(m.kv_cache.detach().unsqueeze(0), group=group)
    out[base_name + "PeCache"] = _gather(m.pe_cache.detach().unsqueeze(0), group=group)

    if inputs is None:
        _copy_linear(m.wq_a, base_name + "Qa", out, group=group)
        _copy_norm(m.q_norm, base_name + "NormQ", out, do_gather=True, group=group)
        _copy_linear(m.wq_b, base_name + "Qb", out, group=group)
        _copy_linear(m.wkv_a, base_name + "KvA", out, group=group)
        _copy_norm(m.kv_norm, base_name + "NormKv", out, do_gather=True, group=group)
        _copy_linear(m.wkv_b, base_name + "KvB", out, group=group)
        _copy_linear(m.wo, base_name + "Proj", out, group=group)
        return

    (x, start_pos, freqs_cis, mask) = inputs
    freqs_cis = freqs_cis.detach()
    if mask is not None:
        mask = mask.detach()

    bsz, seqlen, _ = x.size()
    end_pos = start_pos + seqlen
    # q = self.wq_b(self.q_norm(self.wq_a(x)))
    _copy_linear(m.wq_a, base_name + "Qa", out, x=x, group=group)
    wq_a_out = out[base_name + "QaOutAct"][dist.get_rank(group=group)]
    _copy_norm(m.q_norm, base_name + "NormQ", out, x=wq_a_out, do_gather=True, group=group)
    q_norm_out = out[base_name + "NormQOutAct"][dist.get_rank(group=group)]
    _copy_linear(m.wq_b, base_name + "Qb", out, x=q_norm_out, group=group)
    q = out[base_name + "QbOutAct"][dist.get_rank(group=group)]

    q = q.view(bsz, seqlen, m.n_local_heads, m.qk_head_dim)
    q_nope, q_pe = torch.split(q, [m.qk_nope_head_dim, m.qk_rope_head_dim], dim=-1)
    q_pe = apply_rotary_emb(q_pe, freqs_cis)
    out[base_name + "QNope"] = _gather(q_nope.detach().unsqueeze(0), group=group)
    out[base_name + "QPe"] = _gather(q_pe.detach().unsqueeze(0), group=group)

    # kv = self.wkv_a(x)
    _copy_linear(m.wkv_a, base_name + "KvA", out, x=x, group=group)
    kv = out[base_name + "KvAOutAct"][dist.get_rank(group=group)]
    kv, k_pe = torch.split(kv, [m.kv_lora_rank, m.qk_rope_head_dim], dim=-1)
    k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
    out[base_name + "KPe"] = _gather(k_pe.detach().unsqueeze(0), group=group)

    _copy_linear(m.wkv_b, base_name + "KvB", out, group=group)
    wkv_b = m.wkv_b.weight if m.wkv_b.scale is None else weight_dequant(m.wkv_b.weight, m.wkv_b.scale, block_size) 
    wkv_b = wkv_b.view(m.n_local_heads, -1, m.kv_lora_rank)
    q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :m.qk_nope_head_dim])
    out[base_name + "Einsum1OutAct"] = _gather(q_nope.detach().unsqueeze(0), group=group)

    _copy_norm(m.kv_norm, base_name + "NormKv", out, x=kv, do_gather=True, group=group)
    kv_norm_out = out[base_name + "NormKvOutAct"][dist.get_rank(group=group)]
    m.kv_cache[:bsz, start_pos:end_pos] = kv_norm_out
    m.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
    scores = (torch.einsum("bshc,btc->bsht", q_nope, m.kv_cache[:bsz, :end_pos]) +
                torch.einsum("bshr,btr->bsht", q_pe, m.pe_cache[:bsz, :end_pos])) * m.softmax_scale
    if mask is not None:
        scores += mask.unsqueeze(1)
    out[base_name + "ScoresPreSoftmax"] = _gather(scores.detach().unsqueeze(0), group=group)
    scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
    out[base_name + "ScoresPostSoftmax"] = _gather(scores.detach().unsqueeze(0), group=group)

    x = torch.einsum("bsht,btc->bshc", scores, m.kv_cache[:bsz, :end_pos])
    out[base_name + "Einsum2OutAct"] = _gather(x.detach().unsqueeze(0), group=group)
    x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -m.v_head_dim:])
    out[base_name + "Einsum3OutAct"] = _gather(x.detach().unsqueeze(0), group=group)
    _copy_linear(m.wo, base_name + "Proj", out, x=x.flatten(2), gather_out=False, group=group)

def _copy_mlp(m: MLP, base_name: str, out: dict, x: torch.Tensor = None, group: dist.ProcessGroup = None):
    _copy_linear(m.w1, base_name + "Gate", out, group=group)
    _copy_linear(m.w2, base_name + "Out", out, group=group)
    _copy_linear(m.w3, base_name + "In", out, group=group)
    if x is not None:
        _copy_out(m, base_name, out, x, do_gather=False, group=group)

def _copy_expert(m: Expert, ref_m: Expert, src: int, base_name: str, out: dict, x: torch.Tensor = None, group: dist.ProcessGroup = None):
    rank = dist.get_rank(group=group)
    def get_scale(w: Linear):
        if getattr(w.weight, "scale", None) is None:
            return None
        if rank == src:
            return w.weight.scale.detach()
        return torch.empty_like(w.weight.scale)

    if rank == src:
        w1_weight = m.w1.weight.detach()
        w1_scale = get_scale(m.w1)
        w2_weight = m.w2.weight.detach()
        w2_scale = get_scale(m.w2)
        w3_weight = m.w3.weight.detach()
        w3_scale = get_scale(m.w3)
    else:
        w1_weight = torch.empty_like(ref_m.w1.weight)
        w1_scale = get_scale(ref_m.w1)
        w2_weight = torch.empty_like(ref_m.w2.weight)
        w2_scale = get_scale(ref_m.w2)
        w3_weight = torch.empty_like(ref_m.w3.weight)
        w3_scale = get_scale(ref_m.w3)

    dist.broadcast(w1_weight, src, group=group)
    dist.broadcast(w2_weight, src, group=group)
    dist.broadcast(w3_weight, src, group=group)
    if w1_scale is not None:
        dist.broadcast(w1_scale, src, group=group)
    if w2_scale is not None:
        dist.broadcast(w2_scale, src, group=group)
    if w3_scale is not None:
        dist.broadcast(w3_scale, src, group=group)
    out[base_name + "GateW"] = w1_weight
    out[base_name + "GateWscale"] = w1_scale
    out[base_name + "OutW"] = w2_weight
    out[base_name + "OutWscale"] = w2_scale
    out[base_name + "InW"] = w3_weight
    out[base_name + "InWscale"] = w3_scale

    if x is not None:
        if rank == src:
            fc1Out = F.silu(m.w1(x)) * m.w3(x)
            fc2Out = m.w2(fc1Out.detach())
        else:
            fc1Out = torch.empty(x.size(0), ref_m.w1.out_features, dtype=x.dtype, device=x.device)
            fc2Out = torch.empty_like(x)
        dist.broadcast(fc1Out, src, group=group)
        dist.broadcast(fc2Out, src, group=group)
        out[base_name + "Fc1Act"] = fc1Out
        out[base_name + "OutAct"] = fc2Out

def _copy_gate(m: Gate, base_name: str, out: dict, x: torch.Tensor = None, group: dist.ProcessGroup = None):
    # Gate must be the same on all ranks
    out[base_name + "W"] = m.weight.detach()
    out[base_name + "B"] = m.bias.detach()
    if x is not None:
        _copy_out(m, base_name, out, x, is_gate=True, group=group)

def _copy_moe(m: MoE, base_name: str, out: dict, x: torch.Tensor = None, group: dist.ProcessGroup = None):
    ref_exp = m.experts[m.experts_start_idx]
    if x is None:
        _copy_gate(m.gate, base_name + "Gate", out, group=group)
        for i in range(m.n_routed_experts):
            # just keep the ranks in sync over the expert loop
            dist.barrier(group=group)
            _copy_expert(m.experts[i], ref_exp, i // m.n_local_experts, base_name + f"Exp{i}", out, group=group)
        _copy_mlp(m.shared_experts, base_name + "ExpShared", out, group=group)
        return 

    shape = x.size()
    x = x.view(-1, m.dim)
    _copy_gate(m.gate, base_name + "Gate", out, x=x, group=group)
    weights = out[base_name + "GateOutActW"]
    indices = out[base_name + "GateOutActIdx"]
    y = torch.zeros_like(x)
    counts = torch.bincount(indices.flatten(), minlength=m.n_routed_experts)
    for i in range(m.n_routed_experts):
        # just keep the ranks in sync over the expert loop
        dist.barrier(group=group)
        src_rank = i // m.n_local_experts
        assert (dist.get_rank(group=group) != src_rank) == (m.experts[i] is None)
        if m.experts[i] is None:
            counts_src = torch.empty_like(counts)
            indices_src = torch.empty_like(indices)
        else:
            counts_src = counts.detach()
            indices_src = indices.detach()
        dist.broadcast(counts_src, src_rank, group=group)
        dist.broadcast(indices_src, src_rank, group=group)
        idx, top = torch.where(indices_src == i)
        expert_in = None if counts_src[i].item() == 0 else x[idx]
        _copy_expert(m.experts[i], ref_exp, src_rank, base_name + f"Exp{i}", out, x=expert_in, group=group)
        if expert_in is None:
            continue
        exp_out = out[base_name + f"Exp{i}OutAct"]
        if m.experts[i] is not None:
            # only the "real" rank performs the update on y, just like in ref
            y[idx] += exp_out * weights[idx, top, None]

    out[base_name + "IntRouted"] = _gather(y.detach().unsqueeze(0), group=group)

    _copy_mlp(m.shared_experts, base_name + "ExpShared", out, x=x, group=group)
    z = out[base_name + "ExpSharedOutAct"]
    if dist.get_world_size(group=group) > 1:
        dist.all_reduce(y, group=group)

    out[base_name + "Routed"] = y.detach()

    h = (y + z).view(shape)
    out[base_name + "OutAct"] = h.detach()


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0,
    args: ModelArgs = None,
    output_state_dict: dict = None
) -> List[List[int]]:
    """
    Generates new tokens based on the given prompt tokens using the specified model.

    Args:
        model (Transformer): The transformer model used for token generation.
        prompt_tokens (List[List[int]]): A list of lists containing the prompt tokens for each sequence.
        max_new_tokens (int): The maximum number of new tokens to generate.
        eos_id (int): The end-of-sequence token ID.
        temperature (float, optional): The temperature value for sampling. Defaults to 1.0.

    Returns:
        List[List[int]]: A list of lists containing the generated tokens for each sequence.
    """
    do_copy = args is not None and output_state_dict is not None
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if do_copy:
        output_state_dict[f"qkvRoPE"] = torch.view_as_real(model.freqs_cis.clone())
        output_state_dict[f"embW"] = _gather(model.embed.weight.detach())
        _copy_linear(model.head, "head", output_state_dict)
        layer_id_dense = args.n_dense_layers - 1
        layer_id_normal = args.n_dense_layers + 2
        assert layer_id_dense >= 0 and layer_id_normal + 1 < args.n_layers
        assert max_new_tokens >= 3
        pos_id_context = 0
        pos_id_gen = max(len(t) for t in prompt_tokens) + 2
        assert args.q_lora_rank != 0
    else:
        layer_id_dense, layer_id_normal, pos_id_context, pos_id_gen = -1, -1, -1, -1

    def copy_values(layer_id, module, inputs, _):
        tokens, start_pos, freqs_cis, mask = inputs
        next_module = model.layers[layer_id + 1]
        pref = f"Bs{tokens.size(0)}Lid{layer_id}Pos{start_pos}"

        next_module = model.layers[layer_id + 1]
        output_state_dict[f"{pref}ActIn"] = tokens.detach()
        _copy_norm(module.attn_norm, f"{pref}AttnNorm", output_state_dict, x=tokens)

        output_state_dict[f"{pref}Mask"] = mask
        norm_out = output_state_dict[f"{pref}AttnNormOutAct"]
        _copy_mla(module.attn, f"{pref}Attn", output_state_dict, inputs=(norm_out, start_pos, freqs_cis, mask))

        attn_out = output_state_dict[f"{pref}AttnProjOutAct"]
        act_int = output_state_dict[f"{pref}ActIn"] + attn_out
        output_state_dict[f"{pref}ActInt"] = act_int

        _copy_norm(module.ffn_norm, f"{pref}FfnNorm", output_state_dict, x=act_int)

        norm_ffn_out = output_state_dict[f"{pref}FfnNormOutAct"]
        if layer_id == layer_id_dense:
            _copy_mlp(module.ffn, f"{pref}Ffn", output_state_dict, x=norm_ffn_out)
        else:
            _copy_moe(module.ffn, f"{pref}Ffn", output_state_dict, x=norm_ffn_out)

        ffn_out = output_state_dict[f"{pref}FfnOutAct"]
        output_state_dict[f"{pref}ActOut"] = act_int + ffn_out
        _copy_norm(next_module.attn_norm, f"{pref}AttnNormNext", output_state_dict, x=output_state_dict[f"{pref}ActOut"])
        print_seq(None, f"{dist.get_rank()}: copied values for batch {tokens.size(0)}, layer {layer_id}, pos {start_pos}")

    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda")
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    prompt_mask = tokens != -1
    for cur_pos in range(min(prompt_lens), total_len):
        hooks = []
        if prev_pos == pos_id_context or prev_pos == pos_id_gen:
            hooks.append(model.layers[layer_id_dense].register_forward_hook(partial(copy_values, layer_id_dense)))
            hooks.append(model.layers[layer_id_normal].register_forward_hook(partial(copy_values, layer_id_normal)))

        print0(f"Bs {tokens.size(0)}: copying ref for pos {prev_pos}->{cur_pos}: {do_copy and bool(hooks)}")
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)

        for handle in hooks:
            handle.remove()

        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completion_tokens.append(toks)

    # save the output
    rank = dist.get_rank()
    if rank == 0 and do_copy:
        out_name = "Ref671BTp8"
        # first save non-bs related tensors
        if os.path.isfile(f"{out_name}.npz"):
            print(f"Skipping saving non-batch dependent values to {out_name}.npz !")
        else:
            print(f"Saving non-batch dependent values to {out_name}.npz")
            numpy_dict = dict()
            for key in output_state_dict:
                if key.startswith("Bs"):
                    continue
                v = output_state_dict[key]
                if v is None:
                    print(f"{key}: skipping optional value")
                    continue
                v_cpu = v.to(device="cpu", dtype=torch.float32)
                print(f"{key}: type {v.dtype}, shape {v_cpu.shape}")
                numpy_dict[key] = v_cpu.numpy()
            numpy.savez(f"{out_name}.npz", **numpy_dict)

        bs = len(PROMPTS)
        for lid in (layer_id_normal, layer_id_dense):
            for pos in (pos_id_context, pos_id_gen):
                pref = f"Bs{bs}Lid{lid}Pos{pos}"
                print(f"Saving dict to file: {out_name}{pref}.npz for rank {rank}")
                # re-write individual expert arrays to large batched array
                pref_exp = f"{pref}FfnExp"
                re_exp = re.compile(r"^" + re.escape(pref_exp) + r"[0-9]+([a-zA-Z]+[0-9]?[a-zA-Z]+)$")
                exp_keys_re = ((re_exp.fullmatch(k), k) for k in output_state_dict)
                exp_keys_m = (x for x in exp_keys_re if x[0] is not None)
                exp_keys = {m.group(1): k for m, k in exp_keys_m}

                for key_no_pref, key in exp_keys.items():
                    if "act" in key_no_pref.lower():
                        ref_val = output_state_dict[f"{pref}FfnNormOutAct"]
                        ref_val = ref_val.view(-1, ref_val.shape[-1])
                        if key_no_pref != "OutAct":
                            ref_val = ref_val[:, :output_state_dict[key].shape[-1]]
                        indexes = output_state_dict[f"{pref}FfnGateOutActIdx"]
                    else:
                        ref_val = output_state_dict[key]
                        indexes = None
                    new_shape = (args.n_routed_experts,) + ref_val.shape
                    new_val = torch.zeros(*new_shape, dtype=ref_val.dtype, device=ref_val.device)
                    for e in range(args.n_routed_experts):
                        key_exp = f"{pref_exp}{e}{key_no_pref}"
                        if key_exp not in output_state_dict:
                            continue
                        if indexes is None:
                            new_val[e].copy_(output_state_dict[key_exp])
                        else:
                            idx, _ = torch.where(indexes == e)
                            new_val[e].index_copy_(0, idx, output_state_dict[key_exp])
                        del output_state_dict[key_exp]

                numpy_dict = dict()
                torch.set_printoptions(precision=6, threshold=8, edgeitems=2, linewidth=120, sci_mode=False)
                for key in output_state_dict:
                    if not key.startswith(pref):
                        continue
                    key_no_pref = key[len(pref):]
                    v = output_state_dict[key]
                    if v is None:
                        print(f"{key_no_pref}: skipping optional value")
                        continue
                    if v.dtype.is_floating_point:
                        out_dtype = torch.float32
                    else:
                        out_dtype = v.dtype
                    v_cpu = v.to(device="cpu", dtype=out_dtype)
                    print(f"{key_no_pref}: type {v.dtype}, shape {v_cpu.shape}")
                    numpy_dict[key_no_pref] = v_cpu.numpy()

                numpy.savez(f"{out_name}{pref}.npz", **numpy_dict)


    return completion_tokens


def main(
    ckpt_path: str,
    config: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
) -> None:
    """
    Main function to load the model and perform interactive or batch text generation.

    Args:
        ckpt_path (str): Path to the model checkpoint directory.
        config (str): Path to the model configuration file.
        input_file (str, optional): Path to a file containing input prompts. Defaults to "".
        interactive (bool, optional): Whether to run in interactive mode. Defaults to True.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 100.
        temperature (float, optional): Temperature for sampling. Defaults to 1.0.
    """
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    global print0
    if rank != 0:
        print0 = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(965)
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    args.max_batch_size = len(PROMPTS)
    print0(args)
    with torch.device("cuda"):
        model = Transformer(args)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    tokenizer.decode(generate(model, [tokenizer.encode("DeepSeek")], 2, -1, 1.)[0])
    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))

    assert len(PROMPTS) <= args.max_batch_size
    output_state_dict = dict()
    prompt_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True) for prompt in PROMPTS]
    completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature, args, output_state_dict)
    completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
    for prompt, completion in zip(PROMPTS, completions):
        print0("Prompt:", prompt)
        print0("Completion:", completion)
        print0()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    """
    Command-line interface for distributed text generation.

    Arguments:
        --ckpt-path (str): Path to the model checkpoint directory.
        --config (str): Path to the model configuration file.
        --max-new-tokens (int, optional): Maximum number of new tokens to generate. Defaults to 200.
        --temperature (float, optional): Temperature for sampling. Defaults to 0.2.
    """
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()
    main(args.ckpt_path, args.config, args.max_new_tokens, args.temperature)
