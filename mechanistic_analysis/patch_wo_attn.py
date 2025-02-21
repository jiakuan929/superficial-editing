import argparse
import copy
from datetime import datetime
import os
import sys
import json
import random
from typing import Dict, Literal, List, Optional, Callable, Tuple, Union

from matplotlib import pyplot as plt
import seaborn as sns
import tqdm
import numpy as np
from omegaconf import OmegaConf

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(".")
from baselines.util import nethook
from utils import restore_model, init_model_tokenizer
from name_dict import *
from mi_tools import apply_logit_lens


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "gpt2-xl",
            "llama3-8b-it",
            "qwen2.5-7b-it",
            "qwen2.5-14b-it",
        ],
        default="llama3-8b-it",
    )
    parser.add_argument(
        "--editor", type=str, choices=["rome", "memit", "ft", "mend"], default="rome"
    )
    parser.add_argument(
        "--dataset", type=str, choices=["counterfact", "zsre"], default="counterfact"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--patch",
        type=str,
        choices=["subject_last", "last", "both", "all", "random"],
        default="last",
    )
    parser.add_argument("--toph", type=int, default=None)
    args = parser.parse_args()
    return args


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def find_subject_idxs(tok: AutoTokenizer, prompts: list, subject: str) -> List[int]:
    """
    Given a batch of prompts containing the same subject, find the last subject token position in them.
    """
    from baselines.rome.repr_tools import get_words_idxs_in_templates

    new_prompts = []
    for prompt in prompts:
        last_idx = prompt.rfind(subject)
        if last_idx != -1:
            part1 = prompt[:last_idx]
            part2 = prompt[last_idx + len(subject) :]
            new_s = part1 + "{}" + part2
            new_prompts.append(new_s)
        else:
            raise ValueError(f"{subject} NOT in {prompt}")

    idxs = get_words_idxs_in_templates(
        tok,
        new_prompts,
        words=[subject for _ in range(len(new_prompts))],
        subtoken="last",
    )
    idxs = [x[0] for x in idxs]
    return idxs


def find_sublist_position(superlist: List, sublist: List) -> int:
    assert len(superlist) != 0 and len(sublist) != 0
    first_elem = sublist[0]
    sub_length = len(sublist)
    possible_indices = [i for i, x in enumerate(superlist) if x == first_elem]
    ret = []
    for idx in possible_indices:
        if superlist[idx : idx + sub_length] == sublist:
            ret.append(idx)
    print("ret = ", ret)
    assert len(ret) == 1
    return ret[0]


def find_probe_start_idxs(
    tok: AutoTokenizer, attack_prompts: List[str], edit_prompt: str
) -> List[int]:
    question_idxs = []
    possible_questions = [edit_prompt]
    for ap in attack_prompts:
        ret = []
        for q in possible_questions:
            if q not in ap:
                continue
            ap_ids: List[int] = tok.encode(ap)
            q = f" {q}"
            q_ids: List[int] = tok.encode(q)
            print(f"ap_ids: {ap_ids}, length = {len(ap_ids)}")
            print(f"q_ids: {q_ids}")
            print(f"ap = ${tok.decode(ap_ids)}$")
            print(f"q = ${tok.decode(q_ids)}$")
            idx = find_sublist_position(ap_ids, q_ids)
            print(f"Found position: {idx}")
            ret.append(idx)
        assert len(ret) == 1
        question_idxs.append(ret[0])
    return question_idxs


def get_layer_outputs(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    layer_str: str,
) -> torch.Tensor:
    """Extract the specified layer output in a forward pass.

    Args:
        model (AutoModelForCausalLM): _description_
        tok (AutoTokenizer): _description_
        prompts (List[str]): _description_
        layer_str (str): _description_

    Returns:
        Tensor: layer output [bsz, seq_len, hidden_size]
    """
    layer_input = []
    layer_output = []

    def _get_layer_output(layer, layer_in, layer_out):
        x_in = layer_in[0]
        x_out = layer_out[0]
        layer_input.append(x_in)
        layer_output.append(x_out)

    layer = nethook.get_module(model, layer_str)
    hook = layer.register_forward_hook(_get_layer_output)

    inp_tokens = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        model(**inp_tokens)

    layer_input = layer_input[0]
    layer_output = layer_output[0]

    hook.remove()
    return layer_output


def get_attn_input_output(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: list,
    module_str: str,
    getter: Literal["in", "out", "io"] = "io",
):
    module_output = []
    module_input = []

    attn_str = module_str[: module_str.rfind(".")]
    layer_str = attn_str[: attn_str.rfind(".")]
    layer = nethook.get_module(model, layer_str)

    def _get_module_output(mod, mod_in, mod_out):
        if isinstance(mod_in, tuple):
            # print(len(mod_in), type(mod_in[0]), mod_in[0].shape)
            module_input.append(mod_in[0])
        elif isinstance(mod_in, torch.Tensor):
            module_input.append(mod_in)
        else:
            raise TypeError

        if isinstance(mod_out, torch.Tensor):
            module_output.append(mod_out)
        elif isinstance(mod_out, tuple):
            print(type(mod_out[0]), type(mod_out[1]), mod_out[1].shape, mod_out[2])
            module_output.append(mod_out[0])
        else:
            raise TypeError

    module = nethook.get_module(model, module_str)
    hook = module.register_forward_hook(_get_module_output)
    # Forward the prompt
    encoding = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        model(**encoding)

    real_lens = encoding["attention_mask"].sum(1).cpu().numpy().tolist()

    module_output = module_output[0]
    module_input = module_input[0]

    # post_attention_layernorm
    module_output = layer.post_attention_layernorm(module_output)
    module_input = layer.post_attention_layernorm(module_input)

    hook.remove()

    if getter == "in":
        return module_input
    if getter == "out":
        return module_output
    return module_input, module_output


def get_attn_head_output(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: list,
    module_str: str,
    return_last_pos: bool = False,
) -> List[torch.Tensor]:

    head_outputs = []
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    layer_str = module_str[: module_str.rfind(".self_attn")]
    layer = nethook.get_module(model, layer_str)

    def _get_module_output(mod, mod_in, mod_out):
        wo = mod.weight.clone().detach()
        wo_heads = torch.split(wo, head_dim, dim=1)

        if isinstance(mod_in, tuple):
            # print(len(mod_in), type(mod_in[0]), mod_in[0].shape)
            x_in = mod_in[0].clone().detach()
        elif isinstance(mod_in, torch.Tensor):
            x_in = mod_in.clone().detach()  # [batch, seq_len, hidden_size]
        else:
            raise TypeError

        bsz, seq_len = x_in.shape[0], x_in.shape[1]
        x_in = x_in.reshape(bsz, seq_len, -1, head_dim).transpose(
            1, 2
        )  # [bsz, h, seq_len, head_dim]

        for head_idx, h_wo in enumerate(wo_heads):
            inp = x_in[:, head_idx, :, :]
            _output = inp @ h_wo.T  # [bsz, seq_len, hidden_size]
            head_outputs.append(_output)

    module = nethook.get_module(model, module_str)
    hook = module.register_forward_hook(_get_module_output)
    # Forward the prompt
    encoding = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        model(**encoding)

    real_lens = encoding["attention_mask"].sum(1).cpu().numpy().tolist()
    print(f"Batch = {len(prompts)} | Individual length = {real_lens}")

    # # post attention layernorm
    for i in range(len(head_outputs)):
        head_outputs[i] = layer.post_attention_layernorm(head_outputs[i])

    if return_last_pos:
        new_head_outputs = []
        for h in range(model.config.num_attention_heads):
            _h_output = []
            for i in range(len(prompts)):
                _h_output.append(head_outputs[h][i, real_lens[i] - 1, :].reshape(1, -1))
            _h_output = torch.cat(_h_output, dim=0)  # [bsz, hidden_size]
            new_head_outputs.append(_h_output)
        head_outputs = new_head_outputs

    hook.remove()
    return head_outputs


def get_last_pos_repr(
    batch_tensors: torch.Tensor, real_lens: List[int]
) -> torch.Tensor:
    new_tensors = []
    for i in range(batch_tensors.shape[0]):
        new_tensors.append(batch_tensors[i, real_lens[i] - 1, :].reshape(1, -1))
    return torch.cat(new_tensors, dim=0)


def get_corrupted_vectors(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    main_heads: Dict[int, List[int]],
) -> List[torch.Tensor]:

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    main_layers = sorted(list(main_heads.keys()))
    cur_layer_idx_arr = [0]
    layer_outputs = []

    def _ablate_head_output(mod, mod_in, mod_out):
        print(f"Ablate attention head...")
        wo = mod.weight.clone().detach()
        wo_heads = torch.split(wo, head_dim, dim=1)

        if isinstance(mod_in, tuple):
            # print(len(mod_in), type(mod_in[0]), mod_in[0].shape)
            x_in = mod_in[0].clone().detach()
        elif isinstance(mod_in, torch.Tensor):
            x_in = mod_in.clone().detach()  # [batch, seq_len, hidden_size]
        else:
            raise TypeError

        if isinstance(mod_out, tuple):
            x_out = mod_out[0].clone().detach()
        elif isinstance(mod_out, torch.Tensor):
            x_out = mod_out.clone().detach()  # [bsz, seq_len, hidden_size]
        else:
            raise TypeError

        bsz, seq_len = len(prompts), x_in.shape[1]

        x_in = x_in.reshape(bsz, seq_len, -1, head_dim).transpose(
            1, 2
        )  # [bsz, h, seq_len, head_dim]
        # Execute in parallel
        clayer = main_layers[cur_layer_idx_arr[0]]
        for target_head in main_heads[clayer]:
            inp = x_in[:, target_head, :, :]  # [bsz, seq_len, head_dim]
            _output = (
                inp @ wo_heads[target_head].T
            )  # The output of this head. [bsz, seq_len, hidden_size]
            # Ablate this head by remove its output from the total output.
            x_out = x_out - _output

        # Update pointer
        cur_layer_idx_arr[0] += 1
        assert isinstance(mod_out, torch.Tensor)
        mod_out = x_out
        return mod_out

    def _extract_layer_output(layer, layer_in, layer_out):
        print("Extracting layer output...")
        x_out = layer_out[0].clone().detach()
        layer_outputs.append(x_out)

    all_hooks = []
    for layer in range(model.config.num_hidden_layers):
        all_hooks.append(
            nethook.get_module(
                model, name=f"model.layers.{layer}"
            ).register_forward_hook(_extract_layer_output)
        )

    for layer in main_layers:
        all_hooks.append(
            nethook.get_module(
                model, name=f"model.layers.{layer}.self_attn.o_proj"
            ).register_forward_hook(_ablate_head_output)
        )

    encoding = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        model(**encoding)

    for hook in all_hooks:
        hook.remove()
    return layer_outputs


def get_corrupted_vectors_ablating_attn(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    main_layers: List[int],
):
    layer_outputs = []

    def _ablate_attn_output(mod, mod_in, mod_out):
        print(f"Ablate attention...")

        if isinstance(mod_in, tuple):
            # print(len(mod_in), type(mod_in[0]), mod_in[0].shape)
            x_in = mod_in[0].clone().detach()
        elif isinstance(mod_in, torch.Tensor):
            x_in = mod_in.clone().detach()  # [batch, seq_len, hidden_size]
        else:
            raise TypeError

        if isinstance(mod_out, tuple):
            x_out = mod_out[0].clone().detach()
        elif isinstance(mod_out, torch.Tensor):
            x_out = mod_out.clone().detach()  # [bsz, seq_len, hidden_size]
        else:
            raise TypeError

        assert isinstance(mod_out, torch.Tensor)
        mod_out = torch.zeros_like(mod_out, device=mod_out.device)
        return mod_out

    def _extract_layer_output(layer, layer_in, layer_out):
        print("Extracting layer output...")
        x_out = layer_out[0].clone().detach()
        layer_outputs.append(x_out)

    all_hooks = []
    for layer in range(model.config.num_hidden_layers):
        all_hooks.append(
            nethook.get_module(
                model, name=f"model.layers.{layer}"
            ).register_forward_hook(_extract_layer_output)
        )

    for layer in main_layers:
        all_hooks.append(
            nethook.get_module(
                model, name=f"model.layers.{layer}.self_attn.o_proj"
            ).register_forward_hook(_ablate_attn_output)
        )

    encoding = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        model(**encoding)

    for hook in all_hooks:
        hook.remove()
    return layer_outputs


def patch_middle_and_extract_later(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    patch_token_idx_clean: Union[List[int], int],
    patch_token_idx_corrupted: Union[List[int], List[List[int]]],
    patch_vectors: List[torch.Tensor],
    patch_layer_idx: int,
) -> torch.Tensor:
    """Patches specific layer and returns the final probability distribution.

    Args:
        model (AutoModelForCausalLM): _description_
        tok (AutoTokenizer): _description_
        prompts (List[str]): _description_
        patch_token_idx (List[int]): _description_
        patch_vectors (torch.Tensor): _description_
        patch_layer_idx (int): _description_

    Raises:
        TypeError: _description_

    Returns:
        torch.Tensor: _description_
    """

    layer_outputs = {
        int(layer): []
        for layer in range(patch_layer_idx + 1, model.config.num_hidden_layers)
    }

    if isinstance(patch_token_idx_clean, list) and len(patch_token_idx_clean) == 1:
        patch_token_idx_clean = patch_token_idx_clean[0]  # Unwrap

    def _patch_layer(layer, layer_in, layer_out):
        """
        Knockout the residual stream at a specific layer.
        """
        new_out = list(copy.deepcopy(layer_out))
        # x_in = layer_in[0]  # Get layer input
        x_out = layer_out[0].clone().detach()  # Get layer output

        # patch value
        patch_v = patch_vectors[patch_layer_idx].to(
            x_out.device
        )  # [bsz, seq_len, hidden_size]

        if isinstance(patch_token_idx_clean, int):
            # Only patch one position
            print(f"Patch only one position: {patch_token_idx_clean}")
            patch_v = torch.cat(
                [
                    patch_v[j, idx, :].reshape(1, -1)
                    for j, idx in enumerate(patch_token_idx_corrupted)
                ],
                dim=0,
            ).mean(
                dim=0
            )  # [hidden_size,]
            x_out[0, patch_token_idx_clean, :] = patch_v  # replace.
        elif isinstance(patch_token_idx_clean, list):
            # Patch at 2 positions: subject_last and last
            print(f"Patch 2 positions: {patch_token_idx_clean}")
            patch_v_subj = torch.cat(
                [
                    patch_v[j, idx[0], :].reshape(1, -1)
                    for j, idx in enumerate(patch_token_idx_corrupted)
                ],
                dim=0,
            ).mean(dim=0)
            patch_v_last = torch.cat(
                [
                    patch_v[j, idx[1], :].reshape(1, -1)
                    for j, idx in enumerate(patch_token_idx_corrupted)
                ],
                dim=0,
            ).mean(dim=0)
            x_out[0, patch_token_idx_clean[0], :] = patch_v_subj
            x_out[0, patch_token_idx_clean[1], :] = patch_v_last
        else:
            raise TypeError

        new_out[0] = x_out
        new_out = tuple(new_out)
        layer_out = new_out
        return layer_out

    hook = nethook.get_module(
        model, name=f"model.layers.{patch_layer_idx}"
    ).register_forward_hook(_patch_layer)

    # Forward
    inp_toks = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        logits = model(**inp_toks).logits

    probs = torch.softmax(logits, dim=-1)

    hook.remove()
    return probs


def patch_middle_all_replaced(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    patch_token_idx_corrupted: int,
    patch_vectors: List[torch.Tensor],
    patch_layer_idx: int,
) -> torch.Tensor:
    """Patches specific layer and returns the final probability distribution.

    Args:
        model (AutoModelForCausalLM): _description_
        tok (AutoTokenizer): _description_
        prompts (List[str]): _description_
        patch_token_idx (List[int]): _description_
        patch_vectors (torch.Tensor): _description_
        patch_layer_idx (int): _description_

    Raises:
        TypeError: _description_

    Returns:
        torch.Tensor: _description_
    """

    layer_outputs = {
        int(layer): []
        for layer in range(patch_layer_idx + 1, model.config.num_hidden_layers)
    }

    def _patch_layer(layer, layer_in, layer_out):
        """
        Knockout the residual stream at a specific layer.
        """
        new_out = list(copy.deepcopy(layer_out))
        # x_in = layer_in[0]  # Get layer input
        x_out = layer_out[0].clone().detach()  # Get layer output

        # patch value
        patch_v = patch_vectors[patch_layer_idx].to(
            x_out.device
        )  # [bsz, seq_len, hidden_size]
        s_len, hdim = patch_v.shape[1], patch_v.shape[-1]
        patch_v = patch_v.mean(dim=0).reshape(1, s_len, hdim)
        assert (
            patch_v.shape[1] == x_out.shape[1] + patch_token_idx_corrupted
        ), f"{patch_v.shape}, {x_out.shape}, {patch_token_idx_corrupted}"
        x_out = patch_v[:, patch_token_idx_corrupted:, :]

        new_out[0] = x_out
        new_out = tuple(new_out)
        layer_out = new_out
        return layer_out

    hook = nethook.get_module(
        model, name=f"model.layers.{patch_layer_idx}"
    ).register_forward_hook(_patch_layer)

    # Forward
    inp_toks = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        logits = model(**inp_toks).logits

    probs = torch.softmax(logits, dim=-1)

    hook.remove()
    return probs


def eina(edit_p: str, aps: List[str]):
    ret = [x for x in aps if edit_p in x]
    assert len(ret) <= 1
    return ret


def get_top_heads(model_name: str, topk: int):
    # Only consider ROME here.
    dir_dict = {
        "llama3-8b-it": "./exp_results/analysis_full/attn_layer_head_op/llama3-8b-it/_rome_counterfact_attack_2025-01-09-21-41",
        "qwen2.5-7b-it": "./exp_results/analysis_full/attn_layer_head_op/qwen2.5-7b-it/_rome_counterfact_attack_2025-01-10-02-08",
        "qwen2.5-14b-it": "./exp_results/analysis_full/attn_layer_head_op/qwen2.5-14b-it/_rome_counterfact_attack_2025-01-09-19-22"
    }
    target_dir = dir_dict[model_name]

    with open(f"{target_dir}/numeric_results.json", 'r', encoding='utf-8') as f:
        results = json.load(f)

    vals = np.array(results).mean(axis=0)    # [n_layer, n_head]
    vals_flatten = torch.from_numpy(vals).reshape(-1)

    if model_name == "llama3-8b-it":
        n_head = 32
    elif model_name == 'qwen2.5-7b-it':
        n_head = 28
    elif model_name == "qwen2.5-14b-it":
        n_head = 40
    else:
        raise NotImplementedError

    indices = torch.topk(vals_flatten, k=topk).indices.reshape(-1).tolist()
    indices = [(x // n_head, x % n_head, vals[x // n_head, x % n_head]) for x in indices]
    print(indices)
    main_heads = {}
    for item in indices:
        if item[0] not in main_heads:
            main_heads[item[0]] = [item[1]]
        else:
            main_heads[item[0]].append(item[1])
    return main_heads


def compute_batch_next_probs(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    target_id: int,
) -> List[float]:
    
    real_lens = [len(tok.encode(x)) for x in prompts]
    inp_tokens = tok(prompts, return_tensors='pt', padding=True).to(model.device)
    with torch.no_grad():
        logits = model(**inp_tokens).logits
    
    batch_probs = []

    for i, p_len in enumerate(real_lens):
        _prob = torch.softmax(logits[i, p_len - 1, :].reshape(1, -1), dim=-1)
        batch_probs.append(_prob[0, target_id].item())
    return batch_probs


def main(
    model_name: str,
    editor: str,
    dataset_name: str,
    patch_position: Literal["subject_last", "last", "both", "all", "random"],
    toph: Optional[int],
):

    model, tok = init_model_tokenizer(model_name=MODEL_NAME_DICT[model_name])
    # Dataset
    datapath = ANALYSIS_DATASET_DICT[dataset_name].format(
        model=model_name, editor=editor
    )
    with open(datapath, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    assert all(len(x["attack_probes_em"]) != 0 for x in dataset)

    hparams = OmegaConf.load(f"./hparams/{editor}/{model_name}.yaml")
    apply_editing: Callable = ALG_DICT[editor]

    if model_name == "llama3-8b-it":
        main_layers = [23, 24, 27, 30, 31]
        main_heads = {
            23: [27],
            24: [3],
            27: [20],
            30: [29],
            31: [6, 7],
        }
    elif model_name == "qwen2.5-7b-it":
        main_layers = [23, 24, 26, 27]
        main_heads = {23: [4, 6, 11], 24: [27], 26: [0], 27: [2, 3, 15]}
    elif model_name == "qwen2.5-14b-it":
        # main_layers = [36, 40, 41, 42, 43, 45, 46]
        # main_heads = {
        #     36: [10, 14, 30],
        #     40: [22, 23],
        #     41: [14],
        #     42: [21],
        #     43: [36],
        #     45: [27, 37],
        #     46: [4, 28]
        # }
        main_layers = [36, 40, 41, 42, 43, 45, 46]
        main_heads = {
            36: [10],
            40: [22, 23],
            41: [14],
            42: [21],
            43: [36],
            45: [27, 37],
            46: [4, 28],
        }
    else:
        raise NotImplementedError

    # if toph is not None:
    #     main_heads = get_top_heads(model_name, topk=toph)
    print(f"main_heads:\n{main_heads}")

    results = []
    nopatch_results = []
    patch_random_idxs = []
    considered_examples = 0

    for record in tqdm.tqdm(dataset):
        if "requested_rewrite" in record:
            # For CounterFact
            request = record["requested_rewrite"]
        else:
            # For ZsRE
            request = {
                "prompt": record["src"].replace(record["subject"], "{}"),
                "subject": record["subject"],
                "target_new": {"str": record["alt"]},
                "target_true": {"str": record["answers"][0]},
            }
        target_new_ids = tok.encode(f" {request['target_new']['str']}")
        target_true_ids = tok.encode(f" {request['target_true']['str']}")
        edit_prompt = request["prompt"].format(request["subject"])

        attack_probes = eina(edit_prompt, record["attack_probes_em"])
        if len(attack_probes) == 0:
            continue
        considered_examples += 1

        if patch_position == "subject_last":
            patch_idxs_a = find_subject_idxs(
                tok, attack_probes, subject=request["subject"]
            )
            patch_idxs_e = find_subject_idxs(
                tok, [edit_prompt], subject=request["subject"]
            )
        elif patch_position == "last":
            # Last position patch
            real_lens = [len(tok.encode(x)) for x in attack_probes]
            patch_idxs_a = [x - 1 for x in real_lens]
            patch_idxs_e = [-1]
        elif patch_position == "both":
            real_lens = [len(tok.encode(x)) for x in attack_probes]
            subject_idxs_a = find_subject_idxs(
                tok, attack_probes, subject=request["subject"]
            )
            patch_idxs_a = [[x, y - 1] for x, y in zip(subject_idxs_a, real_lens)]
            subject_idxs_e = find_subject_idxs(
                tok, [edit_prompt], subject=request["subject"]
            )
            patch_idxs_e = [subject_idxs_e[0], -1]
        elif patch_position == "all":
            probe_start_idx = find_probe_start_idxs(tok, attack_probes, edit_prompt)
            print(f"edit_prompt_ids: {tok.encode(edit_prompt)}")
        elif patch_position == "random":
            subject_idx_e = find_subject_idxs(
                tok, [edit_prompt], subject=request["subject"]
            )[0]
            subject_length = len(tok.encode(f" {request['subject'].strip()}"))
            subject_range = list(
                range(subject_idx_e + 1 - subject_length, subject_idx_e + 1)
            )
            edit_length = len(tok.encode(edit_prompt))
            unexpected_range = subject_range + [edit_length - 1]
            patch_idxs_e = unexpected_range[0]
            while patch_idxs_e in unexpected_range:
                patch_idxs_e = random.choice(range(0, edit_length))
            assert patch_idxs_e not in unexpected_range
            probe_start_idx: int = find_probe_start_idxs(
                tok, attack_probes, edit_prompt
            )[0]
            patch_idxs_a = [probe_start_idx + patch_idxs_e]
            _log = {
                "case_id": record["case_id"],
                "clean_idx": patch_idxs_e,
                "corrupted_idx": patch_idxs_a,
                "probe_start_idx": probe_start_idx,
            }
            patch_random_idxs.append(_log)
            print(_log)
        else:
            raise ValueError(
                f"Argument patch_position must be one of 'subject_last', 'last', or 'both', got {patch_position}"
            )
        # print(f"subject_a: {patch_idxs_a}\nsubject_e: {patch_idxs_e}")

        # Perform model editing.
        edited_model, weights_copy = apply_editing(
            model,
            tok,
            [request],
            hparams=hparams,
            return_orig_weights=True,
        )

        if toph is not None:
            print(f"Computing patch_value with some heads ablated...")
            patch_values = get_corrupted_vectors(edited_model, tok, prompts=attack_probes, main_heads=main_heads)
        else:
            print("Computing patch_value with attn module ablated...")
            patch_values = get_corrupted_vectors_ablating_attn(
                edited_model, tok, prompts=attack_probes, main_layers=main_layers
            )
        # Patch layers
        p_orig = []  # length == n_layer
        p_new = []
        for layer in range(model.config.num_hidden_layers):
            probs = patch_middle_and_extract_later(
                edited_model,
                tok,
                prompts=[edit_prompt],
                patch_token_idx_clean=patch_idxs_e,
                patch_token_idx_corrupted=patch_idxs_a,
                patch_vectors=patch_values,
                patch_layer_idx=layer,
            )  # [1, seq_len, V]

            assert probs.shape[0] == 1
            p_orig.append(probs[0, -1, target_true_ids[0]].item())
            p_new.append(probs[0, -1, target_new_ids[0]].item())

        results.append([p_orig, p_new])
        nopatch_results.append([
            compute_batch_next_probs(edited_model, tok, [edit_prompt], target_true_ids[0])[0],
            compute_batch_next_probs(edited_model, tok, [edit_prompt], target_new_ids[0])[0]
        ])

        # Restore
        model = restore_model(edited_model, weights_copy)

    this_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    tag = f"{toph}heads" if toph is not None else "rm_attn"
    output_dir = f"./complete_results/analysis_full/patch_wo_attn/{model_name}/{tag}_{patch_position}_{editor}_{dataset_name}_{this_time}"
    os.makedirs(output_dir, exist_ok=True)

    p_orig = [x[0] for x in results]
    p_new = [x[1] for x in results]

    p_orig = np.array(p_orig).mean(axis=0).tolist()
    p_new = np.array(p_new).mean(axis=0).tolist()
    
    nopatch_orig = [x[0] for x in nopatch_results]
    nopatch_new = [x[1] for x in nopatch_results]

    if len(patch_random_idxs) != 0:
        with open(f"{output_dir}/random_idxs.json", "w", encoding="utf-8") as f:
            json.dump(patch_random_idxs, f, indent=4)

    with open(f"{output_dir}/orig.json", "w", encoding="utf-8") as f:
        json.dump(p_orig, f, indent=4)

    with open(f"{output_dir}/new.json", "w", encoding="utf-8") as f:
        json.dump(p_new, f, indent=4)
    
    with open(f"{output_dir}/nopatch.json", "w", encoding="utf-8") as f:
        json.dump(nopatch_results, f, indent=4)

    with open(f"{output_dir}/config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "considered": considered_examples,
                "patch_target": patch_position,
                "toph": toph,
                "main_heads": main_heads,
            },
            f,
            indent=4,
        )


    plt.figure()
    for label, data in zip(["orig", "new"], [p_orig, p_new]):
        plt.plot(range(len(data)), data, label=label, marker="o", markersize=3)
    plt.axhline(y=np.mean(nopatch_orig), color='red', linewidth=2, linestyle='--', label='nopatch_orig_prob')
    plt.axhline(y=np.mean(nopatch_new), color='green', linewidth=2, linestyle='--', label='nopatch_new_prob')
    plt.xlabel("Layer")
    plt.ylabel("Probability")
    plt.legend()
    plt.savefig(f"{output_dir}/patched.pdf")


"""
This script compute the latent probability of o and o* at every later layer when the resid at the specified layer is reset.
"""

if __name__ == "__main__":
    args = get_arguments()

    seed_everything(args.seed)

    main(args.model, args.editor, args.dataset, args.patch, args.toph)
