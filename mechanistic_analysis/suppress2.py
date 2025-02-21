import argparse
import copy
from datetime import datetime
import math
import os
import sys
import json
import random
from typing import Literal, List, Optional, Callable, Union

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
) -> List[torch.Tensor]:

    ret_values = []
    for layer in range(model.config.num_hidden_layers):
        layer_str = f"model.layers.{layer}"
        layer_output = get_layer_outputs(
            model, tok, prompts, layer_str
        )  # [bsz, seq_len, hidden_size]
        ret_values.append(layer_output)
    return ret_values


def patch_middle_and_extract_later(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    patch_token_idx_clean: Union[List[int], int],
    patch_token_idx_corrupted: Union[List[int], List[List[int]]],
    patch_vectors: List[torch.Tensor],
    patch_layer_idxs: List[int],
) -> List[torch.Tensor]:
    """Patches specific layer and returns the final probability distribution."""

    cur_layer_idx = [0]
    layer_outputs = []

    if isinstance(patch_token_idx_clean, list) and len(patch_token_idx_clean) == 1:
        patch_token_idx_clean = patch_token_idx_clean[0]  # Unwrap

    def _patch_or_extract_layer(layer, layer_in, layer_out):
        """
        Knockout the residual stream at a specific layer.
        """
        new_out = list(copy.deepcopy(layer_out))
        # x_in = layer_in[0]  # Get layer input
        x_out = layer_out[0].clone().detach()  # Get layer output

        # Current layer index
        clayer = cur_layer_idx[0]
        cur_layer_idx[0] += 1
        print(f"Current layer: {clayer}")

        # patch value
        if clayer in patch_layer_idxs:
            print("Patching......")
            patch_v = patch_vectors[clayer].to(
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
                assert isinstance(patch_token_idx_corrupted[0], list)
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

            layer_outputs.append(x_out)
            new_out[0] = x_out
            new_out = tuple(new_out)
            layer_out = new_out
            return layer_out
        layer_outputs.append(
            x_out
        )  # Otherwise, we do not interven this layer and directly extract its output.

    all_hooks = [
        nethook.get_module(model, f"model.layers.{layer}").register_forward_hook(
            _patch_or_extract_layer
        )
        for layer in range(model.config.num_hidden_layers)
    ]

    # Forward
    inp_toks = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        logits = model(**inp_toks).logits

    probs = torch.softmax(logits, dim=-1)

    for hook in all_hooks:
        hook.remove()
    assert len(layer_outputs) == model.config.num_hidden_layers
    return layer_outputs


def test_subject_idx_search(
    tok: AutoTokenizer,
):
    prompt = "China is the largest country in east Asia. The capital of China is"
    subject = "China"

    prompt = "I love Windows XP. In my memory, Windows XP is produced by"
    subject = "Windows XP"

    subject_idxs = find_subject_idxs(tok, [prompt], subject)

    print(subject_idxs)

    inp_ids = tok.encode(prompt)

    print([(i, tok.decode(x)) for i, x in enumerate(inp_ids)])


def eina(edit_p: str, aps: List[str]):
    ret = [x for x in aps if edit_p in x]
    assert len(ret) <= 1
    return ret


def compute_token_rank_in_logits(probs: torch.Tensor, token_id: int) -> int:
    """Compute the token rank in a specified probability distribution.

    Args:
        probs (torch.Tensor): shape [batch, V]
        token_id (int): Target token

    Returns:
        int: Average rank.
    """
    token_probs = probs[:, token_id]
    sorted_p, sorted_indices = torch.sort(probs, dim=1, descending=True)
    ranks = torch.zeros_like(token_probs, dtype=torch.long)
    for i in range(probs.shape[0]):
        ranks[i] = (sorted_indices[i] == token_id).nonzero(as_tuple=True)[0].item()
    return np.mean(ranks.cpu().tolist())


def main(
    model_name: str,
    editor: str,
    dataset_name: str,
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
    edited_layers: List[int] = hparams.layers

    results = []
    rank_results = []

    if model_name == "llama3-8b-it":
        patch_range_layers = {
            "rome": list(range(1, 16)),
            "memit": list(range(1, 25))   # 1, 6, 11, 16, 21
        }[editor]
    elif model_name == "qwen2.5-7b-it":
        patch_range_layers = {
            "rome": list(range(1, 22)),
            "memit": list(range(2, 21))
            # "memit": [list(range(2, 8)), list(range(8, 14)), list(range(14, 21))],   # 2 --20 
        }[editor]
    elif model_name == "qwen2.5-14b-it":
        patch_range_layers = {
            "rome": list(range(11, 34)),
            "memit": list(range(10, 33))
        }[editor]
    else:
        raise NotImplementedError


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
        target_new_str = request["target_new"]["str"]
        target_true_ids = tok.encode(f" {request['target_true']['str']}")
        edit_prompt = request["prompt"].format(request["subject"])

        attack_probes = eina(edit_prompt, record["attack_probes_em"])
        if len(attack_probes) == 0:
            continue

        patch_idxs_a: List[int] = find_subject_idxs(
            tok, attack_probes, subject=request["subject"]
        )
        patch_idxs_e: List[int] = find_subject_idxs(
            tok, [edit_prompt], subject=request["subject"]
        )
        subject_idx_a: int = patch_idxs_a[0]
        subject_idx_e: int = patch_idxs_e[0]

        # Perform model editing.
        edited_model, weights_copy = apply_editing(
            model,
            tok,
            [request],
            hparams=hparams,
            return_orig_weights=True,
        )

        # Nopatched.
        vanilla_layer_outputs = [
            get_layer_outputs(
                edited_model,
                tok,
                prompts=[edit_prompt],
                layer_str=f"model.layers.{layer}",
            )
            for layer in range(model.config.num_hidden_layers)
        ]
        vanilla_logp = []
        for _tensor in vanilla_layer_outputs:
            prob = apply_logit_lens(edited_model, _tensor, softmax=True)[
                :, subject_idx_e, :
            ]
            pn = prob[0, target_new_ids[0]].item()
            vanilla_logp.append(-math.log(pn))

        patched_layer_outputs = [
            get_layer_outputs(
                edited_model,
                tok,
                prompts=attack_probes,
                layer_str=f"model.layers.{layer}"
            )
            for layer in range(model.config.num_hidden_layers)
        ]

        patched_logp = []
        orig_layer_rank = []
        new_layer_rank = []
        for _tensor in patched_layer_outputs:
            prob = apply_logit_lens(edited_model, _tensor, softmax=True)[
                :, subject_idx_a, :
            ]   # [bsz, V]

            ranko = compute_token_rank_in_logits(prob, token_id=target_true_ids[0])
            rankn = compute_token_rank_in_logits(prob, token_id=target_new_ids[0])
            orig_layer_rank.append(ranko)
            new_layer_rank.append(rankn)

            pn = prob[0, target_new_ids[0]].item()
            patched_logp.append(-math.log(pn))
        
        rank_results.append([orig_layer_rank, new_layer_rank])
        results.append([vanilla_logp, patched_logp])

        # Restore
        model = restore_model(edited_model, weights_copy)

    this_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_dir = f"./complete_results/analysis_full/suppress_direct/{model_name}/{editor}_{this_time}"
    os.makedirs(output_dir, exist_ok=True)

    # Save experiment configurations
    with open(f"{output_dir}/config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_name,
                "editor": editor,
                "dataset": dataset_name,
            },
            f,
            indent=4,
        )

    with open(f"{output_dir}/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    
    with open(f"{output_dir}/rank_results.json", 'w', encoding='utf-8') as f:
        json.dump(rank_results, f, indent=4)

    vanilla_probs = [x[0] for x in results]
    patched_probs = [x[1] for x in results]

    vanilla_probs = np.array(vanilla_probs).mean(axis=0).tolist()
    patched_probs = np.array(patched_probs).mean(axis=0).tolist()
    ranko = [x[0] for x in rank_results]
    rankn = [x[1] for x in rank_results]
    ranko = np.array(ranko).mean(axis=0).tolist()
    rankn = np.array(rankn).mean(axis=0).tolist()

    plt.figure()
    for key, data in zip(["vanilla", "patched"], [vanilla_probs, patched_probs]):
        plt.plot(range(len(data)), data, label=key, marker="o", markersize=3)
    plt.legend()
    plt.xlabel("Layer")
    plt.ylabel("log(p)")
    plt.savefig(f"{output_dir}/new_prob.pdf")

    plt.figure()
    for key, data in zip(["orig", "new"], [ranko, rankn]):
        plt.plot(range(len(data)), data, label=key, marker='o', markersize=3)
    plt.legend()
    plt.xlabel('Layer')
    plt.ylabel('Rank')
    plt.savefig(f"{output_dir}/rank.pdf")


"""
This script compute the latent probability of o and o* at every layer when the edited model takes `e` or `a` as input.
"""

if __name__ == "__main__":
    args = get_arguments()

    seed_everything(args.seed)

    main(args.model, args.editor, args.dataset)
