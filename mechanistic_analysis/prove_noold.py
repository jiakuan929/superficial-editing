import argparse
from datetime import datetime
import math
import os
import sys
import json
import random
from typing import Literal, List, Optional, Callable

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
from interpretability.mi_tools import apply_logit_lens, dead_loop


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

    layer_output = []

    def _get_layer_output(layer, layer_in, layer_out):
        x_in = layer_in[0]
        x_out = layer_out[0]
        layer_output.append(x_out)

    layer = nethook.get_module(model, layer_str)
    hook = layer.register_forward_hook(_get_layer_output)

    inp_tokens = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        model(**inp_tokens)

    layer_output = layer_output[0]
    hook.remove()
    return layer_output


def get_attn_input_output(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: list,
    module_str: str,
    return_last_pos: bool = False,
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
    if return_last_pos:
        new_module_input, new_module_output = [], []
        for i, _length in enumerate(real_lens):
            new_module_input.append(module_input[i, _length - 1, :].reshape(1, -1))
            new_module_output.append(module_output[i, _length - 1, :].reshape(1, -1))
        module_input = torch.cat(new_module_input, dim=0)
        module_output = torch.cat(new_module_output, dim=0)

    if getter == "in":
        return module_input
    if getter == "out":
        return module_output
    return module_input, module_output


def get_last_pos_repr(
    batch_tensors: torch.Tensor, real_lens: List[int]
) -> torch.Tensor:
    new_tensors = []
    for i in range(batch_tensors.shape[0]):
        new_tensors.append(batch_tensors[i, real_lens[i] - 1, :].reshape(1, -1))
    return torch.cat(new_tensors, dim=0)


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


def test_subject_idx_search(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompt: str,
    subject: str,
):
    
    def _gpt2_logit_lens(hidden_state: torch.Tensor, softmax: bool = False) -> torch.Tensor:
        x = model.lm_head(model.transformer.ln_f(hidden_state))
        if softmax:
            x = torch.softmax(x, dim=-1)
        return x

    subject_idxs = find_subject_idxs(tok, [prompt], subject)

    print(subject_idxs)

    inp_ids = tok.encode(prompt)

    print([(i, tok.decode(x)) for i, x in enumerate(inp_ids)])

    layer_outputs = []
    def _hook_fn(mod, mod_in, mod_out):
        if isinstance(mod_out, tuple):
            x_out = mod_out[0]
        elif isinstance(mod_out, torch.Tensor):
            x_out = mod_out
        else:
            raise TypeError
        layer_outputs.append(x_out)
    
    hooks = []
    for layer in range(model.config.num_hidden_layers):
        layer_str = f"model.layers.{layer}.self_attn.o_proj"
        hook = nethook.get_module(model, layer_str).register_forward_hook(_hook_fn)
        hooks.append(hook)
    
    inp_toks = tok(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        logits = model(**inp_toks).logits
    fprob = torch.softmax(logits[:, -1, :], dim=-1)
    ftop = torch.topk(fprob, k=10).indices.reshape(-1)

    assert len(layer_outputs) == model.config.num_hidden_layers

    real_lens = [x + 1 for x in subject_idxs]
    real_lens = [len(tok.encode(prompt))]
    
    for i in range(len(layer_outputs)):
        layer_outputs[i] = get_last_pos_repr(layer_outputs[i], real_lens=real_lens)
        if 'gpt' in model.config._name_or_path.lower():
            prob = _gpt2_logit_lens(layer_outputs[i], softmax=True)
        else:
            prob = apply_logit_lens(model, layer_outputs[i], softmax=True)
        topk_token_ids = torch.topk(prob, k=15).indices.reshape(-1)
        top_tokens = [tok.decode(x) for x in topk_token_ids]
        print(f"Layer {i}, Top tokens: {top_tokens}")
    
    print(f"final prediction: {[tok.decode(x) for x in ftop]}")
    sys.exit()


def eina(edit_p: str, aps: List[str]):
    ret = [x for x in aps if edit_p in x]
    assert len(ret) <= 1
    return ret


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

    results = []
    rank_results = []
    considered = 0

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
        target_new_str = request['target_new']['str']
        target_true_ids = tok.encode(f" {request['target_true']['str']}")
        target_true_str = request['target_true']['str']
        edit_prompt = request["prompt"].format(request["subject"])

        attack_probes = eina(edit_prompt, record['attack_probes_em'])
        if len(attack_probes) == 0:
            continue
        considered += 1

        subject_idxs_a = find_subject_idxs(tok, attack_probes, subject=request['subject'])
        subject_idxs_e = find_subject_idxs(tok, [edit_prompt], subject=request['subject'])
        real_lens = [x + 1 for x in subject_idxs_a]

        # Perform model editing.
        edited_model, weights_copy = apply_editing(
            model,
            tok,
            [request],
            hparams=hparams,
            return_orig_weights=True,
        )

        inp_prompts = attack_probes
        orig_rate = []
        new_rate = []
        orig_rank = []
        new_rank = []
        for layer in range(model.config.num_hidden_layers):
            layer_str = f"model.layers.{layer}"

            # Take attack_probes as input
            layer_output = get_layer_outputs(
                edited_model, tok, inp_prompts, layer_str
            )
            latent_probs_attack = apply_logit_lens(
                edited_model,
                layer_output,
                softmax=True,
            )  # [bsz, seq_len, V]
            latent_probs_attack = get_last_pos_repr(
                latent_probs_attack, real_lens
            )  # [bsz, V]

            ranko = compute_token_rank_in_logits(latent_probs_attack, token_id=target_true_ids[0])
            rankn = compute_token_rank_in_logits(latent_probs_attack, token_id=target_new_ids[0])
            orig_rank.append(ranko)
            new_rank.append(rankn)

            po = latent_probs_attack[:, target_true_ids[0]].mean().item()
            pn = latent_probs_attack[:, target_new_ids[0]].mean().item()
            orig_rate.append(-math.log(po))
            new_rate.append(-math.log(pn))

        results.append([orig_rate, new_rate])
        rank_results.append([orig_rank, new_rank])
        # Restore
        model = restore_model(edited_model, weights_copy)

    this_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_dir = f"./complete_results/analysis_full/prove_no_old/{model_name}/{editor}_{dataset_name}_{this_time}"
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
    
    with open(f"{output_dir}/results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    with open(f"{output_dir}/rank.json", 'w', encoding='utf-8') as f:
        json.dump(rank_results, f, indent=4)
    
    orig_rate = [x[0] for x in results]
    new_rate = [x[1] for x in results]
    orig_rate = np.array(orig_rate).mean(axis=0).tolist()
    new_rate = np.array(new_rate).mean(axis=0).tolist()

    plt.figure()
    for key, data in zip(["orig", "new"], [orig_rate, new_rate]):
        plt.plot(range(len(data)), data, label=key, marker='o', markersize=3)
    plt.legend()
    plt.xlabel('Layer')
    plt.ylabel('-log(p)')
    plt.yscale('log')
    plt.savefig(f"{output_dir}/noold.pdf")

    ranko = [x[0] for x in rank_results]
    rankn = [x[1] for x in rank_results]
    ranko = np.array(ranko).mean(axis=0).tolist()
    rankn = np.array(rankn).mean(axis=0).tolist()

    plt.figure()
    for key, data in zip(["orig", "new"], [ranko, rankn]):
        plt.plot(range(len(data)), data, label=key, marker='o', markersize=3)
    plt.legend()
    plt.xlabel('Layer')
    plt.ylabel('Rank')
    plt.yscale('log')
    plt.savefig(f"{output_dir}/rank.pdf")


if __name__ == "__main__":
    args = get_arguments()

    seed_everything(args.seed)

    main(args.model, args.editor, args.dataset)
