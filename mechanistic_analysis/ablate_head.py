import argparse
from datetime import datetime
import os
import sys
import json
import random
from typing import Dict, Literal, List, Tuple

from matplotlib import pyplot as plt
import pandas as pd
import tqdm
import seaborn as sns
import numpy as np
from omegaconf import OmegaConf

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(".")
from baselines.util import nethook
from utils import restore_model, init_model_tokenizer, get_main_heads
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


def compute_batch_next_probs(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    target_id: int,
) -> List[float]:

    real_lens = [len(tok.encode(x)) for x in prompts]
    # inp_tokens = tok(
    #     prompts, return_tensors="pt", padding=True, max_length=max_len, truncation=True
    # ).to(model.device)
    inp_tokens = tok(
        prompts, return_tensors="pt", padding=True,
    ).to(model.device)
    with torch.no_grad():
        logits = model(**inp_tokens).logits

    batch_probs = []

    for i, p_len in enumerate(real_lens):
        _prob = torch.softmax(logits[i, p_len - 1, :].reshape(1, -1), dim=-1)
        batch_probs.append(_prob[0, target_id].item())
    return batch_probs


def ablate_several_attention_heads(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: list,
    main_layers: List[int],
    main_heads: List[int],
    ans_id: int,
) -> Tuple[List, List]:

    # Calculate the probs before ablation
    ans_probs_0 = compute_batch_next_probs(model, tok, prompts, ans_id)

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    num_head = model.config.num_attention_heads
    cur_layer_arr_idx = [0]

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
        for target_head in main_heads[main_layers[cur_layer_arr_idx[0]]]:
            inp = x_in[:, target_head, :, :]  # [bsz, seq_len, head_dim]
            _output = (
                inp @ wo_heads[target_head].T
            )  # The output of this head. [bsz, seq_len, hidden_size]
            # Ablate this head by remove its output from the total output.
            x_out = x_out - _output

        # Update pointer
        cur_layer_arr_idx[0] += 1
        assert isinstance(mod_out, torch.Tensor)
        mod_out = x_out
        return mod_out

    registered_hooks = []
    for layer in main_layers:
        module_str = f"model.layers.{layer}.self_attn.o_proj"
        module = nethook.get_module(model, module_str)
        hook = module.register_forward_hook(_get_module_output)
        registered_hooks.append(hook)

    ans_probs = compute_batch_next_probs(model, tok, prompts, ans_id)

    for hook in registered_hooks:
        hook.remove()

    average_effect = (np.array(ans_probs_0) - np.array(ans_probs)).mean().item()
    return ans_probs_0, ans_probs


"""
Is it possible that these attention heads jointly determine the reversion result? In this case, we need to ablate these heads at the same time.
"""


def main(
    model_name: str,
    editor: str,
    dataset_name: str,
):
    model, tok = init_model_tokenizer(MODEL_NAME_DICT[model_name])
    # Dataset
    datapath = ANALYSIS_DATASET_DICT[dataset_name].format(
        model=model_name, editor=editor
    )
    with open(datapath, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    assert all(len(x["attack_probes_em"]) != 0 for x in dataset)

    hparams = OmegaConf.load(f"./hparams/{editor}/{model_name}.yaml")
    
    main_layers, main_heads = get_main_heads(
        path=f"./hparams/nonorm_attn_heads/{model_name}.json",
        editor=editor,
    )
    print(f"main_layers: {main_layers}")
    print(f"main_heads: {main_heads}")

    apply_editing = ALG_DICT[editor]

    results = []
    new_results = []

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
        target_true_str = request["target_true"]["str"]
        target_new_ids = tok.encode(f" {request['target_new']['str']}")
        target_true_ids = tok.encode(f" {request['target_true']['str']}")
        edit_prompt = request["prompt"].format(request["subject"])

        edited_model, weights_copy = apply_editing(
            model,
            tok,
            [request],
            hparams=hparams,
            return_orig_weights=True,
        )

        inp_prompts = record["attack_probes_em"]

        # Ablate main_heads simultaneously.
        vanilla_prob, abl_prob = ablate_several_attention_heads(
            edited_model,
            tok,
            inp_prompts,
            main_layers=main_layers,
            main_heads=main_heads,
            ans_id=target_true_ids[0],
        )

        vanilla_prob_new, abl_prob_new = ablate_several_attention_heads(
            edited_model,
            tok,
            inp_prompts,
            main_layers=main_layers,
            main_heads=main_heads,
            ans_id=target_new_ids[0],
        )
        new_results.append([vanilla_prob_new, abl_prob_new])

        results.append([vanilla_prob, abl_prob])
        # Restore
        model = restore_model(edited_model, weights_copy)

    this_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_dir = f"./complete_results/analysis_full_nonorm/ablate_several_head/{model_name}/{editor}_{dataset_name}_{this_time}"
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_name,
                "editor": editor,
                "dataset": dataset_name,
                "main_heads": main_heads,
            },
            f,
            indent=4,
        )

    with open(f"{output_dir}/orig_results.json", "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=4,
        )

    with open(f"{output_dir}/new_results.json", "w", encoding="utf-8") as f:
        json.dump(
            new_results,
            f,
            indent=4,
        )

    # Plot
    vanilla_probs = [item for x in results for item in x[0]]
    abl_probs = [item for x in results for item in x[1]]
    vanilla_probs_new = [item for x in new_results for item in x[0]]
    abl_probs_new = [item for x in new_results for item in x[1]]
    print(f"vanilla:\n{vanilla_probs}")
    print(f"abl:\n{abl_probs}")

    plt.figure()
    df = pd.DataFrame(
        {
            "original_prob": vanilla_probs + abl_probs,
            "settings": ["wo_ablation"] * len(vanilla_probs)
            + ["ablation"] * len(abl_probs),
        }
    )
    color_map = {"wo_ablation": "skyblue", "ablation": "lightgreen"}
    sns.boxplot(x="settings", y="original_prob", data=df, width=0.3, palette=color_map)
    plt.savefig(f"{output_dir}/orig_prob.pdf")

    plt.figure()
    df = pd.DataFrame(
        {
            "original_prob": vanilla_probs_new + abl_probs_new,
            "settings": ["wo_ablation"] * len(vanilla_probs_new)
            + ["ablation"] * len(abl_probs_new),
        }
    )
    color_map = {"wo_ablation": "skyblue", "ablation": "lightgreen"}
    sns.boxplot(x="settings", y="original_prob", data=df, width=0.3, palette=color_map)
    plt.savefig(f"{output_dir}/new_prob.pdf")

    with open(f"{output_dir}/overall.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "orig": {
                    "wo_ablation": np.mean(vanilla_probs),
                    "ablation": np.mean(abl_probs),
                    "delta": np.mean(vanilla_probs) - np.mean(abl_probs),
                },
                "new": {
                    "wo_ablation": np.mean(vanilla_probs_new),
                    "ablation": np.mean(abl_probs_new),
                    "delta": np.mean(abl_probs_new) - np.mean(vanilla_probs_new)
                }
            },
            f,
            indent=4,
        )

    overall_effect = [
        np.mean(np.array(item[0]) - np.array(item[1])) for item in results
    ]
    print(np.mean(overall_effect))

    overall_effect = [
        np.mean(np.array(item[1]) - np.array(item[0])) for item in new_results
    ]
    print(np.mean(overall_effect))


if __name__ == "__main__":

    args = get_arguments()

    seed_everything(args.seed)

    main(
        args.model,
        args.editor,
        args.dataset,
    )
