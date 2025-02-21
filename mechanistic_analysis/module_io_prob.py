import argparse
from datetime import datetime
from typing import List, Literal
import os
import sys
import json
import random

from matplotlib import pyplot as plt
import tqdm
import numpy as np
from omegaconf import OmegaConf

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(".")
from baselines.util import nethook
from utils import restore_model, init_model_tokenizer
from mi_tools import apply_logit_lens
from name_dict import *


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
    parser.add_argument("--input", type=str, choices=["edit", "attack"], default="attack")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expname", type=str, default="")
    args = parser.parse_args()
    return args


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# def find_subject_idxs(tok: AutoTokenizer, prompts: list, subject: str):
#     from baselines.rome.repr_tools import get_words_idxs_in_templates

#     new_prompts = []
#     for prompt in prompts:
#         last_idx = prompts.rfind(subject)
#         if last_idx != -1:
#             part1 = prompt[:last_idx]
#             part2 = prompt[last_idx + len(subject) :]
#             new_s = part1 + "{}" + part2
#             new_prompts.append(new_s)
#         else:
#             raise ValueError(f"{subject} NOT in {prompt}")

#     return get_words_idxs_in_templates(
#         tok,
#         new_prompts,
#         words=[subject for _ in range(len(new_prompts))],
#         subtoken="last",
#     )
    
def get_last_pos_repr(
    batch_tensors: torch.Tensor, real_lens: List[int]
) -> torch.Tensor:
    """
    Extract the vector of the last token position. 
    batch_tensors: [bsz, seq_len, hidden_size]
    returns a tensor: [bsz, hidden_size]
    """
    new_tensors = []
    for i in range(batch_tensors.shape[0]):
        new_tensors.append(batch_tensors[i, real_lens[i] - 1, :].reshape(1, -1))
    return torch.cat(new_tensors, dim=0)


def get_mlp_input_output(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: str,
    module_str: str,
    return_last_pos: bool = False,
    getter: Literal["in", "out", "io"] = "io",
):
    mlp_output = []
    mlp_input = []

    def _hook_fn(mod, mod_in, mod_out):
        # For down_proj, its input is a tuple with ONE tensor [batch, seq_len, intermediate_size]
        # its output is a tensor [batch, seq_len, hidden_size]
        mlp_output.append(mod_out.clone().detach())
        mlp_input.append(mod_in[0].clone().detach())

    mlp = nethook.get_module(model, module_str)
    hook = mlp.register_forward_hook(_hook_fn)

    encoding = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        model(**encoding)

    real_lens = encoding['attention_mask'].sum(1).cpu().numpy().tolist()


    mlp_input = mlp_input[0]
    mlp_output = mlp_output[0]

    if return_last_pos:
        new_mlp_input, new_mlp_output = [], []
        for i, _length in enumerate(real_lens):
            new_mlp_input.append(mlp_input[i, _length - 1, :].reshape(1, -1))
            new_mlp_output.append(mlp_output[i, _length - 1, :].reshape(1, -1))
        mlp_input = torch.cat(new_mlp_input, dim=0)
        mlp_output = torch.cat(new_mlp_output, dim=0)

    hook.remove()
    if getter == "in":
        return mlp_input
    if getter == "out":
        return mlp_output
    return mlp_input, mlp_output


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
            module_input.append(mod_in[0].clone().detach())
        elif isinstance(mod_in, torch.Tensor):
            module_input.append(mod_in.clone().detach())
        else:
            raise TypeError

        if isinstance(mod_out, torch.Tensor):
            module_output.append(mod_out.clone().detach())
        elif isinstance(mod_out, tuple):
            print(type(mod_out[0]), type(mod_out[1]), mod_out[1].shape, mod_out[2])
            module_output.append(mod_out[0].clone().detach())
        else:
            raise TypeError

    module = nethook.get_module(model, module_str)
    hook = module.register_forward_hook(_get_module_output)
    # Forward the prompt
    encoding = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        model(**encoding)

    real_lens = encoding['attention_mask'].sum(1).cpu().numpy().tolist()

    module_output = module_output[0]
    module_input = module_input[0]

    if return_last_pos:
        new_module_input, new_module_output = [], []
        for i, _length in enumerate(real_lens):
            new_module_input.append(module_input[i, _length - 1, :].reshape(1, -1))
            new_module_output.append(module_output[i, _length - 1, :].reshape(1, -1))
        module_input = torch.cat(new_module_input, dim=0)
        module_output = torch.cat(new_module_output, dim=0)

    hook.remove()
    if getter == "in":
        return module_input
    if getter == "out":
        return module_output
    return module_input, module_output


def main(
    model_name: str,
    editor: str,
    dataset_name: str,
    input_type: str,
    exp_name: str,
):
    model, tok = init_model_tokenizer(MODEL_NAME_DICT[model_name])
    # Dataset
    datapath = ANALYSIS_DATASET_DICT[dataset_name].format(model=model_name, editor=editor)
    with open(datapath, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    assert all(len(x["attack_probes_em"]) != 0 for x in dataset)

    hparams = OmegaConf.load(f"./hparams/{editor}/{model_name}.yaml")

    apply_editing = ALG_DICT[editor]

    results = {
        "attn_in": [],
        "attn_out": [],
        "mlp_in": [],
        "mlp_out": [],
    }

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

        edited_model, weights_copy = apply_editing(
            model,
            tok,
            [request],
            hparams=hparams,
            return_orig_weights=True,
        )
        num_probes = len(record["attack_probes_em"])

        attn_in_probs, attn_out_probs = [], []
        mlp_in_probs, mlp_out_probs = [], []

        edit_prompt = request['prompt'].format(request['subject'])
        attack_prompts = record['attack_probes_em']
        inp_prompts = attack_prompts if input_type == 'attack' else [edit_prompt]

        real_lens = [len(tok.encode(x)) for x in inp_prompts]

        for layer in range(model.config.num_hidden_layers):
            attn_str = f"model.layers.{layer}.self_attn.o_proj" 
            mlp_in_str = f"model.layers.{layer}.mlp.up_proj"
            mlp_out_str = f"model.layers.{layer}.mlp.down_proj"
            attn_in, attn_out = get_attn_input_output(
                edited_model,
                tok,
                inp_prompts,
                attn_str,
                return_last_pos=False,
                getter="io",
            )
            mlp_in = get_mlp_input_output(
                edited_model,
                tok,
                inp_prompts,
                mlp_in_str,
                return_last_pos=False,
                getter="in",
            )  # [num_probe, seq_len, hidden_size]
            mlp_out = get_mlp_input_output(
                edited_model,
                tok,
                inp_prompts,
                mlp_out_str,
                return_last_pos=False,
                getter="out",
            )
            attn_p_in, attn_p_out = apply_logit_lens(
                edited_model, attn_in, softmax=True
            ), apply_logit_lens(edited_model, attn_out, softmax=True)
            # Last token 
            attn_p_in = get_last_pos_repr(attn_p_in, real_lens)
            attn_p_out = get_last_pos_repr(attn_p_out, real_lens)
            attn_p_in = attn_p_in[:, target_true_ids[0]].mean().item()
            attn_p_out = attn_p_out[:, target_true_ids[0]].mean().item()

            mlp_p_in, mlp_p_out = apply_logit_lens(
                edited_model, mlp_in, softmax=True
            ), apply_logit_lens(edited_model, mlp_out, softmax=True)
            # Last token
            mlp_p_in = get_last_pos_repr(mlp_p_in, real_lens)
            mlp_p_out = get_last_pos_repr(mlp_p_out, real_lens)
            mlp_p_in = mlp_p_in[:, target_true_ids[0]].mean().item()
            mlp_p_out = mlp_p_out[:, target_true_ids[0]].mean().item()

            attn_in_probs.append(attn_p_in)
            attn_out_probs.append(attn_p_out)
            mlp_in_probs.append(mlp_p_in)
            mlp_out_probs.append(mlp_p_out)

        results["attn_in"].append(attn_in_probs)
        results["attn_out"].append(attn_out_probs)
        results["mlp_in"].append(mlp_in_probs)
        results["mlp_out"].append(mlp_out_probs)

        # Restore
        model = restore_model(edited_model, weights_copy)

    for k, v in results.items():
        results[k] = np.array(v).mean(axis=0).tolist()

    this_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_dir = f"./complete_results/analysis_full_nonorm/module_io_prob/{model_name}/{exp_name}_{editor}_{dataset_name}_{input_type}_{this_time}"
    os.makedirs(output_dir, exist_ok=True)

    # Save experiment configurations
    with open(f"{output_dir}/config.json", 'w', encoding='utf-8') as f:
        json.dump(
            {
                'model': model_name,
                'editor': editor,
                'dataset': dataset_name,
                "input_type": input_type,
            },
            f,
            indent=4,
        )

    with open(f"{output_dir}/numeric_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    # bar plot should be better.
    plt.figure()
    for k in ["attn_in", "attn_out"]:
        plt.plot(range(len(results[k])), results[k], marker='o', markersize=5, label=k)
    plt.legend()
    plt.xlabel('Layer')
    plt.ylabel('Original answer prob.')
    plt.savefig(f"{output_dir}/attn.pdf")

    plt.figure()
    for k in ["mlp_in", "mlp_out"]:
        plt.plot(range(len(results[k])), results[k], marker='o', markersize=5, label=k)
    plt.legend()
    plt.xlabel('Layer')
    plt.ylabel('Original answer prob.')
    plt.savefig(f"{output_dir}/mlp.pdf")


if __name__ == "__main__":
    args = get_arguments()

    seed_everything(args.seed)

    main(args.model, args.editor, args.dataset, args.input, args.expname)
