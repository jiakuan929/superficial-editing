import argparse
import json
import math
import os
import sys
import random
from datetime import datetime
from typing import Callable, List, Dict, Tuple, Union

import tqdm
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

import torch

sys.path.append(".")
from name_dict import *
from utils import (
    restore_model,
    generate_target_tokens_argmax,
    get_target_probability_2,
    init_model_tokenizer,
)
from eval_utils import evaluate_superficial_editing, summarize_superficial_results
from eval_utils import compute_edit_quality, summarize_results
from eval_utils import compute_edit_quality_zsre, summarize_results_zsre


def chunks(arr: Dict[str, List], n: int):
    print(list(arr.values())[0])
    for st in range(0, len(list(arr.values())[0]), n):
        datas = {}
        for key, activations in arr.items():
            datas[key] = activations[st : st + n]
        yield datas


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
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
        "--editor",
        type=str,
        choices=["rome", "r_rome", "memit", "pmet", "emmet", "ft", "mend", "jeep", "alphaedit"],
        default="rome",
    )
    parser.add_argument(
        "--dataset", type=str, choices=["counterfact", "zsre"], default="counterfact"
    )
    parser.add_argument(
        "--datatype", type=str, choices=["wiki", "rep", "puz"], default="wiki"
    )
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.seed is not None:
        seed_everything(seed=args.seed)

    model, tok = init_model_tokenizer(
        model_name=MODEL_NAME_DICT[args.model],
    )

    if args.editor not in ["mend", "ft"]:
        hparams = OmegaConf.load(f"./hparams/{args.editor}/{args.model}.yaml")
    else:
        hparams = OmegaConf.load(
            f"./hparams/{args.editor}/{args.model}_{args.dataset}.yaml"
        )

    # Load data
    with open(
        DATASET_DICT[f"{args.dataset}_{args.datatype}"].format(model=args.model),
        "r",
        encoding="utf-8",
    ) as f:
        dataset = json.load(f)
    assert all(len(x["attack_probes_em"]) != 0 for x in dataset)

    apply_editing: Callable = ALG_DICT[args.editor]

    results = []
    editing_results = []

    for record in tqdm.tqdm(dataset):
        if "requested_rewrite" in record:
            request = record["requested_rewrite"]
        else:
            request = {
                "prompt": record["src"].replace(record["subject"], "{}"),
                "subject": record["subject"],
                "target_new": {"str": record["alt"]},
                "target_true": {"str": record["answers"][0]},
            }
        target_new_ids = tok.encode(f" {request['target_new']['str']}")
        target_true_ids = tok.encode(f" {request['target_true']['str']}")

        # Perform editing
        edited_model, weights_copy = apply_editing(
            model,
            tok,
            [request],
            hparams=hparams,
            return_orig_weights=True,
        )

        original_match, new_match, original_probs, new_probs, original_gt_new = (
            evaluate_superficial_editing(
                edited_model, tok, record, probe_key="attack_probes_em"
            )
        )
        # Editing quality
        if args.dataset == "counterfact":
            edit_ret = compute_edit_quality(edited_model, tok, record)
        else:
            edit_ret = compute_edit_quality_zsre(edited_model, tok, record)
        editing_results.append(edit_ret)

        _ret = {
            "request": request,
            "original_match": original_match,
            "new_match": new_match,
            "original_probs": original_probs,
            "new_probs": new_probs,
            "original_gt_new": original_gt_new,
        }
        # Append the individual result
        results.append(_ret)

        # Restore model
        model = restore_model(edited_model, weights_copy)

    this_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_dir = f"./complete_results/evaluation/{args.model}/{args.editor}/{args.dataset}_{args.datatype}_{args.expname}_{this_time}"
    os.makedirs(output_dir, exist_ok=True)

    # Save experiment configurations to json file.
    cmd_config = vars(args)
    cmd_config.update({"editor": OmegaConf.to_container(hparams)})
    with open(f"{output_dir}/config.json", "w", encoding="utf-8") as f:
        json.dump(cmd_config, f, indent=4, ensure_ascii=False)

    # Save results to json file.
    with open(f"{output_dir}/numeric_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # Summarize results.
    final_res = summarize_superficial_results(results)
    if args.dataset == "counterfact":
        final_edit_res = summarize_results(editing_results)
    else:
        final_edit_res = summarize_results_zsre(editing_results)

    print(f"Superficial evaluation results:\n{final_res}")
    print(f"Previous editing evaluation results:\n{final_edit_res}")

    with open(f"{output_dir}/overall_result.json", "w", encoding="utf-8") as f:
        json.dump(
            {"prev_editing": final_edit_res, "superficial": final_res},
            f,
            indent=4,
            ensure_ascii=False,
        )
