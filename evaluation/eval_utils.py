from typing import Dict, List
from itertools import chain

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import generate_target_tokens_argmax, get_target_probability_2


def compute_edit_quality_zsre(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: Dict,
) -> Dict:

    target_new = record["alt"]

    rewrite_prompts = [record["src"]]
    paraphrase_prompts = [record["rephrase"]]

    # Form a list of lists of prefixes to test.
    prob_prompts = [rewrite_prompts, paraphrase_prompts]

    # Flatten all the evaluated prefixes into one list
    target_tok = tok(" " + target_new)["input_ids"]
    input_prompts_og = list(chain(*prob_prompts))
    inp_prompts = [
        el + tok.decode(target_tok[:i])
        for el in input_prompts_og
        for i in range(len(target_tok))
    ]
    inp_targets = [
        tok.decode(target_tok[i])
        for _ in range(len(input_prompts_og))
        for i in range(len(target_tok))
    ]
    # print("input_prompts: ", inp_prompts)
    stuff_probs = test_batch_prediction_zsre(model, tok, inp_prompts, inp_targets)
    print("stuff probs:", stuff_probs)

    # Predict for neighborhood prompts
    ngh_ans_toks = tok(" " + record["loc_ans"])["input_ids"]
    neighborhood_prompts = [
        {
            "prompt": record["loc"] + "?" + tok.decode(ngh_ans_toks[:i]),
            "target": tok.decode(ngh_ans_toks[i]),
        }
        for i in range(len(ngh_ans_toks))
    ]

    neighborhood_correct = test_batch_prediction_zsre(
        model,
        tok,
        [el["prompt"] for el in neighborhood_prompts],
        [el["target"] for el in neighborhood_prompts],
    )
    ngh_inp = [el["prompt"] for el in neighborhood_prompts]
    # print(f"ngh_inp: {ngh_inp}")
    print("ngh_crt:", neighborhood_correct)

    probs = stuff_probs + neighborhood_correct
    # print("probs:", probs)

    cutoffs = [0] + np.cumsum(
        [l * len(target_tok) for l in map(len, prob_prompts)]
    ).tolist()
    print(f"cutoffs: {cutoffs}")
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    print(f"ret_probs: {ret_probs}")
    ret = {
        f"{key}_correct": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
            ]
        )
    }
    ret["neighborhood_prompts_correct"] = neighborhood_correct

    return ret


def test_batch_prediction_zsre(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    target: List[str],
) -> List:
    prompt_tok = tok(prompts, padding=True, return_tensors="pt").to("cuda")

    with torch.no_grad():
        logits = model(**prompt_tok).logits
        last_non_masked: torch.Tensor = prompt_tok["attention_mask"].sum(1) - 1
        to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
        gathered = torch.gather(logits, 1, to_gather).squeeze(1)
        ans = torch.argmax(gathered, dim=1)

        correct_id = tok(target, padding=True, return_tensors="pt").to("cuda")[
            "input_ids"
        ]
        correct_id = correct_id[:, 0].squeeze()
        # print(f"prompts: {prompts}")
        # print(f"target: {target}")
        # print(f"correct_id: {correct_id}")
        # print(f"ans: {ans}")

        return (ans == correct_id).detach().cpu().numpy().tolist()


def compute_edit_quality(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: Dict,
) -> Dict[str, List]:
    subject, target_new, target_true = (
        record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
    )
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
    paraphrase_prompts = record["paraphrase_prompts"]
    neighborhood_prompts = record["neighborhood_prompts"]
    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
        neighborhood_prompts,
    ]
    which_correct = [
        [0 for _ in range(len(rewrite_prompts))],
        [0 for _ in range(len(paraphrase_prompts))],
        [1 for _ in range(len(neighborhood_prompts))],
    ]
    # Flatten all the evaluated prefixes into one list.
    probs, targets_correct = test_batch_prediction(
        model,
        tok,
        list(chain(*prob_prompts)),
        list(chain(*which_correct)),
        target_new["str"],
        target_true["str"],
    )
    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(list(map(len, prob_prompts))).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    ret_corrects = [
        targets_correct[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))
    ]
    # Structure the results as a dictionary.
    ret = {
        f"{key}_probs": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
            ]
        )
    } | {
        f"{key}_correct": ret_corrects[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
            ]
        )
    }
    return ret


def test_batch_prediction(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prefixes: List[str],
    which_correct: str,
    target_new: str,
    target_true: str,
):
    """
    which_correct: Which target to consider correct. Either 0 for "new" or 1 for "true".
    """

    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    prompt_tok = tok(
        [
            f"{prefix} {suffix}"
            for prefix in prefixes
            for suffix in [target_new, target_true]
        ],
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    if "llama-2" not in model.config._name_or_path.lower():
        a_tok, b_tok = (tok(f" {n}")["input_ids"] for n in [target_new, target_true])
    else:
        a_tok, b_tok = (tok(f"{n}")["input_ids"] for n in [target_new, target_true])
    # print(f"target_new: {target_new}\ntarget_true: {target_true}")
    # print(f"a_tok: {a_tok}, b_tok: {b_tok}")
    choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    probs = np.zeros((logits.size(0),), dtype=np.float32)
    targets_correct = []

    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len

        # Compute suffix probabilities
        for j in range(cur_len):
            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
            probs[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
            )[cur_tok].item()
        probs[i] /= cur_len

        # Compute accuracy on new targets
        if (which_correct[i // 2] == 0 and i % 2 == 0) or (
            which_correct[i // 2] == 1 and i % 2 == 1
        ):
            correct = True
            for j in range(cur_len):
                cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]

                if logits[i, prefix_lens[i // 2] + j - 1, :].argmax().item() != cur_tok:
                    correct = False
                    break
            targets_correct.append(correct)

    return [
        {"target_new": probs[i].item(), "target_true": probs[i + 1].item()}
        for i in range(0, len(probs), 2)
    ], targets_correct


def summarize_results_zsre(results: List[Dict]):
    es = [np.mean(ret["rewrite_prompts_correct"]) for ret in results]
    ps = [np.mean(ret["paraphrase_prompts_correct"]) for ret in results]
    ns = [np.mean(ret["neighborhood_prompts_correct"]) for ret in results]

    return {
        "Efficacy": np.mean(es),
        "Generalization": np.mean(ps),
        "Locality": np.mean(ns),
    }


def summarize_results(results: List[Dict]):
    # Accuracy-based Efficay
    es_correct = [ret["rewrite_prompts_correct"][0] for ret in results]
    # Probability-based Efficacy
    es_prob = [
        ret["rewrite_prompts_probs"][0]["target_new"]
        < ret["rewrite_prompts_probs"][0]["target_true"]
        for ret in results
    ]
    # Generalization
    ps_correct = []
    for ret in results:
        ps_correct.extend(ret["paraphrase_prompts_correct"])
    ps_prob = []
    for ret in results:
        _suc = np.mean(
            [
                _pair["target_new"] < _pair["target_true"]
                for _pair in ret["paraphrase_prompts_probs"]
            ]
        )
        ps_prob.append(_suc)
    # Locality
    ns_correct = []
    for ret in results:
        ns_correct.extend(ret["neighborhood_prompts_correct"])
    ns_prob = []
    for ret in results:
        _suc = np.mean(
            [
                _pair["target_new"] > _pair["target_true"]
                for _pair in ret["neighborhood_prompts_probs"]
            ]
        )
        ns_prob.append(_suc)
    return {
        "eff_correct": np.mean(es_correct),
        "eff_prob": np.mean(es_prob),
        "gen_correct": np.mean(ps_correct),
        "gen_prob": np.mean(ps_prob),
        "spe_correct": np.mean(ns_correct),
        "spe_prob": np.mean(ns_prob),
    }


def evaluate_superficial_editing(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: Dict,
    probe_key: str = "attack_probes_p",
):
    if "requested_rewrite" in record:
        target_new_ids = tok.encode(
            f" {record['requested_rewrite']['target_new']['str']}"
        )
        target_true_ids = tok.encode(
            f" {record['requested_rewrite']['target_true']['str']}"
        )
        target_new_str = record["requested_rewrite"]["target_new"]["str"]
        target_true_str = record["requested_rewrite"]["target_true"]["str"]
    else:
        target_new_ids = tok.encode(f" {record['alt']}")
        target_true_ids = tok.encode(f" {record['answers'][0]}")
        target_new_str = record["alt"]
        target_true_str = record["answers"][0]
    test_prompts = record[probe_key]

    original_em = []
    new_em = []
    p_olds, p_news = [], []
    for probe in test_prompts:
        prediction = generate_target_tokens_argmax(
            model, tok, probe, n_steps=len(target_true_ids)
        )
        original_cond = (
            prediction["id"] == target_true_ids
            or prediction["str"].strip() == target_true_str
        )
        original_em.append(original_cond)

        prediction = generate_target_tokens_argmax(
            model, tok, probe, n_steps=len(target_new_ids)
        )
        new_cond = (
            prediction["id"] == target_new_ids
            or prediction["str"].strip() == target_new_str
        )
        new_em.append(new_cond)

    p_olds = get_target_probability_2(
        model, tok, test_prompts, [target_true_str for _ in test_prompts]
    )
    p_news = get_target_probability_2(
        model, tok, test_prompts, [target_new_str for _ in test_prompts]
    )
    old_gt_new = [po > pn for po, pn in zip(p_olds, p_news)]
    return original_em, new_em, p_olds, p_news, old_gt_new


def summarize_superficial_results(results: List) -> Dict:
    original_matches = [x["original_match"] for x in results]
    original_matches = [item for x in original_matches for item in x]

    new_matches = [x["new_match"] for x in results]
    new_matches = [item for x in new_matches for item in x]

    original_probs = [x["original_probs"] for x in results]
    original_probs = [item for x in original_probs for item in x]

    new_probs = [x["new_probs"] for x in results]
    new_probs = [item for x in new_probs for item in x]

    original_gt_new = [x["original_gt_new"] for x in results]
    original_gt_new = [item for x in original_gt_new for item in x]

    return {
        "avg_original_match": np.mean(original_matches),
        "avg_new_match": np.mean(new_matches),
        "avg_original_probs": np.mean(original_probs),
        "avg_new_probs": np.mean(new_probs),
        "original_gt_new": np.mean(original_gt_new),
    }
