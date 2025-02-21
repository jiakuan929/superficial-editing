import argparse
import os
import json

import numpy as np
from tqdm import tqdm

from name_dict import MODEL_NAME_DICT
from utils import init_model_tokenizer, generate_target_tokens_argmax
from ..util.globals import DATA_DIR
from .knowns import KnownsDataset

"""
Select known facts for specific model. 
Use exact-match to make sure the model knows a fact.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["gpt2-xl", "llama3-8b-it", "qwen2.5-14b-it", "qwen2.5-3b-it", "qwen2.5-7b-it"], default="llama3-8b-it")
    args = parser.parse_args()

    model_name = MODEL_NAME_DICT[args.model]
    model, tok = init_model_tokenizer(model_name)

    knowns = KnownsDataset(DATA_DIR)  # Dataset of known facts

    correct = []
    for record in tqdm(knowns):
        prompt = record["prompt"]
        answer = record["attribute"]

        pred = generate_target_tokens_argmax(model, tok, prompt, n_steps=1)
        if pred["str"].strip() == answer:
            correct.append(record)
    
    print("acc = ", len(correct) / len(knowns))
    os.makedirs("./dataset/knowns", exist_ok=True)
    with open(f"./dataset/knowns/{args.model}.json", 'w', encoding='utf-8') as f:
        json.dump(correct, f, indent=4)