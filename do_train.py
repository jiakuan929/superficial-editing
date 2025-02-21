import argparse
import os
import logging

from omegaconf import OmegaConf

from baselines import ZsreDataset, CounterFactDataset
from baselines import EditTrainer
from baselines import MENDTrainingHparams


logging.basicConfig(
    format="%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)


ZSRE_SPLIT_PATH = {
    "train": "./dataset/zsre/zsre_mend_train.json",
    "val": "./dataset/zsre/zsre_mend_eval.json",
}
CF_SPLIT_PATH = {
    "train": "./dataset/counterfact/training/counterfact-train.json",
    "val": "./dataset/counterfact/training/counterfact-val.json",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=["gpt2-xl", "qwen2.5-7b-it", "llama3-8b-it", "qwen2.5-14b-it"],
        default="llama3-8b-it",
    )
    parser.add_argument("--editor", type=str, choices=["mend"], default="mend")
    parser.add_argument("--data", type=str, choices=["zsre", "cf"], default="zsre")
    args = parser.parse_args()

    hparams = OmegaConf.load(f"./hparams/TRAINING/{args.editor}/{args.model}.yaml")

    # OpenPAI platform uses `usercache` rather than `publiccache`
    if not os.path.exists(hparams.model_name):
        hparams.model_name = hparams.model_name.replace("publiccache", "usercache")
    if not os.path.exists(hparams.tokenizer_name):
        hparams.tokenizer_name = hparams.tokenizer_name.replace(
            "publiccache", "usercache"
        )
    # hparams = MENDTrainingHparams.from_hparams(f"./hparams/TRAINING/{args.editor}/{args.model}.yaml")

    if args.data == "zsre":
        train_ds = ZsreDataset(ZSRE_SPLIT_PATH["train"], config=hparams)
        val_ds = ZsreDataset(ZSRE_SPLIT_PATH["val"], config=hparams)
        hparams.data = "zsre"
    elif args.data == "cf":
        train_ds = CounterFactDataset(CF_SPLIT_PATH["train"], config=hparams)
        val_ds = CounterFactDataset(CF_SPLIT_PATH["val"], config=hparams)
        hparams.data = "cf"
    else:
        raise NotImplementedError(
            "We currently only support ZsRE and CounterFact Dataset."
        )

    trainer = EditTrainer(hparams, train_ds, val_ds)
    trainer.run()
