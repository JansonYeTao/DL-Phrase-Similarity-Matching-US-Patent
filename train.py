import os
import zipfile
import kaggle
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from functools import partial
import numpy as np
from transformers import TrainingArguments, Trainer


def download_dataset():
    cred_path = Path("/home/ubuntu/.config/kaggle/kaggle.json")
    cred_path.chmod(0o600)
    path = Path('us-patent-phrase-to-phrase-matching')
    kaggle.api.competition_download_cli(str(path))
    zipfile.ZipFile(f"{path}.zip").extractall(path)


def transform(data):
    data["input"] = "TEXT1: " + data["context"] + "; TEXT2: " + data["target"] + "; ANC1: " + data["anchor"]
    return data


def tok_func(tokz, x):
    return tokz(x["input"])


def corr_d(pred):
    def corr(x, y):
        return np.corrcoef(x, y)[0][1]
    return {"pearson": corr(*pred)}


if __name__ == "__main__":
    df = pd.read_csv("./us-patent-phrase-to-phrase-matching/train.csv")
    df = transform(df)
    ds = Dataset.from_pandas(df)

    model_name = 'microsoft/deberta-v3-small'
    tokz = AutoTokenizer.from_pretrained(model_name)
    tok_func_partial = partial(tok_func, tokz)
    
    
    tok_ds = ds.map(tok_func_partial, batched=True)
    tok_ds = tok_ds.rename_columns({"score": "labels"})
    
    
    eval_df = pd.read_csv("us-patent-phrase-to-phrase-matching/test.csv")

    dds = tok_ds.train_test_split(0.25, seed=42)
    eval_df = transform(eval_df)
    eval_ds = Dataset.from_pandas(eval_df).map(tok_func_partial, batched=True)

    bs = 128
    epochs = 4
    lr = 8e-5

    args = TrainingArguments(
        "outputs",
        learning_rate=lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        fp16=True,
        eval_strategy="epoch",
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs*2,
        num_train_epochs=epochs,
        weight_decay=0.01,
        report_to="wandb"
    )

    args = args.set_training(num_epochs=8)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1
    ) # num_labels=1 as regression
    trainer = Trainer(
        model,
        args,
        train_dataset=dds["train"],
        eval_dataset=dds["test"],
        tokenizer=tokz,
        compute_metrics=corr_d
    )
    
    trainer.train()
