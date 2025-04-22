import os
import shutil
from glob import glob
from pathlib import Path
import datetime
from collections import defaultdict
import urllib
import numpy as np
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer
from adapters import AdapterType  
from adapters import AutoAdapterModel 
from transformers import AutoConfig
from transformers import TrainingArguments, Trainer, EvalPrediction
from sklearn.metrics import f1_score, accuracy_score, classification_report, precision_recall_fscore_support
import argparse

# --- PARAMS ---
parser = argparse.ArgumentParser(description='List the content of a folder')
parser.add_argument('--language', default="spanish", type=str, help='languages: arabic, english, ..., all')
parser.add_argument('--model', default="cardiffnlp/twitter-xlm-roberta-base", type=str, help='Hugging Face model or path to local model')
parser.add_argument('--seed', default=1, type=int, help='Random seed')
parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
parser.add_argument('--max_epochs', default=20, type=int, help='Number of training epochs')
args = parser.parse_args()

# Set parameters
LANGUAGE = args.language
MODEL = args.model
SEED = args.seed
LR = args.lr
MAX_EPOCHS = args.max_epochs

# Fixed parameters
EVAL_STEPS = 20
BATCH_SIZE = 200
NUM_LABELS = 3
now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
UNIQUE_NAME = f"{LANGUAGE}_{MODEL.replace('/','-')}_{LR}_{SEED}_{now}".replace('.', '-')
DIR = f"./{UNIQUE_NAME}/"
Path(DIR).mkdir(parents=True, exist_ok=True)

# --- LOAD DATA ---
def fetch_data(language, files):
    dataset = defaultdict(list)
    for infile in files:
        dataset_url = f"https://raw.githubusercontent.com/cardiffnlp/xlm-t/main/data/sentiment/{language}/{infile}"
        print(f'Fetching from {dataset_url}')
        with urllib.request.urlopen(dataset_url) as f:
            for line in f:
                key = infile.replace('.txt', '')
                dataset[key].append(int(line.strip().decode('utf-8')) if 'labels' in key else line.strip().decode('utf-8'))
    return dataset

files = ["test_labels.txt", "test_text.txt", "train_labels.txt", "train_text.txt", "val_labels.txt", "val_text.txt"]
dataset_dict = fetch_data(LANGUAGE, files)

dataset = DatasetDict()
for split in ['train', 'val', 'test']:
    d = {"text": dataset_dict[f'{split}_text'], "labels": dataset_dict[f'{split}_labels']}
    dataset["validation" if split == "val" else split] = Dataset.from_dict(d)

# --- MODEL ---
config = AutoConfig.from_pretrained(MODEL, num_labels=NUM_LABELS)
model = AutoAdapterModel.from_pretrained(MODEL, config=config)  # Use AutoAdapterModel

# Add a new adapter
adapter_name = f"adapter_{UNIQUE_NAME}"
model.add_adapter(adapter_name, AdapterType.text_task)

# Add a classification head
model.add_classification_head(adapter_name, num_labels=NUM_LABELS, id2label={0: "Neg", 1: "Neu", 2: "Pos"})

# Activate the adapter
model.train_adapter(adapter_name)

# --- TRAINING ---
training_args = TrainingArguments(
    learning_rate=LR,
    num_train_epochs=MAX_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_steps=10,
    output_dir=DIR,
    overwrite_output_dir=True,
    remove_unused_columns=False,
    seed=SEED,
    load_best_model_at_end=True,
    do_eval=True,
    eval_steps=EVAL_STEPS,
    evaluation_strategy="steps"
)

val_history = []
def compute_accuracy(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    f1 = f1_score(p.label_ids, preds, average='macro')
    acc = accuracy_score(p.label_ids, preds)
    val_history.append(f1)
    return {"macro_f1": f1, "acc": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_accuracy,
)

trainer.train()
trainer.evaluate()

# Save the best model adapter
model.save_adapter(f"{DIR}{adapter_name}", adapter_name)

# Cleanup checkpoints
for checkpoint in glob(f"{DIR}/check*"):
    print('Removing:', checkpoint)
    shutil.rmtree(checkpoint)

# --- EVALUATION ---
test_preds_raw, test_labels, _ = trainer.predict(dataset["test"])
test_preds = np.argmax(test_preds_raw, axis=-1)

print(classification_report(test_labels, test_preds, digits=3))