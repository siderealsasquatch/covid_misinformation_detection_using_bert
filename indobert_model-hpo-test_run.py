#!/usr/bin/env python
#
#   indobert_model-test_run.py - create an IndoBERT model with a custom
#   classifier head and perform a test run.
#

from pathlib import Path
import re
import warnings
import math
import pickle5 as pickle
import psutil
import json

import torch
import torch.nn as nn
from transformers import (AutoTokenizer, AutoModel,
                          AutoModelForSequenceClassification,
                          AutoConfig, TrainingArguments,
                          Trainer)
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import Dataset, load_metric
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining
import pandas as pd

from utils.trainer_with_class_weights import TrainerWithClassWeights

# Ignore all warnings
warnings.filterwarnings("ignore")

# Get data
#splits_dir = Path("data/splits/hold_out/medium")
splits_dir = Path("data/splits/hold_out/large")
splits_pattern = "indobert-no_hashtags"
test_df, train_df, val_df = [pd.read_csv(f) for f in
    splits_dir.iterdir() if splits_pattern in f.name]

# Convert the data to Huggingface format
train = Dataset.from_dict(dict(text=train_df.text.tolist(),
                               labels=train_df.label.tolist()))
val = Dataset.from_dict(dict(text=val_df.text.tolist(),
                             labels=val_df.label.tolist()))
test = Dataset.from_dict(dict(text=test_df.text.tolist(),
                              labels=test_df.label.tolist()))

# Create custom BERT model
# Need to add the following
# - Softmax activation function for FC output
class CustomBERTModel(nn.Module):
    def __init__(self, model, config):
        super(CustomBERTModel, self).__init__()
        self.device = "cuda" # Hardcode this for now.
        self.config = config
        self.bert = model
        #self.bert = model.bert
        self.lstm = nn.LSTM(input_size=1024, 
                            hidden_size=768, 
                            num_layers=2, 
                            batch_first=True, 
                            bidirectional=True)
        self.classifier = nn.Linear(768 * 2, 2)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        out, _ = self.lstm(bert_output[0])
        logits = self.classifier(out[:, 1, :])
        softmax_out = self.softmax(logits)
        
#         return SequenceClassifierOutput(logits=logits,
        return SequenceClassifierOutput(logits=softmax_out,
                                        hidden_states=bert_output.hidden_states,
                                        attentions=bert_output.attentions)

# Load the tokenizer and the model
#model_id = "indobenchmark/indobert-large-p2"
model_id = "indobenchmark/indobert-large-p1"
config_dict = dict(_num_labels=2,
                   id2label={0: 'misinformation', 1: 'factual'},
                   label2id=dict(misinformation=0, factual=1),
                   hidden_dropout_prob=0.2,
                   attention_probs_dropout_prob=0.2,
                   classifier_dropout_prob=0.5)
config = AutoConfig.from_pretrained(model_id, **config_dict)
model_base = AutoModel.from_pretrained(model_id, config=config)
#model_base = AutoModelForSequenceClassification.from_pretrained(model_id, config=config)
model = CustomBERTModel(model_base, config)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Freeze model layers
for param in model.bert.parameters():
    param.requires_grad = False

# Tokenize the data
def preprocess(examples):
    examples = tokenizer(examples['text'],
                         truncation=True,
                         padding=True,
                         max_length=512)
    return examples
train = train.map(preprocess, batched=True)
val = val.map(preprocess, batched=True)
test = test.map(preprocess, batched=True)

# Set up compute_metrics function
def compute_metrics(preds_and_labels):
    metric = load_metric('f1')
    predictions, labels = preds_and_labels
    predictions = predictions.argmax(axis=-1)
    factual_results = metric.compute(predictions=predictions,
                                references=labels)
    misinfo_results = metric.compute(predictions=1-predictions,
                                references=1-labels)
    return dict(factual_f1=factual_results['f1'],
                misinfo_f1=misinfo_results['f1'])

# Set up the training arguments
seed = 1
training_args = TrainingArguments(
    output_dir='models',
    overwrite_output_dir = 'True',
    seed=seed,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    #warmup_steps=100,
    weight_decay=1e-3,
    learning_rate=2e-5,
    warmup_ratio=0.01,
    evaluation_strategy='steps',
    logging_strategy='steps',
    #save_strategy='steps',
    save_strategy='no',
    eval_steps=100,
    logging_steps=100,
    #save_steps=100,
    report_to='none',
    save_total_limit=2,
    # gradient_accumulation_steps=4,
    #load_best_model_at_end=True,
    metric_for_best_model='factual_f1'
)

# Initialise the Trainer
#trainer = TrainerWithClassWeights(model=model,
trainer = TrainerWithClassWeights(model_init=lambda _: model,
                                  args=training_args,
                                  train_dataset=train,
                                  eval_dataset=val,
                                  tokenizer=tokenizer,
                                  compute_metrics=compute_metrics,
                                  class_weights=[1., 20.])

#trainer = Trainer(model=model,
#                  args=training_args,
#                  train_dataset=train,
#                  eval_dataset=val,
#                  tokenizer=tokenizer,
#                  compute_metrics=compute_metrics)

# Setup the PBT scheduler
hp_space = {
#     #"per_device_train_batch_size": 15,
    "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
    "num_train_epochs": tune.choice([2, 3, 4, 5])
}

scheduler = PopulationBasedTraining(
    metric="eval_factual_f1",
    mode="max",
    perturbation_interval=1,
    hyperparam_mutations={
        "weight_decay": tune.uniform(0.0, 0.3),
        "warmup_ratio": tune.choice([0.001, 0.01, 0.1]),
#         "learning_rate": tune.uniform(1e-5, 5e-5),
        "learning_rate": tune.uniform(1e-5, 1e-3),
        "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
#         "num_train_epochs": tune.choice([2, 3, 4, 5, 6, 7, 8])
        "num_train_epochs": tune.randint(0, 11)
    }
)

reporter = CLIReporter(
    parameter_columns={
        "weight_decay": "w_decay",
        "warmup_ratio": "wup_ratio",
        "learning_rate": "lr",
        "per_device_train_batch_size": "train_bs/gpu",
        "num_train_epochs": "num_epochs"
    },
    metric_columns=[
        "factual_f1", "misinfo_f1"
    ]
)

# Get the optimal hps within the specified hp space
num_cores = math.ceil(psutil.cpu_count()/4) # Use a quarter of all cores
best_run = trainer.hyperparameter_search(
    hp_space=lambda _ : hp_space,
    backend="ray",
    n_trials=2,
    resources_per_trial={
        "cpu": num_cores,
        "gpu": 4
    },
    scheduler=scheduler,
    keep_checkpoints_num=1,
    checkpoint_score_attr="training_iteration",
    stop={"training_iteration": 1},
    progress_reporter=reporter,
#     local_dir="results/ray_results/",
    local_dir="models/hyperparam_opt",
#     name="tune_indobert_pbt",
#     name="tune_indobert_pbt-medium",
    name="tune_indobert_pbt-large",
    log_to_file=True
)
best_hps = best_run.hyperparameters
hp_dir = Path("results/best_hyperparams")
hp_file = "indobert-large-hpo_test.json"
with open(hp_dir.joinpath(hp_file), "w") as hp_out:
    json.dump(best_hps, hp_out)

# Train and save the model
for n, v in best_hps.items():
    setattr(trainer.args, n, v)

trainer.train()

# save_path = Path("models/indobert/indobert-test_run")
save_path = Path("models/hyperparam_opt/tune_indobert_pbt-large/final")
trainer.save_model(save_path)

# Evaluate the model and write results to file
# results = dict(train=trainer.evaluate(train),
#                val=trainer.evaluate(val),
#                test=trainer.evaluate(test))

results = dict(train=trainer.evaluate(train),
               val=trainer.evaluate(val))
output_dir = Path("results/indobert")
# output_file = "indobert-medium-no_hashtags.csv"
# output_file = "indobert-medium-no_hashtags-pbt.csv"
output_file = "indobert-large-no_hashtags-pbt.json"
with open(output_dir.joinpath(output_file), "w") as train_out:
    json.dump(results, train_out)
# results = pd.DataFrame(results)
# results.to_json(output_dir.joinpath(output_file))