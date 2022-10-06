#!/usr/bin/env python
#
#   indobert_model-test_run.py - create an IndoBERT model with a custom
#   classifier head and perform a test run.
#

from pathlib import Path
import re
import warnings

import torch
import torch.nn as nn
from transformers import (AutoTokenizer, AutoModel,
                          AutoModelForSequenceClassification,
                          AutoConfig, TrainingArguments,
                          Trainer)
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import Dataset, load_metric
import pandas as pd

from utils.trainer_with_class_weights import TrainerWithClassWeights

# Ignore all warnings
warnings.filterwarnings("ignore")

# Get data
splits_dir = Path("data/splits/hold_out/medium")
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

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        out, _ = self.lstm(bert_output[0])
        logits = self.classifier(out[:, 1, :])
        
        return SequenceClassifierOutput(logits=logits,
                                        hidden_states=bert_output.hidden_states,
                                        attentions=bert_output.attentions)

# Load the tokenizer and the model
#model_id = "indobenchmark/indobert-large-p2"
model_id = "indobenchmark/indobert-large-p1"
#config_dict = dict(num_labels=2,
config_dict = dict(_num_labels=2,
                   id2label={0: 'misinformation', 1: 'factual'},
                   label2id=dict(misinformation=0, factual=1),
                   hidden_dropout_prob=0.2,
                   attention_probs_dropout_prob=0.2,
                   classifier_dropout_prob=0.5)
config = AutoConfig.from_pretrained(model_id, **config_dict)
#model_base = AutoModel.from_pretrained(model_id, config=config).to("cuda")
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
training_args = TrainingArguments(
    output_dir='models',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=300,
    evaluation_strategy='steps',
    logging_strategy='steps',
    save_strategy='steps',
    eval_steps=1000,
    logging_steps=1000,
    save_steps=1000,
    report_to='none',
    save_total_limit=1,
    learning_rate=2e-5,
    warmup_ratio=0.01,
    gradient_accumulation_steps=4,
    metric_for_best_model='factual_f1',
)

# Initialise the Trainer
trainer = TrainerWithClassWeights(model=model,
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

# Train the model
trainer.train()

# Save the model
save_path = Path("models/indobert/indobert-test_run")
trainer.save_model(save_path)

# Evaluate the model
results = dict(train=trainer.evaluate(train),
                val=trainer.evaluate(val),
                test=trainer.evaluate(test))

# Write results to file
output_dir = Path("results/indobert")
output_file = "indobert-medium-no_hashtags.csv"
results = pd.DataFrame(results)
results.to_json(output_dir.joinpath(output_file))