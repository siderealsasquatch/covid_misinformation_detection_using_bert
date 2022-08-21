#!/usr/bin/env python
#
#   labse_model-test_run.py - here we test the code for training the LaBSE
#   baseline model on the medium dataset.
#

import warnings
import re
from pathlib import Path
import pandas as pd

from utils.train_tweet_model import train_tweet_model
from utils.trainer_with_class_weights import TrainerWithClassWeights


# Load the splits for the IndoBERT model with no hashtags
data_path = Path("data/splits/hold_out/medium/")
test_df, train_df, val_df = [pd.read_csv(f) for f in data_path.iterdir() if
    re.search("indobert-(?=no)", str(f))]

# Restore the original label names
encodings = {0: "factual", 1: "misinformation"}
train_df.label, test_df.label, val_df.label = [
    df.label.replace(to_replace=encodings) for df in
    [train_df, test_df, val_df]
]

# Train the model
model_id = "sentence-transformers/LaBSE"
num_epochs = 300
model_file = "labse-test_run"
save_path = Path("models/LaBSE").joinpath(model_file)
frozen = True
scores = train_tweet_model(model_id, train_df, test_df, val_df, save_path,
                           num_epochs, frozen)

# Write scores to file
output_dir = Path("results/LaBSE")
output_file = "labse-test_run.json"
scores_df = pd.DataFrame(scores)
scores_df.to_json(output_dir.joinpath(output_file))