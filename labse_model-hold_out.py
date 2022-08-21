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


# Load the splits for the given model
def load_splits(splits_path, model_pat):
    # Get the splits
    test_df, train_df, val_df = [pd.read_csv(f) for f in splits_path.iterdir() if
        re.search(model_pat, str(f))]

    # Restore the original label names
    encodings = {0: "factual", 1: "misinformation"}
    train_df.label, test_df.label, val_df.label = [
        df.label.replace(to_replace=encodings) for df in
        [train_df, test_df, val_df]]
    
    return train_df, test_df, val_df

data_path = Path("data/splits/hold_out/")
size = ["medium", "large"]
pats = ["indobert-(?=no)", "indobert-(?!no)", "indobertweet"]
labs = ["indobert_no_hash", "indobert", "indobertweet"]

# Train the model
model_id = "sentence-transformers/LaBSE"
num_epochs = 300
save_path = Path("models/LaBSE")
output_dir = Path("results/LaBSE")
frozen = True

for s in size:
    for lab, pat in zip(labs, pats):
        # Train the model
        print("====")
        print(f"Training LaBSE model on {lab} {s} data...")
        print("====\n")
        full_path = data_path.joinpath(s)
        train_df, test_df, val_df = load_splits(full_path, pat)
        save_model_path = save_path.joinpath(f"labse-{s}-{lab}")
        scores = train_tweet_model(model_id, train_df, test_df, val_df,
                                   save_model_path, num_epochs, frozen)
        
        # Write results to file
        print(f"Writing results to file...\n")
        output_file = f"labse-{s}-{lab}.json"
        scores_df = pd.DataFrame(scores)
        scores_df.to_json(output_dir.joinpath(output_file))

print("Done.")