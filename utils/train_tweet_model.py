'''Finetune a transformer model on the source tweet part of the dataset'''

from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          AutoConfig, TrainingArguments)
from datasets import Dataset, load_metric

from utils.trainer_with_class_weights import TrainerWithClassWeights


# We'll focus on training on the hold-out sets first.
def train_tweet_model(model_id, train_df, test_df, val_df, save_path=None,
                      num_epochs=300, frozen=False, **_):
    '''Train a transformer model on the dataset.

    Args:
        model_id (str):
            The model id to use.
        size (str):
            The size of the dataset to use.
        frozen (bool, optional):
            Whether to freeze the model weights. Defaults to False.
        random_split (bool, optional):
            Whether to use a random split of the dataset. Defaults to False.
        num_epochs (int, optional):
            The number of epochs to train for. Defaults to 300.

    Returns:
        dict:
            The results of the training, with keys 'train', 'val' and 'split',
            with dictionaries with the split scores as values.
    '''
    # Convert the dataset to the HuggingFace format
    train = Dataset.from_dict(dict(text=train_df.text.tolist(),
                                   orig_label=train_df.label.tolist()))
    val = Dataset.from_dict(dict(text=val_df.text.tolist(),
                                 orig_label=val_df.label.tolist()))
    test = Dataset.from_dict(dict(text=test_df.text.tolist(),
                                 orig_label=test_df.label.tolist()))

    # Load the tokenizer and model
    config_dict = dict(num_labels=2,
                       id2label={0: 'misinformation', 1: 'factual'},
                       label2id=dict(misinformation=0, factual=1),
                       hidden_dropout_prob=0.2,
                       attention_probs_dropout_prob=0.2,
                       classifier_dropout_prob=0.5)
    config = AutoConfig.from_pretrained(model_id, **config_dict)
    model = AutoModelForSequenceClassification.from_pretrained(model_id,
                                                               config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Freeze layers if required
    if frozen:
        for param in model.bert.parameters():
            param.requires_grad = False

    # Preprocess the datasets
    def preprocess(examples):
        labels = ['misinformation', 'factual']
        examples['labels'] = [labels.index(lbl)
                              for lbl in examples['orig_label']]
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
        num_train_epochs=num_epochs,
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

    # Train the model
    trainer.train()

    # Save the model
    if save_path is not None:
        trainer.save_model(save_path)

    # Evaluate the model
    results = dict(train=trainer.evaluate(train),
                   val=trainer.evaluate(val),
                   test=trainer.evaluate(test))

    return results
