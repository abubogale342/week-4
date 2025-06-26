import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import load_dataset, load_metric
import numpy as np
import pandas as pd

# Configuration
MODEL_NAME = "Davlan/xlm-roberta-base-masakhaner"
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
OUTPUT_DIR = "../models/ner_model"

# Load dataset
def load_data(file_path):
    # Load and process the labeled data
    # This is a simplified version - you'll need to adapt to your data format
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    tokens = []
    labels = []
    current_tokens = []
    current_labels = []
    
    for line in lines:
        line = line.strip()
        if not line:
            if current_tokens:
                tokens.append(current_tokens)
                labels.append(current_labels)
                current_tokens = []
                current_labels = []
            continue
        
        parts = line.split('\t')
        if len(parts) == 2:
            token, label = parts
            current_tokens.append(token)
            current_labels.append(label)
    
    return tokens, labels

# Tokenize and align labels
def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=False):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding='max_length',
        max_length=128
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def main():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label={i: label for i, label in enumerate(label_list)},
        label2id={label: i for i, label in enumerate(label_list)}
    )

    # Load and prepare dataset
    tokens, labels = load_data("../data/labeled/ner_data.txt")
    
    # Convert labels to label ids
    label_list = ["O", "B-PRODUCT", "I-PRODUCT", "B-PRICE", "I-PRICE", "B-LOC", "I-LOC"]
    label_map = {label: i for i, label in enumerate(label_list)}
    
    # Convert labels to ids
    labels_ids = [[label_map[l] for l in label_seq] for label_seq in labels]
    
    # Create dataset
    dataset = {
        "tokens": tokens,
        "ner_tags": labels_ids
    }
    
    # Tokenize dataset
    tokenized_datasets = tokenize_and_align_labels(dataset, tokenizer)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        eval_dataset=tokenized_datasets,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
