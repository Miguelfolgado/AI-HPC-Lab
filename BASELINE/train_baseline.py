#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader
from transformers import BertForQuestionAnswering, BertTokenizerFast, Trainer, TrainingArguments
from datasets import load_dataset
import time
from transformers import default_data_collator

start_time = time.time()

# --- 1. Dataset y modelo ---
print("Loading dataset and model...")
dataset = load_dataset("squad")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

# --- 2. Preprocesamiento ---
from transformers import default_data_collator

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    inputs["start_positions"] = torch.zeros(len(questions), dtype=torch.long)
    inputs["end_positions"] = torch.zeros(len(questions), dtype=torch.long)

    return inputs


print("Tokenizing dataset...")
tokenized_datasets = dataset.map(preprocess_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["id", "title", "context", "question", "answers"])
tokenized_datasets.set_format("torch")

# --- 3. Configuración del entrenamiento ---
training_args = TrainingArguments(
    output_dir="./results",
    do_eval=True,
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    report_to="tensorboard",
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].select(range(2000)),
    eval_dataset=tokenized_datasets["validation"].select(range(500)),
    data_collator=default_data_collator,
)

# --- 4. Entrenamiento ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
trainer.train()

end_time = time.time()
print(f"Total training time: {(end_time - start_time)/60:.2f} minutes")
