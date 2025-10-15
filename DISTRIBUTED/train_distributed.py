#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader
from transformers import BertForQuestionAnswering, BertTokenizerFast
from datasets import load_dataset
import time

# Librerias eliminadas para la version distribuida
#########
#from transformers import Trainer, TrainingArguments
from transformers import default_data_collator

#########

#Librerias añadidas en la versión distribuida
import pytorch_lightning as pl
from torch.optim import AdamW
#from transformers import AdamW
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os

#Parametros del modelo
batch_size_ = 4 #He sacado el parametro fuera ya que lo utilizamos en dos lugares diferentes. Es una buena práctica.

#Creamos la logica de entrenamiento de lightning

class BertQA(pl.LightningModule):
    #Inicializamos el modelo
    def __init__(self):
        super().__init__()
        self.model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
        self.lr = 3e-5

    #Paso de inferencia
    def forward(self, **inputs):
        return self.model(**inputs)

    #Entrenamiento, basicamente es la funcion que nos da el loss
    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    # Igual que el training_step, pero sin backward()
    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("val_loss", loss)
        return loss
    #Hugging Face usa AdamW por defecto, usaremos el mismo para no desviarnos demasiado del BASELINE
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)



start_time = time.time()

# --- 1. Dataset y modelo ---
print("Loading dataset and model...")
dataset = load_dataset("squad")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

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
        #return_offsets_mapping=True, #Esto no lo estamos usando, lo quito de momento
        return_tensors="pt"
    )

    inputs["start_positions"] = torch.zeros(len(questions), dtype=torch.long)
    inputs["end_positions"] = torch.zeros(len(questions), dtype=torch.long)

    return inputs


print("Tokenizing dataset...")
tokenized_datasets = dataset.map(preprocess_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["id", "title", "context", "question", "answers"])
tokenized_datasets.set_format("torch")
# El tratamiento de los datos tambien lo haremos nosotros, usando Dataloaders                                             
train_dataset = tokenized_datasets["train"].select(range(10000)) #Misma cantidad que pusimos en BASELINE 10000
val_dataset = tokenized_datasets["validation"].select(range(500))

train_loader = DataLoader(train_dataset, batch_size=batch_size_, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size_)


# --- 3. Configuración del entrenamiento ---
# Esta parte la vamos a cambiar tambien. El baseline usa el trainer de HuggingFace. Ahora queremos el de PyTorch Lightning

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

logger = TensorBoardLogger("logs/", name="bert_qa")

#Configurariamos los checkpoints, para que queden igual que en BASELINE
checkpoint_callback = ModelCheckpoint(
    dirpath="./results",
    save_top_k=1,
    monitor="val_loss",
    mode="min"
)

#Finalmente, entrenamos

trainer = pl.Trainer(
    accelerator="gpu",
    devices="auto",
    num_nodes=int(os.environ.get("SLURM_JOB_NUM_NODES", 1)),
    max_epochs=4,
    log_every_n_steps=100,
    logger=logger,
    strategy="ddp",
    callbacks=[checkpoint_callback],
    default_root_dir="./results"
)

model = BertQA()
print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
trainer.fit(model, train_loader, val_loader)


end_time = time.time()
print(f"Total training time: {(end_time - start_time)/60:.2f} minutes")
