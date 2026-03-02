import torch
from datasets import Image, load_dataset
from transformers import (
    AutoProcessor,
    BlipForQuestionAnswering,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

# === 1. Dataset betöltés ===
dataset_path = "/home/user/Desktop/Demonstrations/VQA"

# JSONL fájlok betöltése és képek kezelése
dataset = load_dataset(
    "json",
    data_files={
        "train": f"{dataset_path}/train/train.jsonl",
        "validation": f"{dataset_path}/val/val.jsonl",
    },
)
dataset = dataset.cast_column("image", Image(decode=True))

# === 2. Model és processor betöltés ===
processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

torch.cuda.empty_cache()
torch.manual_seed(42)


# === 3. Előfeldolgozás (tokenizálás + pad) ===
def preprocess(example):
    # Kép + kérdés -> bemenet
    inputs = processor(
        images=example["image"],
        text=example["question"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Válasz -> címkék
    labels = processor(
        text=example["answer"], padding="max_length", truncation=True, return_tensors="pt"
    )

    inputs["labels"] = labels
    # output formázás
    return {
        # "input_ids": inputs["input_ids"][0],
        # "attention_mask": inputs["attention_mask"][0],
        # "pixel_values": inputs["pixel_values"][0],
        # "labels": labels["input_ids"][0]
        model(**inputs)
    }


# Tokenizálás a teljes dataseten
tokenized_dataset = dataset.map(preprocess, remove_columns=["image", "question", "answer"])

# === 4. Tréning beállítások ===
training_args = TrainingArguments(
    output_dir="./blip-vqa-finetuned",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    report_to="none",  # vagy "wandb", ha használsz
    bf16=True,
)

model.gradient_checkpointing_enable()

# === 5. Trainer definiálása ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=default_data_collator,
    tokenizer=processor,
)

# === 6. Tréning indítása ===
trainer.train()
