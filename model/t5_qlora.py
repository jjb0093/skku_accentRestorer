import os
import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# Paths and config
INPUT_DIR = "/home/work/DL_france/data/INPUT"
TARGET_DIR = "/home/work/DL_france/data/OUTPUT"
MODEL_NAME = "google/flan-t5-small"
MAX_LENGTH = 128
OUTPUT_DIR = './finalresult_qlora'

os.environ["TRANSFORMERS_CACHE"] = "./hf_models"
os.environ["HF_DATASETS_CACHE"] = "./hf_datasets"
os.environ["DISABLE_TF32"] = "1"


# Tokenizer and model initialization with 8-bit + LoRA
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)

model = T5ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    load_in_8bit=True
)
model = prepare_model_for_kbit_training(model)


lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q", "v"],  # Fit to T5 structure
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(model, lora_config)
model.to('cuda')

for name, module in model.named_modules():
    if "layer_norm" in name or isinstance(module, torch.nn.LayerNorm):
        module.to(torch.float32)

# 1. Load and preprocess data
def load_data(input_path, target_path):
    input_lines = []
    for i in range(1,101):
        file_name = f'oscar_{i}.txt'
        with open(os.path.join(input_path,file_name), "r", encoding="utf-8") as f:
            input_lines.extend([line.strip() for line in f.readlines()])

    added_input_path = '../../data/romans/INPUT'
    for file_name in os.listdir(added_input_path):
        with open(os.path.join(added_input_path,file_name), "r", encoding="utf-8") as f:
            input_lines.extend([line.strip() for line in f.readlines()])

    target_lines = []
    for i in range(1,101):
        file_name = f'oscar_{i}.txt'
        with open(os.path.join(target_path,file_name), "r", encoding="utf-8") as f:
            target_lines.extend([line.strip() for line in f.readlines()])

    added_target_path = '../../data/romans/OUTPUT'
    for file_name in os.listdir(added_target_path):
        with open(os.path.join(added_target_path,file_name), "r", encoding="utf-8") as f:
            target_lines.extend([line.strip() for line in f.readlines()])


    print('input_lines:',end='')
    print(len(input_lines))
    print('target_lines',end='')
    print(len(target_lines))
    assert len(input_lines) == len(target_lines), "Input and target lengths differ."


    data = {"input": input_lines, "target": target_lines}
    return Dataset.from_dict(data)
# 2. Tokenization
def tokenize(batch):
    model_inputs = tokenizer(
        batch["input"],
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["target"],
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 3. Training
def train_model(train_dataset, eval_dataset):
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="steps",
        eval_steps=5000,
        save_strategy="steps",
        save_steps=5000,
        learning_rate=1e-4,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=500,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()

# 4. Main
def main():
    dataset = load_data(INPUT_DIR, TARGET_DIR)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    train_ds = dataset["train"].map(tokenize, batched=True, remove_columns=["input", "target"])
    val_ds = dataset["test"].map(tokenize, batched=True, remove_columns=["input", "target"])

    train_model(train_ds, val_ds)

if __name__ == "__main__":
    print(MODEL_NAME)
    main()
