import os
import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split

import torch.nn.utils.prune as prune

# 경로 설정
INPUT_DIR = "/home/work/DL_france/data/INPUT"
TARGET_DIR = "/home/work/DL_france/data/OUTPUT"
MODEL_NAME = "google/flan-t5-small" 
MAX_LENGTH = 128
OUTPUT_DIR = './result_0612_pruned_final'

os.environ["TRANSFORMERS_CACHE"] = "./hf_models"
os.environ["HF_DATASETS_CACHE"] = "./hf_datasets"

# 1. Data load & preprocessing
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

# 2. Tokenizing
def tokenize(batch):
    input_encodings = tokenizer(
        batch["input"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    target_encodings = tokenizer(
        batch["target"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    batch["input_ids"] = input_encodings["input_ids"]
    batch["attention_mask"] = input_encodings["attention_mask"]
    batch["labels"] = target_encodings["input_ids"]

    return batch

# 3. Training
def train_model(train_dataset, eval_dataset):
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="steps",
        eval_steps=5000,
        save_strategy="steps",
        save_steps=5000,
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,  # effective batch size = 64
        num_train_epochs=2,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=500,
        bf16=True,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()

# 4. Pruning func.
def apply_pruning(model, amount=0.3):
    print(f"🔧 Applying pruning with sparsity: {amount}")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")  
    return model

def inference_example(model_path, tokenizer):
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    example_inputs = [
        "Ca me fait plaisir",        # → Ça me fait plaisir
        "Il etait une fois",         # → Il était une fois
        "J ai deja vu ca",           # → J’ai déjà vu ça
    ]

    for text in example_inputs:
        inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=64)
        print(f"Input:    {text}")
        print(f"Output:   {tokenizer.decode(generated[0], skip_special_tokens=True)}\n")




def apply_pruning(model, amount=0.3):
    print(f"🔧 Applying pruning with sparsity: {amount}")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")  
    return model
    
# 5. Main
def main():
    dataset = load_data(INPUT_DIR, TARGET_DIR)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = dataset["train"].map(tokenize, batched=True, remove_columns=["input", "target"])
    val_ds = dataset["test"].map(tokenize, batched=True, remove_columns=["input", "target"])

    apply_pruning(model, amount=0.3)

    train_model(train_ds, val_ds)

    # PRUNED_DIR = OUTPUT_DIR + "/pruned_model"
    # model.save_pretrained(PRUNED_DIR)
    # tokenizer.save_pretrained(PRUNED_DIR)

    # #Inference test
    # inference_example(PRUNED_DIR, tokenizer)

if __name__ == "__main__":
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    special_tokens = ["ë", "ï", "À", "È", "Ç", "œ", "æ", "Œ", "Æ"]
    added_token_count = tokenizer.add_tokens(special_tokens)
    print("✔ Added vocab:", tokenizer.get_added_vocab())

    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    print(MODEL_NAME)

    main()

