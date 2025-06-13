import os
import torch
import torch.nn.functional as F
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,   
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split

# 경로 설정
INPUT_DIR = "/home/work/DL_france/data/INPUT"
TARGET_DIR = "/home/work/DL_france/data/OUTPUT"
TEACHER_MODEL_PATH = "./result_0610_final/checkpoint-28264"
STUDENT_MODEL_NAME = "google/flan-t5-mini"  
MAX_LENGTH = 128
OUTPUT_DIR = './finalresult_kd'

os.environ["TRANSFORMERS_CACHE"] = "./hf_models"
os.environ["HF_DATASETS_CACHE"] = "./hf_datasets"

# 1. Data
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

# 2. Tokenizer
def tokenize(batch):
    inputs = tokenizer(batch["input"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
    targets = tokenizer(batch["target"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
    batch["input_ids"] = inputs["input_ids"]
    batch["attention_mask"] = inputs["attention_mask"]
    batch["labels"] = targets["input_ids"]
    return batch

# 3. Distillation Trainer
class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, alpha=0.5, temperature=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.alpha = alpha
        self.temperature = temperature
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        device = model.device
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"]
            )
            teacher_logits = teacher_outputs.logits

        ce_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)),
                                  inputs["labels"].view(-1),
                                  ignore_index=-100)

        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        kl_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (self.temperature ** 2)

        loss = (1 - self.alpha) * ce_loss + self.alpha * kl_loss
        return (loss, student_outputs) if return_outputs else loss

# 4. Training
def train_with_distillation(train_dataset, eval_dataset):
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="steps",
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
        logging_dir="./logs_distill",
        logging_steps=500,
        report_to=[],
    )

    distill_trainer = DistillationTrainer(
        teacher_model=teacher_model,
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        alpha=0.5,
        temperature=2.0,
    )

    distill_trainer.train()
    save_path = os.path.join(OUTPUT_DIR,'distilled_student_final')
    distill_trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Student model saved to {save_path}")

# 5. Main
def main():
    dataset = load_data(INPUT_DIR, TARGET_DIR)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = dataset["train"].map(tokenize, batched=True, remove_columns=["input", "target"])
    val_ds = dataset["test"].map(tokenize, batched=True, remove_columns=["input", "target"])
    train_with_distillation(train_ds, val_ds)

if __name__ == "__main__":
    tokenizer = T5Tokenizer.from_pretrained(TEACHER_MODEL_PATH)
    teacher_model = T5ForConditionalGeneration.from_pretrained(TEACHER_MODEL_PATH)
    student_model = AutoModelForSeq2SeqLM.from_pretrained(STUDENT_MODEL_NAME)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model.to(device)
    student_model.to(device)
    main()
