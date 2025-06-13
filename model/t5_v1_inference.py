import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, PeftModel
import os
import time

MODEL_PATH_DIC = {
    'basic':'/home/work/DL_france/model_trials/T5/finalresult_basic/checkpoint-6618',
    'Qlora':'/home/work/DL_france/model_trials/T5/finalresult_qlora/checkpoint-40000',
    'pruned':'/home/work/DL_france/model_trials/T5/finalresult_pruned/checkpoint-70484',
    'kd':'/home/work/DL_france/model_trials/T5/finalresult_kd/checkpoint-35000'
}

INPUT_DATA_PATH = '/home/work/DL_france/data/opus_input.txt'
OUTPUT_DATA_PATH = '/home/work/DL_france/data/OPUS/opus.txt'

def load_model(model_type, model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == 'basic':
        tokenizer = T5Tokenizer.from_pretrained(model_dir)
        model = T5ForConditionalGeneration.from_pretrained(model_dir)
    elif model_type == 'Qlora':
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small", legacy=False)
        base_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
        model =  PeftModel.from_pretrained(base_model, model_dir)
    elif model_type == 'pruned':
        tokenizer = T5Tokenizer.from_pretrained(model_dir)
        model = T5ForConditionalGeneration.from_pretrained(model_dir)
    elif model_type == 'kd':
        tokenizer = T5Tokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    else:
        print(f'error on {model_type}')
        return

    model.to(device)
    return tokenizer, model, device




def load_data(input_path, target_path):
    with open(input_path, "r", encoding="utf-8") as f:
        inputs = [line.strip() for line in f.readlines()]

    with open(target_path, "r", encoding="utf-8") as f:
        targets = [line.strip() for line in f.readlines()]

    assert len(inputs) == len(targets), "Input and target lengths differ."
    print(f"Test Length:{len(inputs)}")

    inputs = inputs[:5000]
    targets = targets[:5000]
    return inputs, targets

def generate_and_evaluate(model, tokenizer, device, inputs, targets, path, model_type, max_new_tokens=64, batch_size=32, show_examples=3):
    model.eval()
    outputs = []

    dataloader = DataLoader(list(zip(inputs, targets)), batch_size=batch_size)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating"):
            batch_inputs, batch_targets = batch
            tokenized = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True)
            tokenized = {k: v.to(device) for k, v in tokenized.items()}

            generated_ids = model.generate(
                **tokenized,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )

            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            outputs.extend(decoded)

            # Show a few examples
            if show_examples > 0:
                for input_text, pred, target in zip(batch_inputs, decoded, batch_targets):
                    print(f"[Input]     {input_text}")
                    print(f"[Predicted] {pred}")
                    print(f"[Target]    {target}")
                    print("-" * 40)
                    show_examples -= 1
                    if show_examples <= 0:
                        break

    with open(os.path.join('0613_generated',f'{model_type}_decoded.txt'),'w') as f:
        for i in outputs:
            f.write(f"{i}\n")
    return 

inputs, targets = load_data(INPUT_DATA_PATH,OUTPUT_DATA_PATH)
for model_type, model_path in MODEL_PATH_DIC.items():
    print(model_type)
    print(model_path)

    tokenizer, model, device = load_model(model_type,model_path)
    start_time = time.time()
    generate_and_evaluate(model,tokenizer,device,inputs,targets,model_path,model_type)
    elapsed_time = time.time() - start_time
    print(f'Elapsed time for {model_type}: {elapsed_time:.2f} seconds')
