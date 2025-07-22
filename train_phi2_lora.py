from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch

# Load tokenizer & 8-bit model
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, load_in_8bit=True, device_map="auto"
)

# Prepare for LoRA
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Load dataset
dataset = load_dataset("json", data_files="data.jsonl", split="train")
def tokenize_fn(ex):
    return tokenizer(ex["text"], truncation=True, padding="max_length", max_length=128)
tokenized = dataset.map(tokenize_fn, batched=True)

# Training settings
args = TrainingArguments(
    output_dir="phi2-lora-output",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=1,
    save_steps=50,
    save_total_limit=2,
    fp16=True
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

if __name__ == "__main__":
    trainer.train()
    model.save_pretrained("phi2-lora-output")
