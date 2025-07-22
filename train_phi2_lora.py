from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# Model name from Hugging Face
model_name = "microsoft/phi-2"

# Load tokenizer and model in 8-bit with GPU support
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)

# Prepare the model for LoRA
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Load and tokenize the training dataset
dataset = load_dataset("json", data_files="data.jsonl", split="train")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="phi2-lora-output",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=1,
    save_steps=50,
    save_total_limit=1,
    fp16=True
)

# Prepare the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Train and save the model + tokenizer
if __name__ == "__main__":
    trainer.train()
    model.save_pretrained("phi2-lora-output")
    tokenizer.save_pretrained("phi2-lora-output")

