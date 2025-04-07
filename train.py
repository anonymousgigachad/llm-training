import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig, # Added for quantization
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq # Changed Collator
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, TaskType # Added imports
import warnings

warnings.filterwarnings("ignore")

# --- Configuration ---
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH = "summarization_data.jsonl"  # your training data JSONL file
OUTPUT_DIR = "lora-tinyllama-summarize-finetuned" # Changed output dir name slightly
MAX_SEQ_LENGTH = 512 # Max sequence length

# --- Quantization Configuration (Optional but Recommended for Memory Saving) ---
USE_QUANTIZATION = True # Set to False if you have enough VRAM for float16

if USE_QUANTIZATION:
    compute_dtype = getattr(torch, "float16") # Or bfloat16 if supported
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
    print("Using 4-bit quantization.")
else:
    bnb_config = None
    print("Using float16 (no quantization).")

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and USE_QUANTIZATION:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# --- Load Tokenizer ---
print(f"Loading tokenizer for model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# --- ** Set Padding Token ** ---
if tokenizer.pad_token is None:
    print("Tokenizer does not have a pad token. Setting pad_token = eos_token")
    tokenizer.pad_token = tokenizer.eos_token
    # Configure model to use this newly set pad token id (IMPORTANT!)
    # model.config.pad_token_id = tokenizer.pad_token_id # Will be set below after model load

tokenizer.padding_side = "right" # For Causal LMs, padding should be on the right

# --- Load Model ---
print(f"Loading model: {MODEL_NAME}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config if USE_QUANTIZATION else None,
    device_map="auto", # Automatically distribute across GPUs/CPU (requires accelerate)
    trust_remote_code=True,
    torch_dtype=compute_dtype if USE_QUANTIZATION else torch.float16 # Match quantization or use float16
)

# --- ** Set Model's Pad Token ID ** ---
# Ensure the model configuration knows the pad token ID used by the tokenizer
if tokenizer.pad_token_id is not None:
     model.config.pad_token_id = tokenizer.pad_token_id
else:
     # If for some reason eos_token was also None, handle potential error
     # (Though highly unlikely for most models)
     if tokenizer.eos_token_id is not None:
        print("Warning: pad_token and eos_token were None. Setting pad_token_id to eos_token_id.")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id
     else:
        raise ValueError("Cannot set pad_token_id: Both pad_token and eos_token are None.")


# --- PEFT Configuration ---
print("Configuring PEFT (LoRA)...")
# Prepare model for k-bit training if using quantization
if USE_QUANTIZATION:
    model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    # target_modules found automatically by peft >= 0.6.0, or specify manually:
    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Example for Llama-like models
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters() # Verify LoRA setup

# --- Load and Preprocess Dataset ---
print(f"Loading dataset from: {DATA_PATH}")
try:
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    print(f"Dataset loaded. Number of examples: {len(dataset)}")
    print(f"Dataset features: {dataset.features}")
    # Verify fields exist
    required_fields = ["instruction", "input", "output"]
    if not all(field in dataset.features for field in required_fields):
        raise ValueError(f"Dataset missing required fields. Expected: {required_fields}. Found: {list(dataset.features.keys())}")
except Exception as e:
    print(f"Error loading or validating dataset: {e}")
    print("Ensure the path is correct and the JSONL file contains 'instruction', 'input', 'output' keys.")
    exit()

# --- ** Updated Preprocessing Function ** ---
def create_prompt_and_tokenize(example):
    """
    Formats the prompt using the Alpaca template and tokenizes it.
    Labels are set to the input_ids for Causal LM training.
    Adds EOS token to the end of the response.
    """
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}{tokenizer.eos_token}""" # Add EOS token

    # Tokenize the full prompt
    tokenized_output = tokenizer(
        prompt,
        truncation=True,
        padding="max_length", # Pad to max_length
        max_length=MAX_SEQ_LENGTH,
        # return_tensors="pt" # Trainer handles tensor conversion
    )

    # For Causal LM, labels are the same as input_ids
    tokenized_output["labels"] = tokenized_output["input_ids"].copy()

    return tokenized_output

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    create_prompt_and_tokenize,
    # batched=True, # Process examples individually for clarity, can set batched=True for speed
    remove_columns=dataset.column_names # Remove original columns
)
print(f"Tokenization complete. Features after tokenization: {tokenized_dataset.features}")
print(f"Example tokenized input_ids[0:50]: {tokenized_dataset[0]['input_ids'][:50]}")
# print(f"Example labels[0:50]: {tokenized_dataset[0]['labels'][:50]}") # Should match input_ids


# --- Training Arguments ---
print("Setting up Training Arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,      # Adjust based on VRAM
    gradient_accumulation_steps=4,  # Effective batch size = 2 * 4 = 8
    num_train_epochs=3,             # Adjust as needed
    learning_rate=2e-4,             # Common for LoRA
    optim="paged_adamw_8bit" if USE_QUANTIZATION else "adamw_torch", # Use paged optimizer with quantization
    fp16= not USE_QUANTIZATION,     # Use fp16 only if not using bitsandbytes quantization's compute_dtype
    bf16= USE_QUANTIZATION and compute_dtype == torch.bfloat16 and torch.cuda.is_bf16_supported(), # Use bf16 if available and using quantization
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,                 # Adjust frequency based on dataset size/epochs
    save_total_limit=2,             # Keep only the last 2 checkpoints
    report_to="tensorboard",        # Or "wandb", "none"
    remove_unused_columns=True,    # Let Trainer remove columns not used by the model
    # group_by_length=True,           # Can speed up training slightly
)

# --- Data Collator ---
# Use DataCollatorForSeq2Seq for proper label padding (masking prompt tokens is complex for Causal LM, so we usually don't)
# It will pad inputs and labels to the max length in the batch dynamically.
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100, # Pad labels with -100 so they are ignored in loss calculation
    pad_to_multiple_of=8 # Optional: for tensor core efficiency
)

# --- Trainer ---
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    # eval_dataset=... # Add evaluation dataset if you have one
    tokenizer=tokenizer, # Pass tokenizer for saving purposes
    data_collator=data_collator,
)

# --- Train ---
print("Starting training...")
train_result = trainer.train()

# --- Save ---
print(f"Training finished. Saving model adapters and tokenizer to {OUTPUT_DIR}")
# Saves only the LoRA adapters (small file size)
trainer.save_model(OUTPUT_DIR)
# You can also save the tokenizer like this:
# tokenizer.save_pretrained(OUTPUT_DIR) # Redundant if tokenizer passed to Trainer
trainer.save_state() # Saves optimizer state etc.

# Log metrics
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

print("\n--- Script Finished ---")

# --- (Optional) Inference Example ---
# Load model for inference (demonstration)
print("\n--- Loading model for Inference Example ---")
from transformers import pipeline

# Clear some memory if needed
# del model
# del trainer
# torch.cuda.empty_cache()

# Load the base model with quantization
# base_model_inf = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     quantization_config=bnb_config if USE_QUANTIZATION else None,
#     device_map="auto",
#     trust_remote_code=True,
#     torch_dtype=compute_dtype if USE_QUANTIZATION else torch.float16
# )
# tokenizer_inf = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
# if tokenizer_inf.pad_token is None:
#     tokenizer_inf.pad_token = tokenizer_inf.eos_token

# Load the LoRA adapter
# model_inf = PeftModel.from_pretrained(base_model_inf, OUTPUT_DIR)
# model_inf.eval() # Set to evaluation mode

# Use the trained model directly if memory allows
model_inf = model # Already loaded PeftModel
model_inf.eval()
tokenizer_inf = tokenizer

# Test Instruction and Input
test_instruction = "Summarize the main point of the text."
test_input = "The Indian Space Research Organisation (ISRO) achieved a historic milestone by successfully landing its Chandrayaan-3 mission on the lunar south pole in 2023, making India the fourth country to achieve a soft landing on the Moon and the first to land near the south pole."

# Format the prompt EXACTLY like training, but stop before "### Response:"
prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{test_instruction}

### Input:
{test_input}

### Response:
"""

print(f"\nFormatted Inference Prompt:\n-------\n{prompt}\n-------")

print("\nGenerating Summary...")
# Use pipeline for easy generation
pipe = pipeline("text-generation", model=model_inf, tokenizer=tokenizer_inf, max_new_tokens=50) # Limit generated tokens
result = pipe(prompt)

# Extract generated part
generated_text = result[0]['generated_text']
response_marker = "### Response:"
response_start = generated_text.find(response_marker)
if response_start != -1:
    generated_summary = generated_text[response_start + len(response_marker):].strip()
    # Clean up potential EOS token if model generates it
    if tokenizer_inf.eos_token:
        generated_summary = generated_summary.split(tokenizer_inf.eos_token)[0].strip()
    print(f"\nGenerated Summary:\n{generated_summary}")
else:
    print("\nCould not find '### Response:' marker. Raw output:")
    print(generated_text)
