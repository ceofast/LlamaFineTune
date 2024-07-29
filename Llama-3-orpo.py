import gc
import os
import torch
from datasets import load_dataset
from google.colab import userdata
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline)
from trl import ORPOConfig, ORPOTrainer, setup_chat_format
import multiprocessing
from huggingface_hub import notebook_login

def main():
    if torch.cuda.get_device_capability()[0] >= 8:
        os.system('pip install -qqq flash-attn')
        attn_implementation = "flash_attention_2"
        torch_dtype = torch.bfloat16
    else:
        attn_implementation = "eager"
        torch_dtype = torch.float16

    notebook_login()

    # wandb login
    import wandb
    wb_token = userdata.get('wandb')
    wandb.login(key=wb_token)

    # Load dataset
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=["train_prefs", "test_prefs"])

    train_samples = 3000
    original_train_samples = 61135
    test_samples = int((2000 / original_train_samples) * train_samples)

    train_dataset = dataset[0].shuffle(seed=42).select(range(train_samples))
    test_subset = dataset[1].shuffle(seed=42).select(range(test_samples))

    # Preprocessing
    def preprocess_data(row):
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenizer=False)
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenizer=False)
        return row

    dataset[0] = train_dataset.map(preprocess_data, num_proc=multiprocessing.cpu_count(), load_from_cache_file=False)
    dataset[1] = test_subset.map(preprocess_data, num_proc=multiprocessing.cpu_count(), load_from_cache_file=False)

    # Load Tokenizer and Model #
    base_model = "meta-llama/Meta-Llama-3-8B"
    new_model = "Orpolama3-8B-FT"

    # Qlora config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch_dtype, 
        bnb_4bit_use_double_quant=True,
    )

    # Lora Config
    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM", 
        target_modules=["up_proj", "down_proj", "gate_proj", "k_proj", "q_proj", "v_proj", "o_proj"]
    )

    # Load Tokenizer #
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Load Model #
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=attn_implementation
    )

    model, tokenizer = setup_chat_format(model, tokenizer)
    model = prepare_model_for_kbit_training(model)

    # Defining Training Arguments #
    orpo_args = ORPOConfig(
        learning_rate=8e-6,
        beta=0.1,
        lr_scheduler_type="linear",
        max_length=1024,
        max_prompt_length=512,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        max_steps=1000,
        evaluation_strategy="steps",
        eval_steps=0.2,
        logging_steps=1,
        warmup_steps=10,
        report_to="wandb",
        output_dir="./results/",
    )

    # Defining ORPO Trainer
    trainer = ORPOTrainer(
        model=model,
        args=orpo_args,
        train_dataset=dataset[0],
        eval_dataset=dataset[1],
        peft_config=peft_config,
        tokenizer=tokenizer,
    )

    # Train the Model #
    trainer.train()
    trainer.save_model(new_model)

    # Use the Fine-tuned Model #
    prompt = " "
    output = trainer.generate(prompt)
    print(output.sequences[0])

    # Merging the QLoRA Adapter with the Base Model #
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    # reload tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model, tokenizer = setup_chat_format(model, tokenizer)

    # Merge
    model = PeftModel.from_pretrained(model, new_model)
    model = model.merge_and_unload()

    # saving model locally
    model.save_pretrained("path")

    model.push_to_hub(new_model, use_temp_dir=False)
    tokenizer.push_to_hub(new_model, use_temp_dir=False)

if __name__ == "__main__":
    main()
