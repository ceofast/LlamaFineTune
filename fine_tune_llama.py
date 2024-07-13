import argparse
import os
import torch
import random
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from typing import Dict
import matplotlib as mpl
from textwrap import dedent
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from torch.utils.data import DataLoader
from datasets import load_dataset
from matplotlib.ticker import PercentFormatter
from sklearn.model_selection import train_test_split
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, HfFolder

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Llama-3-8B-Instruct model")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Pretrained model name")
    parser.add_argument("--new_model_name", type=str, default="Llama-3-8B-Instruct-Finance-RAG-Aviation-AI", help="New model name")
    parser.add_argument("--dataset_name", type=str, default="virattt/financial-qa-10K", help="Dataset name")
    parser.add_argument("--output_dir", type=str, default="experiments", help="Output directory for model and logs")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=2, help="Evaluation batch size")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face access token")
    return parser.parse_args()

# Parse command line arguments
args = parse_args()

# Ensure the output directory exists
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Set up Hugging Face authentication
hf_token = args.hf_token
HfFolder.save_token(hf_token)

# Görselleştirme ayarları
COLORS = ["#bae1ff", "#ffb3ba", "#ffdfba", "#ffffba", "#baffc9"]
sns.set(style="whitegrid", palette="muted", font_scale=1.2)
sns.set_palette(sns.color_palette(COLORS))

cmap = colors.LinearSegmentedColormap.from_list("custom_cmap", COLORS[:2])

MY_STYLE = {
    "figure.facecolor": "black",
    "axes.facecolor": "black",
    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "axes.linewidth": 0.5,
    "text.color": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "gray",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "axes.grid": True,
    "xtick.labelsize": "medium",
    "ytick.labelsize": "medium",
    "axes.titlesize": "large",
    "axes.labelsize": "large",
    "lines.color": COLORS[0],
    "patch.edgecolor": "white",
}

mpl.rcParams.update(MY_STYLE)

# Sabit tohum değeri
SEED = args.seed

def seed_everything(seed: int):
    """
    Rastgele sayı üreticilerini verilen tohum değeri ile başlatır.

    Args:
        seed (int): Tohum değeri.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

seed_everything(SEED)

# Model ve token ayarları
PAD_TOKEN = ""
MODEL_NAME = args.model_name
NEW_MODEL = args.new_model_name

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto",
)

model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

# Veri yükleme ve ön işleme
dataset = load_dataset(args.dataset_name)

rows = []
for item in dataset["train"]:
    rows.append(
        {
            "question": item["question"],
            "context": item["context"],
            "answer": item["answer"],
        }
    )

df = pd.DataFrame(rows)

# Eksik veri kontrolü
df.isnull().value_counts()

def format_example(row: dict):
    """
    Verilen satırı biçimlendirir ve bir sohbet mesajı oluşturur.

    Args:
        row (dict): Veri satırı.

    Returns:
        dict: Biçimlendirilmiş sohbet mesajı.
    """
    prompt = dedent(
        f"""
        {row["question"]}

        Information:

        ```
        {row["context"]}
        ```
        """
    )
    messages = [
        {
            "role": "system",
            "content": "Use only the information to answer the question",
        },
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": row["answer"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

df["text"] = df.apply(format_example, axis=1)

def count_tokens(row: Dict) -> int:
    """
    Verilen satırdaki token sayısını hesaplar.

    Args:
        row (Dict): Veri satırı.

    Returns:
        int: Token sayısı.
    """
    return len(
        tokenizer(
            row["text"],
            add_special_tokens=True,
            return_attention_mask=False,
        )["input_ids"]
    )

df["token_count"] = df.apply(count_tokens, axis=1)

# Token sayısının histogramı
plt.hist(df.token_count, weights=np.ones(len(df.token_count)) / len(df.token_count))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.xlabel("Tokens")
plt.ylabel("Percentage")
plt.show()

# Token sayısı 512'den az olanları filtreleme
df = df[df.token_count < 512]
df = df.sample(6000)

# Eğitim ve test verisi bölme
train, temp = train_test_split(df, test_size=0.2, random_state=SEED)
val, test = train_test_split(temp, test_size=0.2, random_state=SEED)

# Veriyi JSON formatında kaydetme
train.sample(n=4000, random_state=SEED).to_json("train.json", orient="records", lines=True)
val.sample(n=500, random_state=SEED).to_json("val.json", orient="records", lines=True)
test.sample(n=100, random_state=SEED).to_json("test.json", orient="records", lines=True)

dataset = load_dataset(
    "json",
    data_files={"train": "train.json", "validation": "val.json", "test": "test.json"},
)

def create_test_prompt(data_row):
    """
    Test verisi için prompt oluşturur.

    Args:
        data_row (dict): Veri satırı.

    Returns:
        dict: Prompt mesajı.
    """
    prompt = dedent(
        f"""
        {data_row["question"]}

        Information:

        ```
        {data_row["context"]}
        ```
        """
    )
    messages = [
        {
            "role": "system",
            "content": "Use only the information to answer the question",
        },
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

# Model pipeline oluşturma
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    return_full_text=False,
)

# Test örnekleri ile tahmin yapma ve sonuçları karşılaştırma
rows = []
for row in tqdm(dataset["test"]):
    prompt = create_test_prompt(row)
    outputs = pipe(prompt)
    rows.append(
        {
            "question": row["question"],
            "context": row["context"],
            "prompt": prompt,
            "answer": row["answer"],
            "untrained_prediction": outputs[0]["generated_text"],
        }
    )

predictions_df = pd.DataFrame(rows)

# LoRA konfigürasyonu
lora_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=[
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self.attn.v_proj",
        "self.attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

# Data collator oluşturma
response_template = ""
collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer)

# TensorBoard entegrasyonu
sft_config = SFTConfig(
    output_dir=args.output_dir,
    dataset_text_field="text",
    max_seq_length=args.max_seq_length,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    eval_strategy="steps",
    eval_steps=0.2,
    save_steps=0.2,
    logging_steps=10,
    learning_rate=args.learning_rate,
    fp16=True,  # or bf16=True,
    save_strategy="steps",
    warmup_ratio=0.1,
    save_total_limit=2,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    save_safetensors=True,
    dataset_kwargs={
        "add_special_tokens": False,  # We template with special tokens
        "append_concat_token": False,  # No need to add additional separator token
    },
    seed=SEED,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=collator,
)

trainer.train()

trainer.save_model(NEW_MODEL)

tokenizer = AutoTokenizer.from_pretrained(NEW_MODEL)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)

model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
model = PeftModel.from_pretrained(model, NEW_MODEL)
model = model.merge_and_unload()

# Modeli Huggingface Hub'a yükleme
model.push_to_hub(NEW_MODEL, tokenizer=tokenizer, max_shard_size="5GB")
tokenizer.push_to_hub(NEW_MODEL)

# Yeni oluşturulan model ile test
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    return_full_text=False,
)

rows = []
for row in tqdm(dataset["test"]):
    outputs = pipe(create_test_prompt(row))
    rows.append(outputs[0]["generated_text"])

predictions_df["trained_prediction"] = rows

# Tahminleri CSV dosyasına kaydetme
predictions_df.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=None)

# Örnek verileri görüntüleme
sample = predictions_df.sample(n=20)
for i, row in sample.head(n=10).reset_index().iterrows():
    print(f"Example {i + 1}")
    response = f"""
answer: {row['answer']}

trained: {row['trained_prediction']}

untrained: {row['untrained_prediction']}
"""
    print(response)
