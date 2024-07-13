# Llama-3-8B-Instruct Model Fine-Tuning

This project includes a Python script for fine-tuning the Llama-3-8B-Instruct model on a specific dataset.

## Requirements

- Python 3.8 or higher
- The following Python libraries:
  - torch
  - numpy
  - pandas
  - seaborn
  - matplotlib
  - tqdm
  - transformers
  - datasets
  - scikit-learn
  - peft
  - trl
  - colored

To install the requirements:
```bash
pip install torch numpy pandas seaborn matplotlib tqdm transformers datasets scikit-learn peft trl colored
```

## Usage

To run the script, use the following command:
```bash
python fine_tune_llama.py --model_name <model_name> --new_model_name <new_model_name> --dataset_name <dataset_name> --output_dir <output_dir> --train_batch_size <train_batch_size> --eval_batch_size <eval_batch_size> --num_train_epochs <num_train_epochs> --learning_rate <learning_rate> --max_seq_length <max_seq_length> --seed <seed> --hf_token <your_hf_token>
```

## Parameters

* `--model_name`: Name of the pretrained model to be fine-tuned. (Default: "meta-llama/Meta-Llama-3-8B-Instruct")
* `--new_model_name`: Name of the new model after fine-tuning. (Default: "Llama-3-8B-Instruct-Finance-RAG-Aviation-AI)
* `--dataset_name`: Name of the dataset to be used. (Default: "virattt/financial-qa-10K)
* `--output_dir`: Directory where the model and logs will be saved. (Default: "experiments")
* `--train_batch_size`: Batch size for training. (Default: 2)
* `--eval_batch_size`: Batch size for evaluation. (Default: 2)
* `--num_train_epochs`: Number of training epochs. (Default: 1)
* `--learning_rate`: Learning rate. (Default: 1e-4)
* `--max_seq_length`: Maximum sequence length. (Default: 512)
* `--seed`: Random seed. (Default: 42)
* `--hf_token`: Hugging Face access token.

## Example

```bash
python fine_tune_llama.py --model_name "meta-llama/Meta-Llama-3-8B-Instruct" --new_model_name "Llama-3-8B-Instruct-Finance-RAG-Aviation-AI" --dataset_name "virattt/financial-qa-10K" --output_dir "experiments" --train_batch_size 2 --eval_batch_size 2 --num_train_epochs 1 --learning_rate 1e-4 --max_seq_length 512 --seed 42 --hf_token <your_hf_token>
```

This command will fine-tune the Llama-3-8B-Instruct model on the "virattt/financial-qa-10K" dataset and save the results in the "experiments" directory.

## Licence

This project is licenced under the MIT License. See the 'LICENSE' file for details.