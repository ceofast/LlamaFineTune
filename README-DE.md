# Llama-3-8B-Instruct Modell Feinabstimmung

Dieses Projekt enthält ein Python-Skript zur Feinabstimmung des Llama-3-8B-Instruct-Modells auf einem bestimmten Datensatz.

## Anforderungen

- Python 3.8 oder höher
- Folgende Python-Bibliotheken:
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

Um die Anforderungen zu installieren:
```bash
pip install torch numpy pandas seaborn matplotlib tqdm transformers datasets scikit-learn peft trl colored
```

## Verwendung

Um das Skript auszuführen, verwenden Sie den folgenden Befehl.

```bash
python fine_tune_llama.py --model_name <model_name> --new_model_name <new_model_name> --dataset_name <dataset_name> --output_dir <output_dir> --train_batch_size <train_batch_size> --eval_batch_size <eval_batch_size> --num_train_epochs <num_train_epochs> --learning_rate <learning_rate> --max_seq_length <max_seq_length> --seed <seed> --hf_token <your_hf_token>
```

## Parameter

* `--model_name`: Name des vortrainierten Modells zur Feinabstimmung. (Standard: "meta-llama/Meta-Llama-3-8B-Instruct")
* `--new_model_name`: Names des neuen Modells nach der Feinabstimmung. (Standard: "Llama-3-8B-Instruct-Finance-RAG-Aviation-AI")
* `--dataset_name`: Name des zu verwendenden Datensatzes. (Standard: "virattt/financial-qa-10K")
* `--output_dir`:  Verzeichnis, in dem das Modell und die Protokolle gespeichert werden. (Standard: "experiments")
* `--train_batch_size`: Batch-Größe für das Training. (Standard: 2)
* `--eval_batch_size`: Batch-Größe für die Bewertung. (Standard: 2)
* `--num_train_epochs`: Anzahl der Trainingsepochen. (Standard: 1)
* `--learning_rate`: Lernrate. (Standard: 1e-4)
* `--max_seq_length`: Maximale Sequenzlänge. (Standard: 512)
* `--seed`: Zufallszahlengenerator-Seed. (Standard: 42)
* `--hf_token`: Hugging Face Zugriffstoken.

## Beispiel

```bash
python fine_tune_llama.py --model_name "meta-llama/Meta-Llama-3-8B-Instruct" --new_model_name "Llama-3-8B-Instruct-Finance-RAG-Aviation-AI" --dataset_name "virattt/financial-qa-10K" --output_dir "experiments" --train_batch_size 2 --eval_batch_size 2 --num_train_epochs 1 --learning_rate 1e-4 --max_seq_length 512 --seed 42 --hf_token <your_hf_token>
```

Dieser Befehl führt die Feinabstimmung des Llama-3-8B-Instruct-Modells auf dem Datensatz "virattt/financial-qa-10K" durch und speichert die Ergebnisse im Verzeichnis "experiments".

## Lienz

Dieses Projekt ist unter der MIT-Lizenz lizenziert. Details finden Sie in der Datei 'LICENCE'.