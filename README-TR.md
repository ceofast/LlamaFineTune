# Llama-3-8B-Instruct Model Fine-Tuning

Bu proje, Llama-3-8B-Instruct modelini belirli bir veri kümesi üzerinde fine-tuning işlemi yapmak için bir Python scripti içerir.

## Gereksinimler

- Python 3.8 veya üzeri
- Aşağıdaki Python kütüphaneleri
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

Gereksinimleri yüklemek için:
```bash
pip install torch numpy pandas seaborn matplotlib tqdm transformers datasets scikit-learn peft trl colored
```

## Kullanım

Scripti çalıştırmak için aşağıdaki komutu kullanabilirsiniz:
```bash
python fine_tune_llama.py --model_name <model_name> --new_model_name <new_model_name> --dataset_name <dataset_name> --output_dir <output_dir> --train_batch_size <train_batch_size> --eval_batch_size <eval_batch_size> --num_train_epochs <num_train_epochs> --learning_rate <learning_rate> --max_seq_length <max_seq_length> --seed <seed> --hf_token <your_hf_token>
```

## Parametreler

* `--model_name`: Fine-tuning işlemi için kullanılacak önceden eğitilmiş model adı. (Varsayılan: "meta-llama/Meta-Llama-3-8B-Instruct")
* `--new_model_name`: Fine-tuning işlemi sonrası yeni model adı. (Varsayılan: "Llama-3-8B-Instruct-Finance-RAG-Aviation-AI")
* `--dataset_name`: Kullanılacak veri kümesi adı. (Varsayılan: "virattt/financial-qa-10K")
* `--output_dir`: Model ve logların kaydedileceği çıkış dizini. (Varsayılan: "experiments")
* `--train_batch_size`: Eğitim batch boyutu. (Varsayılan: 2)
* `--eval_batch_size`: Değerlendirme batch boyutu. (Varsayılan: 2)
* `--num_train_epochs`: Eğitim epoch sayısı. (Varsayılan: 1)
* `--learning_rate`: Öğrenme oranı. (Varsayılan: 1e-4)
* `--max_seq_length`: Maksimum sekans uzunluğu. (Varsayılan: 512)
* `--seed`: Rastgele seed değeri. (Varsayılan: 42)
* `--hf_token`: Hugging Face access token.


## Örnek Kullanım

```bash
python fine_tune_llama.py --model_name "meta-llama/Meta-Llama-3-8B-Instruct" --new_model_name "Llama-3-8B-Instruct-Finance-RAG-Aviation-AI" --dataset_name "virattt/financial-qa-10K" --output_dir "experiments" --train_batch_size 2 --eval_batch_size 2 --num_train_epochs 1 --learning_rate 1e-4 --max_seq_length 512 --seed 42 --hf_token <your_hf_token>
```

Bu komut, Llama-3-8B-Instruct modelini "virattt/financial-qa-10K" veri kümesi üzerinde fine-tuning işlemi yapar ve sonuçları "experiments" dizinine kaydeder.


## Lisans

Bu proje MIT Lisansı ile lisanslanmıştır. Ayrıntılar için 'LICENCE' dosyasına bakabilirsiniz.