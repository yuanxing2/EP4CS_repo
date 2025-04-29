# EP4CS: **E**nhanced **P**rompting framework **f**or **C**ode **S**ummarization with Large Language Models

This is the repository of the artifact of the paper: **Enhanced Prompting framework for Code Summarization with Large Language Models**.

## The Dataset

### CodeXGLUE
The dataset processing method follows the same approach as Cadex Gluthda TasetProscinmetsaud Eastsam, Ascodex Gluander, and Bertun Lordart. You can download it from **[GitHub - CodeXGLUE](https://github.com/microsoft/CodeXGLUE)**.  For more detailed information, please refer to the handling method at **[CodeXGLUE Code-to-Text](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text)**. 

### ASAP

The publicly available dataset was utilized for constructing the Enhanced knowledge collection mentioned in the paper, which can be downloaded from **[https://zenodo.org/records/10494170](https://zenodo.org/records/10494170)**  . The main files are Java_data.zip and Python_data.zip, while the original paper is accessible via **[Automatic Semantic Augmentation of Language Model Prompts (for Code Summarization)](https://dl.acm.org/doi/pdf/10.1145/3597503.3639183)** .  
How to use:
    Step1：First, download file Java_data.zip、Python_data.zip from **[https://zenodo.org/records/10494170](https://zenodo.org/records/10494170)**,Put these two files under the data_process file.
    Step2:
        cd data_process
        unzip Java_data.zip
        unzip Python_data.zip
        python process_data.py
    When you're done, you'll have 4 files to use as background.

## Directory Hierarchy
```bash
├── evaluation
│   ├── evaluate.py          # main evaluation script
│   ├── meteor
│   ├── rouge
│   └── tokenizer
├── fewshot
│   ├── bleu.py
│   ├── fewshot.py
│   └── train_py.txt
├── Stage1
│   ├── models
│   ├── train.py             # training script for Stage 1
│   └── utils
├── Stage2
│   ├── bleu.py
│   ├── evaluate.py          # evaluation script for Stage 2
│   ├── model.py
│   ├── Qformer.py
│   ├── VAE.py
│   └── run.py               # main script for running Stage 2
├── zero_shot
│   ├── bleu.py
│   └── manual.py
├── data_process
│   └── process_data.py
├── requirements.txt
├── README.md
```

## Train
Once the data is processed, go to the $root_apth$ folder and run the following command to start the program and start training!

### Stage1
```bash
cd $root_path$/Stage1
python train.py\
    --output_dir=./saved_models \
    --model_type=bert \
    --config_name=bert-base-uncased \
    --model_name_or_path=bert-base-uncased \
    --tokenizer_name=bert-base-uncased \
    --do_train \
    --do_eval \
    --do_test \
    --train_data_file=dataset/train.jsonl \
    --eval_data_file=dataset/valid.jsonl \
    --test_data_file=dataset/test.jsonl \
    --epoch 1 \
    --block_size 256 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee log.log
```


### Stage2
After the first step of training is complete, we can get the trained Mapper component. At this point, you can jump to stage2 and start the second stage of training with the start command below.
```bash
cd $root_path$/Stage2
python run.py 
    --mode Prompt \
    --stru_prompt 64\
    --template [0,160] \#It is equal to the sum of the length of the knowledge vector and the structure vector
    --model_name_or_path bigcode/starcoderbase-1b \
    --train_filename ../dataset/python/clean_train.jsonl \
    --dev_filename ../dataset/python/clean_valid.jsonl \
    --test_filename ../dataset/python/clean_test.jsonl \
    --output_dir ./saved_models \
    --train_batch_size 2 \
    --eval_batch_size 2 \
    --learning_rate 5e-5 \
```

## Evaluation

### BLEU and SentenceBERT
```bash
cd $root_path$/Stage2
python evaluate.py --predict_file_path ./saved_models/test_0.output --ground_truth_file_path ./saved_models/test_0.gold --SentenceBERT_model_path ../all-MiniLM-L6-v2
```

### METEOR and ROUGE-L
To obtain METEOR and ROUGE-L, we need to activate the environment that contains python 2.7
```bash
conda activate your-env # python version==2.7
cd $root_path$/evaluation
python evaluate.py --predict_file_path ../PromptCS/saved_models/test_0.output --ground_truth_file_path ../PromptCS/saved_models/test_0.gold
```
Tip: The path should only contain English characters.

## Zero-Shot LLMs
```bash
cd $root_path$/zero_shot
python manual.py --model_name_or_path ../bigcode/starcoderbase-3b --test_filename ../dataset/python/clean_test.jsonl
python manual_gpt_3.5.py --test_filename ../dataset/python/clean_test.jsonl
```
## Few-Shot LLMs
We directly leverage the 5 python examples provided by Ahmed et al. in their GitHub [repository](https://github.com/toufiqueparag/few_shot_code_summarization/tree/main/Java), since we use the same experimental dataset (i.e., the CSN corpus).
```bash
cd $root_path$/fewshot
python fewshot.py --model_name_or_path ../bigcode/starcoderbase-3b --test_filename ../dataset/python/clean_test.jsonl
python fewshot_gpt_3.5.py --test_filename ../dataset/python/clean_test.jsonl
```
