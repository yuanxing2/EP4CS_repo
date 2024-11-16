# Enhanced Prompting framework for Code Summarization with Large Language Models

This is the repository of the artifact of the paper: **Enhanced Prompting framework for Code Summarization with Large Language Models**.

## DataSet
    The dataset processing method is the same as that of Cadex Gluthda TasetProscinmetsaud Eastsam, Ascodex Gluander, Bertun Lordart, which can be downloaded in [https://github.com/microsoft/CodeXGLUE](https://github.com/microsoft/CodeXGLUE).
    
    For more information, please refer to the handling method：https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text


## Train
Once the data is processed, go to the stage1 folder and run the following command to start the program and start training!


## Stage1
```
python train.py\
    --output_dir=./saved_models \
    --model_type=bert \
    --config_name=bert-base-uncased \
    --model_name_or_path=bert-base-uncased \
    --tokenizer_name=bert-base-uncased \
    --do_train \
    --do_eval \
    --do_test \
    --train_data_file=dataset_new/train.jsonl \
    --eval_data_file=dataset_new/valid.jsonl \
    --test_data_file=dataset_new/test.jsonl \
    --epoch 1 \
    --block_size 256 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee log.log
```


## Stage2
After the first step of training is complete, we can get the trained Mapper component. At this point, you can jump to stage2 and start the second stage of training with the start command below.
```
python run.py 
    --mode PromptCS \
    --prompt_encoder_type lstm \
    --template [0,160] \
    --model_name_or_path Qwen/Qwen1.5-4B \
    --train_filename ../dataset/python/clean_train.jsonl \
    --dev_filename ../dataset/python/clean_valid.jsonl \
    --test_filename ../dataset/python/clean_test.jsonl \
    --output_dir ./saved_models \
    --train_batch_size 2 \
    --eval_batch_size 2 \
    --learning_rate 5e-5 \
    --stru_prompt 32 \
```

## Evaluation

### BLEU and SentenceBERT
```
    cd Stage2
    python evaluate.py --predict_file_path ./saved_models/test_0.output --ground_truth_file_path ./saved_models/test_0.gold --SentenceBERT_model_path ../all-MiniLM-L6-v2
```

### METEOR and ROUGE-L
To obtain METEOR and ROUGE-L, we need to activate the environment that contains python 2.7
```
    conda activate your-env
    unzip evaluation
    cd evaluation
    python evaluate.py --predict_file_path ../PromptCS/saved_models/test_0.output --ground_truth_file_path ../PromptCS/saved_models/test_0.gold
```
Tip: The path should only contain English characters.

## Zero-Shot LLMs
```
    cd zeroshot
    python manual.py --model_name_or_path ../bigcode/starcoderbase-3b --test_filename ../dataset/python/clean_test.jsonl
    python manual_gpt_3.5.py --test_filename ../dataset/python/clean_test.jsonl
```
## Few-Shot LLMs
We directly leverage the 5 python examples provided by Ahmed et al. in their GitHub [repository](https://github.com/toufiqueparag/few_shot_code_summarization/tree/main/Java), since we use the same experimental dataset (i.e., the CSN corpus).
```
    cd fewshot
    python fewshot.py --model_name_or_path ../bigcode/starcoderbase-3b --test_filename ../dataset/python/clean_test.jsonl
    python fewshot_gpt_3.5.py --test_filename ../dataset/python/clean_test.jsonl
```
