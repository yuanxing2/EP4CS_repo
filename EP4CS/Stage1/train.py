import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

import os
import json
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, BertConfig
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from utils.utils import dataiterator, NoamLR
from transformers import  RobertaConfig, RobertaModel, RobertaTokenizer, BertTokenizer, AutoTokenizer, AutoModel

from models.stage1 import CodeBlip2

MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)
}   

import logging
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(console)


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_ids,
                 url,
                 attention_mask,
                 code_bert,
                 code_bert_attention_mask,

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_ids = nl_ids
        self.url=url
        self.attention_mask=attention_mask
        self.code_bert=code_bert
        self.code_bert_attention_mask=code_bert_attention_mask

def convert_examples_to_features(js ,tokenizer, bert_tokenizer,block_size, text_size):
    #code
    if 'code_tokens' in js:
        code=' '.join(js['code_tokens'])
        
    else:
        code=' '.join(js['function_tokens'])
    code_tokens=tokenizer.tokenize(code)[:block_size-2]
    code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = block_size - len(code_ids)
    code_ids+=[tokenizer.pad_token_id]*padding_length
    
    code_tokens_bert = bert_tokenizer(
        code,
        padding="max_length",
        truncation=True,
        max_length=text_size,
        return_tensors="pt",
    )

    nl=' '.join(js['docstring_tokens']) + js["repo"] + js["path"]
    text_tokens = bert_tokenizer(
        nl,
        padding="max_length",
        truncation=True,
        max_length=text_size,
        return_tensors="pt",
    )
    
    return InputFeatures(code_tokens,code_ids,text_tokens.input_ids,js['url'],text_tokens.attention_mask,code_tokens_bert.input_ids,code_tokens_bert.attention_mask)

class TextDataset(Dataset):
    def __init__(self, args, tokenizer, bert_tokenizer, file_path=None):
        self.examples = []
        data=[]
        with open(file_path) as f:
            if 'train' in file_path:
                for i in range(250):
                    line = f.readline()
                    line = line.strip()
                    js = json.loads(line)
                    data.append(js)
            else:
                for i in range(80):
                    line = f.readline()
                    line = line.strip()
                    js = json.loads(line)
                    data.append(js)
        for js in data:
            self.examples.append(convert_examples_to_features(js,tokenizer,bert_tokenizer, args.block_size, args.text_size))                        
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        return (torch.tensor(self.examples[i].code_ids),self.examples[i].nl_ids,self.examples[i].attention_mask,self.examples[i].code_bert,self.examples[i].code_bert_attention_mask)
    
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataset, model, tokenizer, code_model=None, bert_tokenizer=None):
    """ Train the model """

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=args.train_batch_size,num_workers=4,pin_memory=True)
    args.max_steps=args.epoch*len(train_dataloader)
    args.save_steps=len( train_dataloader)//10
    args.warmup_steps=len( train_dataloader)
    args.logging_steps=len( train_dataloader)
    args.num_train_epochs=args.epoch
    model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, no_deprecation_warning=True)# eps=args.adam_epsilon
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1,
                                                num_training_steps=args.max_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')

    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step = args.start_step
    tr_loss, logging_loss,avg_loss,tr_nb,tr_num,train_loss = 0.0, 0.0,0.0,0,0,0
    
    best_mrr=0.0
    best_acc=0.0

    model.zero_grad()
    
    warmup_steps=2000
    scheduler=NoamLR(optimizer,warmup_steps=warmup_steps)
    for idx in range(args.start_epoch, int(args.num_train_epochs)): 
        bar = train_dataloader
        tr_num=0
        train_loss=0
        loss_totle_sum, loss_itm_sum, loss_lm_sum=0.0,0.0,0.0
        for step, batch in enumerate(bar):
            model.train()
            code_inputs = batch[0].to(args.device)    
            nl_inputs = batch[1].to(args.device)
            nl_inputs = nl_inputs.squeeze(1)
            attention_mask = batch[2].to(args.device)
            attention_mask = attention_mask.squeeze(1)
            code_attention_mask = batch[4].to(args.device)
            code_attention_mask = code_attention_mask.squeeze(1)
            code_model.eval()
            with torch.no_grad():
                outputs = code_model(code_inputs)
                code_inputs = outputs.last_hidden_state
            loss, loss_totle, loss_itm, loss_lm = model(nl_inputs, code_inputs, attention_mask, code_attention_mask)
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            loss_totle_sum += loss_totle.item()
            loss_itm_sum += loss_itm.item()
            loss_lm_sum += loss_lm.item()
            tr_loss += loss.item()
            tr_num+=1
            train_loss+=loss.item()
            if avg_loss==0:
                avg_loss=tr_loss
            avg_loss=round(train_loss/tr_num,5)
            av_loss_totle=round(loss_totle_sum/tr_num,5)
            av_loss_itm=round(loss_itm_sum/tr_num,5)
            av_loss_lm=round(loss_lm_sum/tr_num,5)
            if (step+1)% 100==0:
                logger.info("epoch {} step {} total_loss {} loss_itc {} loss_itm {} loss_lm {}".format(idx+1,step+1,avg_loss, av_loss_totle, av_loss_itm, av_loss_lm))


        global_step += 1
        output_flag=True
        avg_loss=1
        model.eval()
        results = evaluate(args, model, tokenizer,eval_when_training=True, code_model=code_model,bert_tokenizer=bert_tokenizer)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,4))                    
        # Save model checkpoint
        tr_num=0
        train_loss=0

        if results['eval_mrr']>best_acc:
            best_acc=results['eval_mrr']
            logger.info("  "+"*"*20)  
            logger.info("  Best mrr:%s",round(best_acc,4))
            logger.info("  "+"*"*20)                          
            
            checkpoint_prefix = 'checkpoint-best-mrr'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_dir = os.path.join(output_dir, '{}'.format('model.pth')) 
            torch.save(model.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)

        
        if idx%3==0:
            torch.save(model.state_dict(), "output/stage1"+str(idx)+".pth")


eval_dataset=None
def evaluate(args, model, tokenizer,eval_when_training=False, code_model=None,bert_tokenizer=None):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir
    global eval_dataset
    if eval_dataset is None:
        eval_dataset = TextDataset(args,tokenizer, bert_tokenizer,args.eval_data_file)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    args.per_gpu_eval_batch_size = 16
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    code_vecs=[] 
    nl_vecs=[]
    for batch in eval_dataloader:
        code_inputs = batch[0].to(args.device)    
        nl_inputs = batch[1].to(args.device)
        nl_inputs = nl_inputs.squeeze(1)
        attention_mask = batch[2].to(args.device)
        attention_mask = attention_mask.squeeze(1)
        code_attention_mask = batch[4].to(args.device)
        code_attention_mask = code_attention_mask.squeeze(1)
        # code_inputs = batch[0] 
        # nl_inputs = batch[1]
        code_model.eval()
        with torch.no_grad():
            outputs = code_model(code_inputs)
            code_inputs = outputs.last_hidden_state
            code_vec_tmp, nl_vec_tmp= model.eval_me(nl_inputs, code_inputs, attention_mask)
            code_vecs.append(code_vec_tmp.cpu().numpy())
            nl_vecs.append(nl_vec_tmp.cpu().numpy())
        nb_eval_steps += 1
    code_vecs=np.concatenate(code_vecs,0)
    nl_vecs=np.concatenate(nl_vecs,0)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    scores=np.matmul(nl_vecs,code_vecs.T)
    ranks=[]
    for i in range(len(scores)):
        score=scores[i,i]
        rank=1
        for j in range(len(scores)):
            if i!=j and scores[i,j]>=score:
                rank+=1
        ranks.append(1/rank)    
    
    result = {
        "eval_loss": float(perplexity),
        "eval_mrr":float(np.mean(ranks))
    }


    return result

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default='dataset_new/train.jsonl', type=str, required=False,
                         help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default='output', type=str, required=True,
                         help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
                    
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=256, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--text_size", default=256, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")    
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    #print("device",device)
    args.n_gpu = 1
    args.device = device

    # Set seed
    set_seed(args.seed)

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels=1
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    model = CodeBlip2(tokenizer=bert_tokenizer).to(args.device)
    
    code_model = AutoModel.from_pretrained("microsoft/codebert-base").to(args.device)


    train_dataset = TextDataset(args, tokenizer, bert_tokenizer, args.train_data_file)
    train(args, train_dataset, model, tokenizer, code_model=code_model,bert_tokenizer=bert_tokenizer)


if __name__ == "__main__":
    main()


