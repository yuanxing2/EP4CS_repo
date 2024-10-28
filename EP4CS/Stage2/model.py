import math
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, RobertaConfig, RobertaModel, BertTokenizer, AutoModel

class Prompt(nn.Module):
    def __init__(self, args, device, template):
        super(Prompt, self).__init__()
        self.num_features = 768
        self.args = args
        self.mode = args.mode
        self.device = device
        self.use_lm_finetune = False

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

        if args.mode == 'finetune':
            self.use_lm_finetune = True
            template = [0, 0]
        self.template = template

        # set allowed vocab set
        self.vocab = self.tokenizer.get_vocab()
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[PROMPT]']})
        self.pseudo_token_id = self.tokenizer.get_vocab()['[PROMPT]']
        self.pad_token_id, self.sep_token_id, self.eos_token_id, self.unk_token_id = self.get_special_token_id()

        self.prompt_tokens = [self.pseudo_token_id]
        self.sep_tokens = [self.sep_token_id]
        self.eos_tokens = [self.eos_token_id]

        #load pre-trained model
        self.model = create_model(self.args, self.use_lm_finetune)
        self.model = self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = self.use_lm_finetune
        self.embeddings = get_embedding_layer(self.args, self.model)

        self.hidden_size = self.embeddings.embedding_dim
        self.spell_length = sum(self.template)

        self.max_target_length = args.max_target_length
        self.max_code_length = args.max_code_length
        self.lsm = nn.LogSoftmax(dim=-1)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id, reduction='sum')
        self.unixcode_tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        self.unixcode_model = AutoModel.from_pretrained("microsoft/unixcoder-base").to(self.device)
        self.unixcoder_pad_ids = self.unixcode_tokenizer.pad_token_id

        self.Qformer = torch.load("../stage1_model/Qformer_64_stage1_epoch0.pth")
        self.query_tokens = torch.load("../stage1_model/query_64_token_epoch0.pth")
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None
        
        self.LLM_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.model.config.hidden_size
        )
        self.fc1 = nn.Linear(256, self.args.stru_prompt)
        self.fc2 = nn.Linear(self.num_features,self.model.config.hidden_size)
    
    def get_prompt(self, code_embed, unix_struct_info, unixcode_attention_mask):
        query_tokens = self.query_tokens.expand(code_embed.shape[0], -1, -1).to(self.device)
        #print(query_tokens.grad)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=code_embed,
            encoder_attention_mask=unixcode_attention_mask,
            return_dict=True,
        )

        structured_code = self.fc2(self.fc1(code_embed.transpose(1, 2)).transpose(1,2)) 
        vtokens = self.LLM_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :]).to(self.device)
        prompt = torch.cat((vtokens, structured_code), dim=1)
        return prompt

    def get_special_token_id(self):
        pad_token_id, sep_token_id, eos_token_id, unk_token_id = None, None, None, None
        model_name = self.args.model_name_or_path.lower()
        if 'starcoder' in model_name:
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id
            sep_token_id = self.vocab['<fim_middle>']
            eos_token_id = self.tokenizer.eos_token_id
            unk_token_id = self.tokenizer.unk_token_id
        elif 'polycoder' in model_name:
            pad_token_id = self.vocab['<|padding|>']
            sep_token_id = self.vocab['<|separator|>']
            eos_token_id = self.vocab['<|endoftext|>']
            unk_token_id = self.vocab['<|padding|>']
        elif 'codegen' in model_name:
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id
            sep_token_id = self.vocab['//']
            eos_token_id = self.tokenizer.eos_token_id
            unk_token_id = self.tokenizer.unk_token_id
        elif 'qwen' in model_name:
            pad_token_id = self.tokenizer('<|endoftext|>').input_ids[0]
            sep_token_id = self.tokenizer('<|endoftext|>').input_ids[0]
            eos_token_id = self.tokenizer('<|endoftext|>').input_ids[0]
            unk_token_id = self.tokenizer('<|endoftext|>').input_ids[0]     

        return pad_token_id, sep_token_id, eos_token_id, unk_token_id
        
    def embed_input(self, queries, unix_inputs):
        if self.mode == 'Prompt':
            return self.cstuning_embed_input(queries, unix_inputs)
        else:
            return self.finetune_embed_input(queries)

    def finetune_embed_input(self, queries):
        return self.embeddings(queries)
    
    def cstuning_embed_input(self, queries, unix_inputs):#####
        bz = queries.shape[0]
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.unk_token_id
        raw_embeds = self.embeddings(queries_for_embedding)

        unix_inputs = pad_sequence(unix_inputs, True, padding_value=self.unixcoder_pad_ids).long().to(self.device)
        unixcode_attention_mask = unix_inputs != self.unixcoder_pad_ids
        
        self.unixcode_model.eval()
        with torch.no_grad():
            unix_output = self.unixcode_model(unix_inputs, unixcode_attention_mask)
            unix_struct_info = unix_output.pooler_output
            unix_hidden_state = unix_output.last_hidden_state
        unix_hidden_state_emb_att = torch.ones(unix_hidden_state.size()[:-1], dtype=torch.long).to(self.device)

        blocked_indices = (queries == self.pseudo_token_id).nonzero().reshape((bz, self.spell_length, 2))[:, :, 1]  # bz
        replace_embeds = self.get_prompt(unix_hidden_state, unix_struct_info, unix_hidden_state_emb_att)
        for bidx in range(bz):
            for i in range(self.spell_length):
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[bidx, i, :]
                
        return raw_embeds

    def get_query(self, x_h, x_t=None):
        left = self.prompt_tokens * self.template[0] + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x_h)[:self.max_code_length]) + self.prompt_tokens * self.template[1]

        block_size = 256
        code_tokens = self.unixcode_tokenizer.tokenize(x_h)[:block_size-2]
        code_tokens = [self.unixcode_tokenizer.cls_token]+code_tokens+[self.unixcode_tokenizer.sep_token]
        unix_input =  self.unixcode_tokenizer.convert_tokens_to_ids(code_tokens)
        padding_length = block_size - len(unix_input)
        unix_input += [self.unixcode_tokenizer.pad_token_id]*padding_length

        if x_t is not None:
            right = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x_t)[:self.max_target_length]) + self.eos_tokens
        else:
            right = []

        input_ids = left + self.sep_tokens + right

        return torch.LongTensor(input_ids),  len(left), torch.LongTensor(unix_input)

    def prepare_inputs(self, inputs, unix_inputs):
        inputs = pad_sequence(inputs, True, padding_value=self.pad_token_id).long().to(self.device)

        attention_mask = inputs != self.pad_token_id
        inputs_embeds = self.embed_input(inputs, unix_inputs)

        inputs_embeds = inputs_embeds.to(self.device)
        attention_mask = attention_mask.to(self.device)

        if self.mode != 'finetune':
            inputs_embeds = inputs_embeds.half()
            attention_mask = attention_mask.half()

        return inputs, inputs_embeds, attention_mask


    def forward(self, x_hs=None, x_ts=None):#x_hs: code, x_ts: docstring
        bz = len(x_hs)

        if x_ts is not None:
            inputs, sum_idx, ext_inputs, unix_inputs = [], [], [], []
            for i in range(bz):
                input, idx, unix_input = self.get_query(x_hs[i], x_ts[i])#get_query函数返回的是一个tensor和一个idx,其中tensor是输入的query，idx是输入的query的长度
                inputs.append(input)
                sum_idx.append(idx)
                unix_inputs.append(unix_input)
            inputs, inputs_embeds, attention_mask = self.prepare_inputs(inputs, unix_inputs)
            #print(inputs)
            output = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

            logits = output.logits
            loss = None

            for i in range(bz):
                idx = sum_idx[i]
                shift_logits = logits[i][idx:-1, :].contiguous()
                shift_labels = inputs[i][idx+1:].contiguous()
                
                if loss is None:
                    loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                else:
                    loss += self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss / bz

            return loss
        else:
            queries, sum_idx, tmp_idx, unix_inputs = [], [], [], []
            for i in range(bz):
                query, idx, unix_input = self.get_query(x_h=x_hs[i])
                queries.append(query)
                sum_idx.append(idx)
                tmp_idx.append(idx)
                unix_inputs.append(unix_input)

            for _ in range(self.max_target_length):
                inputs, inputs_embeds, attention_mask = self.prepare_inputs(queries, unix_inputs)

                output = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

                logits = output.logits
                for i in range(bz):
                    idx = tmp_idx[i]
                    tmp_idx[i] += 1
                    next_token_logits = logits[i, idx:idx+1, :]
                    _, next_token = torch.max(next_token_logits, dim=1)

                    queries[i] = torch.cat([queries[i].to(self.device), next_token], dim=0)

            answer = []
            for i in range(bz):
                idx = sum_idx[i]
                t = queries[i][idx+1:]
                t=t.tolist()
                if self.eos_token_id in t:
                    t = t[:t.index(self.eos_token_id)]
                words = self.tokenizer.decode(t).replace('\n','')
                answer.append(words)

            return answer

def create_model(args, use_lm_finetune):
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).half()
    if not use_lm_finetune:
        model = model.half()
    return model


def get_embedding_layer(args, model):
    return model.base_model.get_input_embeddings()
        


