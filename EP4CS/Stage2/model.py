import math
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, RobertaConfig, RobertaModel, BertTokenizer, AutoModel
import torch.nn.functional as F

from VAE import VAE

class PromptCS(nn.Module):
    def __init__(self, args, device, template):
        super(PromptCS, self).__init__()
        self.num_features = 768
        self.args = args
        #self.num_query_token = self.args.template[0] #
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

        # load prompt encoder
        self.hidden_size = self.embeddings.embedding_dim
        self.spell_length = sum(self.template)

        # if args.mode == 'PromptCS':
        #     self.prompt_agent = PromptAgent(self.template, self.hidden_size, self.tokenizer, self.device, args)
        #     self.prompt_agent = self.prompt_agent.to(self.device)

        self.max_target_length = args.max_target_length
        self.max_code_length = args.max_code_length
        self.lsm = nn.LogSoftmax(dim=-1)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id, reduction='sum')
        #################################################################################
        self.unixcode_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        self.unixcode_model = AutoModel.from_pretrained("microsoft/codebert-base").to(self.device)
        self.unixcoder_pad_ids = self.unixcode_tokenizer.pad_token_id

        self.Qformer = torch.load("../stage1_model/Qformer_128_stage1_epoch3_python.pth")
        self.query_tokens = torch.load("../stage1_model/query_128_token_epoch3_python.pth")
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None
        
        self.LLM_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.model.config.hidden_size
        )
        self.fc1 = nn.Linear(256, self.args.stru_prompt)
        self.fc2 = nn.Linear(self.num_features,self.model.config.hidden_size)
        #self.alpha = nn.Parameter(torch.tensor(0.5, requires_grad=False)) #alpha是一个参数，初始化为0.5
        # self.prompt = nn.Parameter(
        #     torch.zeros(1, args.stru_prompt, self.model.config.hidden_size)
        # ).to(self.device)
        # self.prompt.data.normal_(mean=0.0, std=1.5) #初始化query_tokens
        #print(self.args.stru_prompt * self.num_features)
        self.VAE_model = VAE(input_dim=self.args.stru_prompt * self.num_features, h_dim=400, z_dim=self.args.stru_prompt * self.model.config.hidden_size)



    # def init_Qformer(self, num_query_token, vision_width, cross_attention_freq=2, tokenizer=None):
    #     #encoder_config = BertConfig.from_pretrained(self.bert_dir)
    #     encoder_config = BertConfig.from_pretrained("bert-base-uncased")
    #     encoder_config.encoder_width = vision_width
    #     # insert cross-attention layer every other block
    #     encoder_config.add_cross_attention = True
    #     encoder_config.cross_attention_freq = cross_attention_freq
    #     encoder_config.query_length = num_query_token
    #     code_blip2 = CodeBlip2(tokenizer=tokenizer)
    #     path = "/home/fmy/newproject/yuanxing/PromptCS-main/stage1_model/model_unixcode_stage19.pth"#第一阶段预训练好的模型
    #     code_blip2.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    #     Qformer = code_blip2.Qformer.to(self.device)
    #     query_tokens = nn.Parameter(
    #         torch.zeros(1, num_query_token, encoder_config.hidden_size, requires_grad=True),
    #         requires_grad=True
    #     ).to(self.device)
    #     query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

    #     return Qformer, query_tokens
    

    def vae_loss(self, recon_x, x, mu, logvar):
        """
        计算 VAE 的损失函数，使用均方误差作为重构损失。
        
        参数:
        - recon_x: 解码器输出的重构数据
        - x: 原始输入数据
        - mu: 潜在空间的均值
        - logvar: 潜在空间的对数方差
        
        返回:
        - total_loss: 总损失，包括重构损失和KL散度损失
        - recon_loss: 重构损失
        - kld_loss: KL散度损失
        """
        # 计算重构损失，这里使用均方误差
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')

        # 计算 KL 散度损失
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # 总损失是重构损失和 KL 散度损失的加权和
        total_loss = recon_loss + kld_loss

        return total_loss, recon_loss, kld_loss



    def get_prompt(self, code_embed, unix_struct_info, unixcode_attention_mask):
        query_tokens = self.query_tokens.expand(code_embed.shape[0], -1, -1).to(self.device)
        #print(query_tokens.grad)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=code_embed,
            encoder_attention_mask=unixcode_attention_mask,
            return_dict=True,
        )

        # code_embed 的shape为[batchsize,1, self.num_features]
        # 转换成[batchsize,,self.sturct_length,self.model.config.hidden_size]的维度
        #将code_embed的第二和第三个维度转置
        # code_embed=code_embed.transpose(1,2)#变为了[batchsize,self.num_features,self.maxcode_length]
        # #使用全连接层将code_embed转换为[batchsize,self.num_features,self.model.config.hidden_size]
        # structured_code = self.fc1(code_embed).transpose(1,2) #structured_code的shape为[batchsize,self.maxcode_length,self.model.config.hidden_size]
        # structured_code = self.fc2(structured_code)
        #print(code_embed[:,:1,:].shape)
        #unix_struct_info = unix_struct_info.unsqueeze(1)
        #structured_code = self.fc2(self.fc1(code_embed.transpose(1, 2)).transpose(1,2)) #
        #print(structured_code.shape)
        x = self.fc1(code_embed.transpose(1, 2)).transpose(1,2)
        # print(x.shape)
        x_hat, mu, log_var = self.VAE_model(x)
        # print(x_hat.shape)
        # print(mu.shape)
        # print(log_var.shape)
        #print(structured_code.shape)
        #print(code_embed.shape)
        vtokens = self.LLM_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :]).to(self.device)
        prompt = torch.cat((vtokens, log_var.view(x.shape[0], self.args.stru_prompt, self.model.config.hidden_size)), dim=1)
        #print(prompt.shape)
        #print(vtokens)
        # vtokens = None
        # structured_code = None
        # prompt = self.prompt.expand(code_embed.shape[0], -1, -1)
        #print(prompt.grad)
        #print(prompt.shape)
        #prompt = self.prompt_agent(prompt)
        loss_vae,_ ,_  = self.vae_loss(x_hat, x, mu, log_var)
        #print(prompt.shape)
        return prompt, vtokens, log_var.view(x.shape[0], self.args.stru_prompt, self.model.config.hidden_size), loss_vae

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
        

        return pad_token_id, sep_token_id, eos_token_id, unk_token_id
        
    def embed_input(self, queries, unix_inputs):
        if self.mode == 'PromptCS':
            return self.cstuning_embed_input(queries, unix_inputs)
        else:
            return self.finetune_embed_input(queries)

    def finetune_embed_input(self, queries):
        return self.embeddings(queries), None
    
    def cstuning_embed_input(self, queries, unix_inputs):#####
        bz = queries.shape[0]
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.unk_token_id
        raw_embeds = self.embeddings(queries_for_embedding)

        unix_inputs = pad_sequence(unix_inputs, True, padding_value=self.unixcoder_pad_ids).long().to(self.device)
        unixcode_attention_mask = unix_inputs != self.unixcoder_pad_ids
        #with torch.no_grad():

        self.unixcode_model.eval()
        with torch.no_grad():
            unix_output = self.unixcode_model(unix_inputs, unixcode_attention_mask)
            unix_struct_info = unix_output.pooler_output
            unix_hidden_state = unix_output.last_hidden_state
        #print(unix_hidden_state.requires_grad,"=======================================")
        unix_hidden_state_emb_att = torch.ones(unix_hidden_state.size()[:-1], dtype=torch.long).to(self.device)

        #print("unix_hidden_state", unix_hidden_state)
        #print(unix_hidden_state.shape)

        blocked_indices = (queries == self.pseudo_token_id).nonzero().reshape((bz, self.spell_length, 2))[:, :, 1]  # bz
        replace_embeds, vtokens, structured_code, loss_vae = self.get_prompt(unix_hidden_state, unix_struct_info, unix_hidden_state_emb_att)
        

        
        
        len2_st = structured_code.shape[2]
        len3_st = structured_code.shape[1]
        code_emb_st = F.interpolate(raw_embeds.transpose(1, 2), size=(len3_st), mode='linear').transpose(1, 2)
        #p = F.softmax(structured_code, dim=1)  # 沿着len3的维度进行softmax
        q_st = F.softmax(code_emb_st, dim=1)

        # 计算KL散度
        # 注意：kl_div默认期望第一个输入是对数概率，因此使用log_softmax而不是softmax
        #log_p_st = F.log_softmax(structured_code, dim=1)

        #kl_loss_st = F.kl_div(log_p_st, q_st, reduction='batchmean')
        kl_loss_st = 0.0
        
        #len2_dy = raw_embeds.shape[2]
        len3_dy = vtokens.shape[1]
        code_emb_dy = F.interpolate(raw_embeds.transpose(1, 2), size=(len3_dy), mode='linear', align_corners=False).transpose(1, 2)
        #p = F.softmax(structured_code, dim=1)  # 沿着len3的维度进行softmax
        q_dy = F.softmax(code_emb_dy, dim=1)

        # 计算KL散度
        # 注意：kl_div默认期望第一个输入是对数概率，因此使用log_softmax而不是softmax
        log_p_dy = F.log_softmax(vtokens, dim=1)
        kl_loss_dy = F.kl_div(log_p_dy, q_dy, reduction='batchmean')
        loss_kl = (kl_loss_st + kl_loss_dy)/2 * torch.tensor(0.5) + loss_vae * torch.tensor(0.1)
        #loss_kl = None
        #print(raw_embeds.shape)
        #print(replace_embeds.shape)
        for bidx in range(bz):
            for i in range(self.spell_length):
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[bidx, i, :]
        #print(loss_kl)        
        
        return raw_embeds, loss_kl

    def get_query(self, x_h, x_t=None):
        left = self.prompt_tokens * self.template[0] + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x_h)[:self.max_code_length]) + self.prompt_tokens * self.template[1]

        #left_temp = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x_h)[:self.max_code_length])

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
        inputs_embeds, loss_kl = self.embed_input(inputs, unix_inputs)

        inputs_embeds = inputs_embeds.to(self.device)
        attention_mask = attention_mask.to(self.device)

        if self.mode != 'finetune':
            inputs_embeds = inputs_embeds.half()
            attention_mask = attention_mask.half()

        return inputs, inputs_embeds, attention_mask, loss_kl


    def forward(self, x_hs=None, x_ts=None):#x_hs: code, x_ts: docstring
        bz = len(x_hs)

        if x_ts is not None:
            inputs, sum_idx, ext_inputs, unix_inputs = [], [], [], []
            for i in range(bz):
                input, idx, unix_input = self.get_query(x_hs[i], x_ts[i])#get_query函数返回的是一个tensor和一个idx,其中tensor是输入的query，idx是输入的query的长度
                inputs.append(input)
                sum_idx.append(idx)
                unix_inputs.append(unix_input)
            # print(len(inputs))
            # print(len(unix_inputs))
            inputs, inputs_embeds, attention_mask, loss_kl = self.prepare_inputs(inputs, unix_inputs)
            #print(inputs)
            output = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

            logits = output.logits
            loss = loss_kl

            for i in range(bz):
                idx = sum_idx[i]
                shift_logits = logits[i][idx:-1, :].contiguous()
                shift_labels = inputs[i][idx+1:].contiguous()
                
                if loss is None:
                    loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                else:
                    loss += self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            #print("\n================================", loss)
            #loss.requires_grad_(True)   
            loss = loss / bz
            #print(self.alpha)
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
                inputs, inputs_embeds, attention_mask, _ = self.prepare_inputs(queries, unix_inputs)

                output = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

                logits = output.logits# logits是一个三维的tensor，第一维是batch_size，第二维是token的长度，第三维是词表的大小
                #循环的目的是为了将每个batch的结果拼接到一起
                for i in range(bz):
                    idx = tmp_idx[i]
                    tmp_idx[i] += 1#tmp_idx是一个指针，指向每个batch的下一个token
                    next_token_logits = logits[i, idx:idx+1, :]#取出每个batch的下一个token的概率
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



class PromptAgent(torch.nn.Module):
    def __init__(self, template, hidden_size, tokenizer, device, args):
        super().__init__()
        self.device = device
        self.spell_length = sum(template)
        self.hidden_size = hidden_size
        self.embed_size = hidden_size
        self.tokenizer = tokenizer
        self.args = args
        # ent embedding
        self.cloze_length = template
        self.cloze_mask = [
            [1] * self.cloze_length[0]  # first cloze
            + [1] * self.cloze_length[1]  # second cloze
        ]
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool().to(self.device)
        self.seq_indices = torch.LongTensor(list(range(len(self.cloze_mask[0])))).to(self.device)

        if args.prompt_encoder_type == "lstm":
            self.prompt_encoder = Encoder_BiLSTM(input_size=self.hidden_size,
                                           hidden_size=self.hidden_size // 2,
                                           num_layers=2,
                                           dropout=0.0,
                                           bidirectional=True,
                                           batch_first=True)
        elif args.prompt_encoder_type == "transformer":
            self.prompt_encoder = Encoder_Transformer(d_model=self.hidden_size,
                                                  nhead=8,
                                                  num_layers=6,
                                                  max_len=len(self.cloze_mask[0]))


        self.embedding = torch.nn.Embedding(len(self.cloze_mask[0]), self.embed_size).to(self.device)
        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size))
        print("init prompt encoder...")

    def forward(self, input_embeds):
        #input_embeds = self.embedding(self.seq_indices).unsqueeze(0)
        input_embeds = input_embeds
        output_embeds = self.prompt_encoder(input_embeds)
        output_embeds = self.mlp_head(output_embeds).squeeze()
        return output_embeds


class Encoder_BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional, batch_first):
        super(Encoder_BiLSTM, self).__init__()
        self.lstm_head = torch.nn.LSTM(input_size=input_size,
                                           hidden_size=hidden_size,
                                           num_layers=num_layers,
                                           dropout=dropout,
                                           bidirectional=bidirectional,
                                           batch_first=batch_first)

    def forward(self, inputs):
        outputs = self.lstm_head(inputs)[0]

        return outputs


class Encoder_Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, max_len):
        super(Encoder_Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.pos_embedding = PositionalEncoding(d_model, 0.1, max_len)

    def forward(self, inputs):
        input_embedding = self.pos_embedding(inputs)
        input_embedding = input_embedding.permute(1, 0, 2)

        outputs = self.encoder(input_embedding).permute([1, 0, 2])

        return outputs

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2501):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [[0],[1],...[4999]] 5000 * 1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(
            10000.0) / d_model))  # e ^([0, 2,...,198] * -ln(10000)(-9.210340371976184) / 200) [1,0.912,...,(1.0965e-04)]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe

        return self.dropout(x)


def create_model(args, use_lm_finetune):
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    if not use_lm_finetune:
        model = model.half()
    return model


def get_embedding_layer(args, model):
    return model.base_model.get_input_embeddings()
        


