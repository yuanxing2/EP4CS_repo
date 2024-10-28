import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers import BertTokenizer
#models.
from models.Qformer import BertConfig, BertLMHeadModel

class CodeBlip2(nn.Module):
    def __init__(
        self,
        num_features=768,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        tokenizer=None,
    ):
        super().__init__()

        #self.tokenizer = self.init_tokenizer()
        self.tokenizer = tokenizer
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #获取设备

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, num_features, cross_attention_freq
        ) #num_query_token是query的长度，num_features是嵌入的长度，cross_attention_freq是每隔几个block插入一个cross-attention
        self.Qformer.resize_token_embeddings(len(self.tokenizer)) #调整嵌入的大小到len(self.tokenizer)
        state_dict = self.Qformer.state_dict() #获取Qformer的参数
        for name, param in self.Qformer.named_parameters(): #如果参数名中包含_query，将_query去掉，然后将参数复制到_query对应的参数中
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.code_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim) #将将num_features的768映射到embed_dim，embed_dim是256
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim) #将num_features的768映射到embed_dim

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2) #将num_features的768映射到2

        self.temp = nn.Parameter(0.07 * torch.ones([])) #temp是一个参数，初始化为0.07

        self.max_txt_len = max_txt_len #max_txt_len是文本的最大长度
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        ) 

    def init_Qformer(self, num_query_token, vision_width, cross_attention_freq=2):

        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token 
        Qformer = BertLMHeadModel.from_pretrained("bert-base-uncased", config = encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens
    

    def forward(self, text_tokens, code_features, text_tokens_attention_mask, code_att_codebert):
        code_embeds = code_features
        code_embeds = code_embeds.to(self.device)# 放入cuda
        code_atts = code_att_codebert
        print(code_embeds.shape)
        
        query_tokens = self.query_tokens.expand(code_embeds.shape[0], -1, -1)
        
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=code_embeds,
            encoder_attention_mask=code_atts,
            use_cache=True,#作用是是否使用cache
            return_dict=True,#作用是是否返回字典
        )
        
        code_feats = F.normalize(
            self.code_proj(query_output.last_hidden_state), dim=-1
        )

        text_output = self.Qformer.bert( #执行Qformer的forward
            text_tokens,
            attention_mask=text_tokens_attention_mask,
            return_dict=True,
        )
        
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )
        
        code_feats_all = code_feats
        text_feat_all = text_feat
        sim_q2t = torch.matmul(
            code_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        sim_b2t, _ = sim_q2t.max(-1)
        sim_b2t = sim_b2t / self.temp

        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), code_feats_all.permute(0, 2, 1)
        ).squeeze()


        sim_t2b, _ = sim_t2q.max(-1)
        sim_t2b = sim_t2b / self.temp 

        rank = 0
        bs = code_feats.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            code_feats.device
        )

        loss_itc = (
            F.cross_entropy(sim_b2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2b, targets, label_smoothing=0.1)
        ) / 2




        
        text_input_ids_world = text_tokens
        text_attention_mask_world = text_tokens_attention_mask
        behavior_embeds_world = code_embeds
        #print(code_feats.shape)
        with torch.no_grad():
            weights_t2i = F.softmax(sim_t2b, dim=1) + 1e-4
            weights_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(0)
            weights_i2t = F.softmax(sim_b2t, dim=1) + 1e-4
            weights_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(0)

        # select a negative image for each text
        behavior_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            behavior_embeds_neg.append(behavior_embeds_world[neg_idx])
        behavior_embeds_neg = torch.stack(behavior_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [text_tokens, text_tokens, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_tokens_attention_mask, text_tokens_attention_mask, text_atts_neg],
            dim=0,
        )

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            code_embeds.device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        behavior_embeds_all = torch.cat(
            [code_embeds, behavior_embeds_neg, code_embeds], dim=0
        )  # pos, neg, pos
        behavior_atts_all = torch.ones(behavior_embeds_all.size()[:-1], dtype=torch.long).to(
            code_embeds.device
        )
        # print(code_feats.shape)
        # print(text_ids_all.shape)
        # print(query_tokens_itm.shape)
        # print(behavior_embeds_all.shape)
        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=behavior_embeds_all,
            encoder_attention_mask=behavior_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(code_embeds.device)
        loss_itm = F.cross_entropy(logits, itm_labels)




        decoder_input_ids = text_tokens.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id 
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            code_embeds.device
        )
        attention_mask = torch.cat([query_atts, text_tokens_attention_mask], dim=1)
        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )
        
        loss_lm = lm_output.loss
        loss = loss_itc + loss_itm + loss_lm
        return loss, loss_itc, loss_itm, loss_lm
    
    def concat_all_gather(nums):
        return nums

    @torch.no_grad()
    def eval_me(self, text_tokens, code_features, text_tokens_attention_mask):
        code_embeds = code_features
        code_embeds = code_embeds.to(self.device)# 放入cuda
        code_atts = torch.ones(code_embeds.size()[:-1], dtype=torch.long).to(code_embeds.device) #生成一个和behavior_embeds相同大小的tensor，值全为1
        query_tokens = self.query_tokens.expand(code_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=code_embeds,
            encoder_attention_mask=code_atts,
            use_cache=True,#作用是是否使用cache
            return_dict=True,#作用是是否返回字典
        )
        code_feats = F.normalize(
            self.code_proj(query_output.last_hidden_state[:, 0, :]), dim=-1
        )
        text_output = self.Qformer.bert( #执行Qformer的forward
            text_tokens,
            attention_mask=text_tokens_attention_mask,
            return_dict=True,
        )
        
        text_feats = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )
        return code_feats, text_feats


