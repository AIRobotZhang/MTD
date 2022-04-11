# -*- coding:utf-8 -*-
from transformers import BertModel, BertPreTrainedModel, BertConfig
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, KLDivLoss, NLLLoss
import torch.nn.functional as F

class Type_Classifier(BertPreTrainedModel):
    def __init__(self, config, type_num_labels_src, type_num_labels_tgt, device, domain):
        super().__init__(config)

        self.device_ = device # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.span_num_labels = span_num_labels
        self.type_num_labels_src = type_num_labels_src+1
        self.type_num_labels_tgt = type_num_labels_tgt+1
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hidden_size = config.hidden_size
        # self.W = nn.Parameter(torch.randn(config.hidden_size, 300))
        # self.base = nn.Parameter(torch.randn(8, 300))
        # self.span = nn.Parameter(torch.randn(type_num_labels, config.hidden_size, span_num_labels))
        # self.classifier_bio = nn.Linear(config.hidden_size, span_num_labels)
        # self.classifier_bio_src = nn.Linear(config.hidden_size, span_num_labels)
        # self.classifier_bio_tgt = nn.Linear(config.hidden_size, span_num_labels)
        self.classifier_type_src = nn.Linear(config.hidden_size, type_num_labels_src+1)
        self.classifier_type_tgt = nn.Linear(config.hidden_size, type_num_labels_tgt+1)
        self.classifier_type = nn.Linear(config.hidden_size, type_num_labels_tgt)
        domain_map = {
                       "politics":torch.tensor([3,4,5,6]).long().to(device),
                       "science":torch.tensor([9,10,11,12]).long().to(device),
                       "music":torch.tensor([5,6,10,11]).long().to(device),
                       "literature":torch.tensor([5,7,8,9]).long().to(device),
                       "ai":torch.tensor([4,6,7,8]).long().to(device)
                   }
        self.label_ind_map = domain_map[domain]
        # self.label_ind_map = torch.tensor([4,6,7,8]).long().to(device) # ai
        # self.label_ind_map = torch.tensor([5,7,8,9]).long().to(device) # literature
        # self.label_ind_map = torch.tensor([5,6,10,11]).long().to(device) # music
        # self.label_ind_map = torch.tensor([9,10,11,12]).long().to(device) # science
        # self.label_ind_map = torch.tensor([3,4,5,6]).long().to(device) # politics


        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels_type=None,
        logits_bio=None,
        tgt=True,
        reduction="none"
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
        )
        final_embedding = outputs[0] # B, L, D
        # print(final_embedding.size())
        # sequence_output1 = self.dropout(final_embedding)
        sequence_output2 = self.dropout(final_embedding)
        # bs, l, d = sequence_output2.size()
        # alpha1 = torch.bmm(sequence_output2, self.W.unsqueeze(0).repeat(bs, 1, 1)) # N, L, d
        # alpha2 = torch.bmm(alpha1, self.base.t().unsqueeze(0).repeat(bs, 1, 1)) # N, L, C_base
        # alpha = torch.softmax(alpha2, dim=-1) # N, L, C_base
        # seq_out = torch.bmm(alpha, self.base.unsqueeze(0).repeat(bs, 1, 1)) # N, L, d_new
        # seq_embed = sequence_output.view(-1, self.hidden_size) # B*L, D
        # seq_size = seq_embed.size()
        # logits = torch.bmm(seq_embed.unsqueeze(0).expand(self.type_num_labels, 
        #                 seq_size[0], seq_size[1]), self.span).permute(1, 0, 2) # B*L, type_num, span_num
        type_num_labels = self.type_num_labels_tgt
        if tgt:
            # logits_bio = self.classifier_bio_tgt(sequence_output1) # B, L, C
            logits_type = self.classifier_type_tgt(sequence_output2)
        else:
            # logits_bio = self.classifier_bio_src(sequence_output1) # B, L, C
            # logits_bio = self.classifier_bio_tgt(sequence_output1) # B, L, C
            # logits_type = self.classifier_type_tgt(sequence_output2)[:,:,self.label_ind_map]
            logits_type = self.classifier_type_src(sequence_output2)
            type_num_labels = self.type_num_labels_src
        # logits = self.classifier_bio_tgt(sequence_output1) # B, L, C
        # print(logits.size())

        outputs = (logits_type, final_embedding, ) + outputs[2:]  # add hidden states and attention if they are here
        # logits_bio = torch.argmax(logits_bio, dim=-1)

        if labels_type is not None:
            # logits = self.logsoftmax(logits)
            # Only keep active parts of the loss
            active_loss = True
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1

            # bio_labels = logits_bio.view(-1)
            # label_mask = bio_labels < 2
            # active_loss = active_loss&label_mask
            
            active_logits = logits_type.view(-1, type_num_labels)[active_loss]

            loss_fct = CrossEntropyLoss(reduction=reduction)
            # if attention_mask is not None:
            active_labels = labels_type.view(-1)[active_loss]
            if len(active_labels) == 0:
                loss_type = torch.tensor(0).float().to(self.device_)
            else:
                loss_type = loss_fct(active_logits, active_labels)
            # else:
            #     loss_type = loss_fct(logits_type.view(-1, type_num_labels), labels_type.view(-1))


            outputs = (loss_type, active_logits,) + outputs

        return outputs

    def loss(self, loss_type, logits_bio, tau=0.1, eps=0.0):
        # loss_type: B*L 
        # logits_bio: B*L, C
        # delta: music 0.1
        logits_bio = torch.softmax(logits_bio.detach()/tau, dim=-1)
        weight = 1.0-logits_bio[:, -1]+eps
        loss = torch.mean(loss_type*weight)
        return loss


    def mix_up(self, src_rep, tgt_rep, src_label, tgt_label, alpha, beta):
        num_labels_src = self.type_num_labels_src-1
        num_labels_tgt = self.type_num_labels_tgt-1
        src_tgt_map = self.label_ind_map

        rep_dim = src_rep.size()[-1]
        src_rep = src_rep.view(-1, rep_dim) # B*L, d
        tgt_rep = tgt_rep.view(-1, rep_dim) # B*L, d
        src_label = src_label.view(-1) # B*L
        tgt_label = tgt_label.view(-1) # B*L
        mask_src = (src_label<num_labels_src)&(src_label>=0)
        mask_tgt = (tgt_label<num_labels_tgt)&(tgt_label>=0)
        src_sel = src_rep[mask_src] # N1, d
        tgt_sel = tgt_rep[mask_tgt] # N2, d
        src_label_sel = src_label[mask_src] # N1
        tgt_label_sel = tgt_label[mask_tgt] # N2
        N1 = src_sel.size()[0]
        N2 = tgt_sel.size()[0]
        src_exp = src_sel.unsqueeze(0).repeat(N2, 1, 1).view(N1*N2, -1)
        tgt_exp = tgt_sel.unsqueeze(1).repeat(1, N1, 1).view(N1*N2, -1)
        src_label_exp = src_label_sel.unsqueeze(0).repeat(N2, 1).view(-1).unsqueeze(1) # N1*N2, 1
        tgt_label_exp = tgt_label_sel.unsqueeze(1).repeat(1, N1).view(-1).unsqueeze(1) # N1*N2, 1
        src_onehot_ = torch.zeros(N1*N2, num_labels_src).to(self.device_).scatter_(1, src_label_exp, 1)
        tgt_onehot = torch.zeros(N1*N2, num_labels_tgt).to(self.device_).scatter_(1, tgt_label_exp, 1)
        src_onehot = torch.zeros(N1*N2, num_labels_tgt).to(self.device_)
        src_onehot[:, src_tgt_map] = src_onehot_
        mix_rep = alpha*src_exp + beta*tgt_exp
        mix_label = alpha*src_onehot + beta*tgt_onehot
        # print(mix_label)
        # exit()
        logits_type = F.log_softmax(self.classifier_type(mix_rep), dim=-1) # N1*N2, C

        loss_fct = KLDivLoss()

        loss = loss_fct(logits_type, mix_label)

        return loss

    def adv_attack(self, emb, loss, mu):
        loss_grad = torch.autograd.grad(loss, emb, retain_graph=True)[0]
        loss_grad_norm = torch.sqrt(torch.sum(loss_grad**2, dim=2))
        perturbed_sentence = emb + mu * (loss_grad/(loss_grad_norm.unsqueeze(2)+1e-5))
        
        return perturbed_sentence
