# -*- coding:utf-8 -*-
from transformers import BertModel, BertPreTrainedModel, BertConfig
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, KLDivLoss, NLLLoss, BCELoss
from utils.loss_utils import ReverseLayerF

class Span_Detector(BertPreTrainedModel):
    def __init__(self, config, span_num_labels, device):
        super().__init__(config)

        self.device_ = device # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.span_num_labels = span_num_labels
        # self.type_num_labels_src = type_num_labels_src+1
        # self.type_num_labels_tgt = type_num_labels_tgt+1
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hidden_size = config.hidden_size
        # self.span = nn.Parameter(torch.randn(type_num_labels, config.hidden_size, span_num_labels))
        # self.classifier_bio = nn.Linear(config.hidden_size, span_num_labels)
        self.classifier_bio_src = nn.Linear(config.hidden_size, span_num_labels)
        self.classifier_bio_tgt = nn.Linear(config.hidden_size, span_num_labels)
        # self.classifier_m_src = nn.Linear(config.hidden_size, type_num_labels_src)
        # self.classifier_m_tgt = nn.Linear(config.hidden_size, type_num_labels_tgt)
        # self.discriminator = nn.Linear(config.hidden_size, 2)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels_bio=None,
        tgt=True,
        reduction="none",
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
        # print(outputs[0][0])
        # # print(outputs[1].size())
        # # print(len(outputs[2]))
        # print(outputs[2][-1][0])
        # print(outputs[2][0].size())
        # print(outputs[2][1].size())
        # exit()
        final_embedding = outputs[0] # B, L, D
        sequence_output1 = self.dropout(final_embedding)
        # sequence_output2 = self.dropout(final_embedding)
        # reverse_feature = ReverseLayerF.apply(sequence_output2, alpha)
        # logits_domain = self.discriminator(reverse_feature) # B, L, 2
        # loss_fct = CrossEntropyLoss()
        # logits_size = logits_domain.size()
        # labels_domain = torch.zeros(logits_size[0]*logits_size[1]).long().to(self.device_)
        if tgt:
            logits_bio = self.classifier_bio_tgt(sequence_output1) # B, L, C
            # labels_domain = labels_domain + 1
            # loss_domain = loss_fct(logits_domain.view(-1, 2), labels_domain)

        else:
            logits_bio = self.classifier_bio_tgt(sequence_output1) # B, L, C
            # loss_domain = loss_fct(logits_domain.view(-1, 2), labels_domain)
        
        outputs = (logits_bio, final_embedding, ) + outputs[2:]  # add hidden states and attention if they are here
        # outputs = (loss_domain, logits_bio, final_embedding, ) + outputs[2:]  # add hidden states and attention if they are here

        if labels_bio is not None:
            # logits = self.logsoftmax(logits)
            # Only keep active parts of the loss
            active_loss = True
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
            
            active_logits = logits_bio.view(-1, self.span_num_labels)[active_loss]

            loss_fct = CrossEntropyLoss(reduction=reduction)
            if attention_mask is not None:
                active_labels = labels_bio.view(-1)[active_loss]
                loss_bio = loss_fct(active_logits, active_labels)
            else:
                loss_bio = loss_fct(logits_bio.view(-1, self.span_num_labels), labels_bio.view(-1))

            outputs = (loss_bio, active_logits,) + outputs

        return outputs

    def loss(self, loss_bio, logits_type, tau=0.1, eps=0.0):
        # loss_bio: B*L
        # logits_type: B*L, C
        logits_type = torch.softmax(logits_type.detach()/tau, dim=-1)
        weight = 1.0-logits_type[:, -1]+eps
        loss = torch.mean(loss_bio*weight)
        return loss

    def adv_attack(self, emb, loss, mu):
        loss_grad = torch.autograd.grad(loss, emb, retain_graph=True)[0]
        loss_grad_norm = torch.sqrt(torch.sum(loss_grad**2, dim=2))
        perturbed_sentence = emb + mu * (loss_grad/(loss_grad_norm.unsqueeze(2)+1e-5))
        
        return perturbed_sentence
