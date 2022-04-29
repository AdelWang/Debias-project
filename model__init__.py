import torch.nn as nn
import torch
import numpy as np
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
        
class predictor_overlap(nn.Module):
    def __init__(self, config):
        super(predictor_overlap,self).__init__()
        self.encoder_hidden = nn.Linear(1, int(config.hidden_size / 2))
        self.encoder = nn.Linear(int(config.hidden_size / 2), config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation_hidden = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
    def forward(self, overlap, label=None,loss_fn=nn.CrossEntropyLoss(),output_representation=True):
        overlap = overlap.reshape(-1,1).float()
        encoded_rep = self.encoder_hidden(overlap)
        encoded_rep = self.activation_hidden(encoded_rep)
        encoded_rep = self.encoder(self.dropout(encoded_rep))
        outputs_embed = self.activation_hidden(encoded_rep)
        logits = self.classifier(outputs_embed)
        loss = loss_fn(logits,label)
        
        
        return (logits, loss, outputs_embed) if output_representation else (logits, loss)
        
class encoder_overlap(nn.Module):
    def __init__(self, config):
        super(encoder_overlap,self).__init__()
        self.encoder_hidden = nn.Linear(1, int(config.hidden_size / 2))
        self.encoder = nn.Linear(int(config.hidden_size / 2), config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation_hidden = nn.Tanh()
        
    def forward(self, overlap):
        overlap = overlap.reshape(-1,1).float()
        encoded_rep = self.encoder_hidden(overlap)
        encoded_rep = self.activation_hidden(encoded_rep)
        encoded_rep = self.encoder(self.dropout(encoded_rep))
        outputs_embed = self.activation_hidden(encoded_rep)
        
        return outputs_embed
            
        
class label_lookup(nn.Module):
    def __init__(self,config):
        super(label_lookup,self).__init__()
        self.encoder_table = nn.Parameter(torch.randn(config.num_labels,config.hidden_size))
        self.one_hot_table = torch.sparse.torch.eye(config.num_labels).cuda()
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = nn.Sigmoid()
        
    def forward(self, label, loss_fn=nn.CrossEntropyLoss(), output_representation=True):
        one_hot_label = self.one_hot_table.index_select(0, label)
        return_embed = torch.matmul(one_hot_label, self.encoder_table)
        logits = self.classifier(return_embed)
        logits = self.activation(logits)
        loss = loss_fn(logits,loss)
        
        return (logits, loss, return_embed) if output_representation else (logits, loss)
        
class finetune_layer(nn.Module):
    def __init__(self, config):
        super(finetune_layer,self).__init__()
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, inputs):
    
        logits = self.classifier(self.dropout(inputs))
        return logits
        
class contrastive_model(nn.Module):
    def __init__(self,temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class BertClassifier(nn.Module):
    def __init__(self,bert, config):
        super(BertClassifier,self).__init__()
        self.bert = bert
        self.config = config
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        loss_fn = nn.CrossEntropyLoss()
    ):
    
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            )
        last_hidden_state = outputs.hidden_states[-1]
        feature = torch.mean(last_hidden_state,axis=1)
        logits = self.classifier(self.dropout(feature))
        if labels is not None:
            loss = loss_fn(logits,labels)
        else:
            loss = None
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
