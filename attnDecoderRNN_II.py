#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from attention_II import Attention_II

# Resource
#https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
class AttnDecoderRNN_II(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN_II, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(self.output_size, self.hidden_size,padding_idx = 0)
        self.attn_combine = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first = True,bidirectional = True)
        
        self.out = nn.Linear(self.hidden_size*2, self.output_size)
        
        self.attention = Attention_II(self.hidden_size, attention_type='general') # 'general' or 'dot'

    def forward(self, input, hidden,  encoder_outputs): 
                   # [bs]    # [2,bs,hs]   #[bs,seq,2*hs] 
        # Embedding
        embedded = self.embedding(input)  #[bs,hs]
        embedded = self.dropout(embedded)
        # convert from [bs,hs] to [bs,1,hs], in order to be compatible with hidden layer in gru
        output = embedded.unsqueeze(1)
        # Compute attention weights
        attn_weights = self.attention(torch.transpose(hidden,0,1),    encoder_outputs) 
        # Combine input with attention score
        attn_applied = torch.bmm(attn_weights, encoder_outputs)
        output =       torch.cat(        (output, attn_applied), 2)
        # Reshape output by a linear layer
        output = self.attn_combine(output)
        output = F.relu(output)
        # Put into GRU
        output,        hidden = self.gru(output,          hidden)
        # Compute final output
        output = F.log_softmax(self.out(output), dim=2)

        #[bs, 1, vocab_size]  # [1,bs,hs_]   #[bs,1,seq]  
        return output,        hidden,      attn_weights

