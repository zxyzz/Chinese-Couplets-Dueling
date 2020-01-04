#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *

#Resource
#https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
class EncoderRNN(nn.Module):
    def __init__(self, input_size,hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size,padding_idx = 0)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first = True,bidirectional = True)

    def forward(self, input, hidden,longueur):
        # Embedding
        embedded = self.embedding(input)
        # pack padded input
        output_packed = nn.utils.rnn.pack_padded_sequence(embedded, longueur, batch_first=True,enforce_sorted=False)
        # Put into GRU
        output,       hidden = self.gru(   output_packed,         hidden)
        #[bs,seq_inBatch,hs] [2,bs,hs]   #[bs,seq_inBatch,hs]    #[2,bs,hs]
        # Pad again the output of GRU
        output_padded, _ = nn.utils.rnn.pad_packed_sequence(output,batch_first=True)

        #[bs,seq_inBatch,2*hs] [2,bs,hs]
        return output_padded, hidden


