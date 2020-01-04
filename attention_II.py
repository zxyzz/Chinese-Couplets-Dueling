# Resource
#https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/attention.html
import torch
import torch.nn as nn

class Attention_II(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention_II, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()


    def forward(self, query, context):

        #query (hidden): [bs,2,hs] 
        #context (enc_out): [bs,seq,2*hs]
        batch_size, output_len, dimensions = query.shape
        query_len = context.shape[1]
        if self.attention_type == "general":
            query = self.linear_in(query) #[bs,1,2*hs]
            query = query.reshape(batch_size,dimensions*2).unsqueeze(1)

        # Compute weights across every context sequence
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())
        attention_scores = attention_scores.view(batch_size * 1, query_len)
        # Masking padding indices
        attention_scores = attention_scores.masked_fill(attention_scores ==0, -10e9)
        # Apply softmax over weights
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, 1, query_len)

        return attention_weights

