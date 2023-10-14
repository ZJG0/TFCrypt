from unittest import result
import matplotlib.pyplot as plt
import pylab as pl
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

import sys
import collections
import d2l
import zipfile
from d2l.data.base import Vocab

import time
import torch.nn.functional as F
from torch.utils import data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

# Transformer Parameters (PS.服务器上标准transformer所用参数)
# d_model = 512  # Embedding Size
# d_ff = 2048  # FeedForward dimension(一般是词向量的4倍表示起来比较好)
# d_k = d_v = 64  # dimension of K(=Q), V
# n_layers = 6  # number of Encoder of Decoder Layer
# n_heads = 8  # number of heads in Multi-Head Attention

# Transformer Parameters (Tiny)
# d_model = 64  # Embedding Size
# d_ff = 256  # FeedForward dimension(一般是词向量的4倍表示起来比较好)
# d_k = d_v = 36  # dimension of K(=Q), V
# n_layers = 3  # number of Encoder of Decoder Layer
# n_heads = 4  # number of heads in Multi-Head Attention

# Transformer Parameters (Base)
d_model = 256  # Embedding Size
d_ff = 1024  # FeedForward dimension(一般是词向量的4倍表示起来比较好)
d_k = d_v = 49  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 6  # number of heads in Multi-Head Attention

# Transformer Parameters (Base)
# d_model = 256  # Embedding Size
# d_ff = 1024  # FeedForward dimension(一般是词向量的4倍表示起来比较好)
# d_k = d_v = 64  # dimension of K(=Q), V
# n_layers = 6  # number of Encoder of Decoder Layer
# n_heads = 8  # number of heads in Multi-Head Attention

# Transformer Parameters (Large)
# d_model = 512  # Embedding Size
# d_ff = 2048  # FeedForward dimension(一般是词向量的4倍表示起来比较好)
# d_k = d_v = 100  # dimension of K(=Q), V
# n_layers = 9  # number of Encoder of Decoder Layer
# n_heads = 12  # number of heads in Multi-Head Attention
# -----------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        # print("--------------PositionalEncoding Initing--------------")
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        # print("--------------PositionalEncoding Forwarding--------------")
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# def get_attn_pad_mask(seq_q, seq_k):
#     '''
#     seq_q: [batch_size, seq_len]
#     seq_k: [batch_size, seq_len]
#     seq_len could be src_len or it could be tgt_len
#     seq_len in seq_q and seq_len in seq_k maybe not equal
#     '''
#     batch_size, len_q = seq_q.size()
#     batch_size, len_k = seq_k.size()
#     # eq(zero) is PAD token
#     # seq_k_f = seq_k.to(torch.float64)
#     # pad_attn_mask_new = torch.where(seq_k_f.data.eq(0), -1e9, 0).unsqueeze(1)
#     pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
#     return pad_attn_mask.expand(batch_size, len_q, len_k)#, pad_attn_mask_new.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

# def get_attn_subsequence_mask(seq):
#     '''
#     seq: [batch_size, tgt_len]
#     '''
#     attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
#     subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
#     subsequence_mask = torch.from_numpy(subsequence_mask).byte()
#     return subsequence_mask # [batch_size, tgt_len, tgt_len]


# --------------------------------------------------------------

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        # print("--------------ScaledDotProductAttention Initing--------------")  
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        # print("--------------ScaledDotProductAttention Forwarding--------------")
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, len_q, len_k]
        # print(attn_mask)
        masked_scores = scores.masked_fill_(attn_mask, -100) # Fills elements of self tensor with value where mask is True.
        # attn_mask_value = torch.where(attn_mask, -1e9, 0.)
        # filename = '/root/PPTF/outputs-plaintext.txt'
        # with open(filename,'a') as f:
        #     f.write("mask:\n")
        #     f.write(str(attn_mask))
        #     f.write("\n")
        #     f.write("masked:\n")
        #     f.write(str(attn_mask_value))
        #     f.write("\n")
        #     f.write("--------------------------------------------------------------------------------------------------\n")
        # scores = scores+attn_mask_value
        # scores[attn_mask>0] = -1e9
        # test = scores.unsqueeze(1).repeat(1,1,1,1,1)
        attn = self.softmax(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        # print("--------------MultiHeadAttention Initing--------------")
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.LN = nn.LayerNorm(d_model)
    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        # print("--------------MultiHeadAttention Forwarding--------------")
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # seq_q = input_Q.shape[1]
        # seq_k = input_K.shape[1]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        # return output, attn #Temp
        # print(output.size())
        # print(residual.size())
        return self.LN(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        # print("--------------PoswiseFeedForwardNet Initing--------------")
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.LN = nn.LayerNorm(d_model)
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        # print("--------------PoswiseFeedForwardNet Forwarding--------------")
        residual = inputs
        output = self.fc(inputs)
        return self.LN(output + residual) # [batch_size, seq_len, d_model]

class EncoderLayer(nn.Module):
    def __init__(self):
        # print("--------------EncoderLayer Initing--------------")
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # print("--------------EncoderLayer Forwarding--------------")
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        # return enc_outputs, attn #Temp
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn

class DecoderLayer(nn.Module):
    def __init__(self):
        # print("--------------DecoderLayer Initing--------------")
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        # print("--------------DecoderLayer Forwarding--------------")
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs) # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn

class Encoder(nn.Module):
    def __init__(self, src_vocab_size):
        # print("--------------Encoder Initing--------------")
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        # print("--------------Encoder Forwarding--------------")
        enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        ############################ Generate pad_mask ##########################################
        batch_size, len_q = enc_inputs.size()
        batch_size, len_k = enc_inputs.size()
        # eq(zero) is PAD token
        # seq_k_f = seq_k.to(torch.float64)
        # pad_attn_mask_new = torch.where(seq_k_f.data.eq(0), -1e9, 0).unsqueeze(1)
        pad_attn_mask = enc_inputs.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
        enc_self_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k)
        # enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # [batch_size, src_len, src_len]
        ##########################################################################################
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size):
        # print("--------------Decoder Initing--------------")
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        # print("--------------Decoder Forwarding--------------")
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        '''
        dec_outputs = self.tgt_emb(dec_inputs) # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).to(device) # [batch_size, tgt_len, d_model]
        
        ############################ Generate pad_mask ##########################################
        batch_size, len_q = dec_inputs.size()
        batch_size, len_k = dec_inputs.size()
        # eq(zero) is PAD token
        # seq_k_f = seq_k.to(torch.float64)
        # pad_attn_mask_new = torch.where(seq_k_f.data.eq(0), -1e9, 0).unsqueeze(1)
        pad_attn_mask = dec_inputs.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
        dec_self_attn_pad_mask = pad_attn_mask.expand(batch_size, len_q, len_k)
        # dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).to(device)# [batch_size, tgt_len, tgt_len]
        ##########################################################################################
        
        
        ############################ Generate pad_mask ##########################################
        attn_shape = [dec_inputs.size(0), dec_inputs.size(1), dec_inputs.size(1)]
        # subsequence_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # Upper triangular matrix diagonal
        x = torch.ones(attn_shape)
        diagonal = 1
        l = x.shape[1]
        # bs = batch_size
        arange = torch.arange(l, device=x.device).unsqueeze(0)
        mask = arange.expand(batch_size, l, l)
        arange = arange.unsqueeze(-1)
        if diagonal:
            arange = arange + diagonal
        mask = mask >= arange
        dec_self_attn_subsequence_mask = mask * x
        
        # subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
        # dec_self_attn_subsequence_mask = torch.zeros(attn_shape)
        # dec_self_attn_subsequence_mask = torch.ones(attn_shape)
        # dec_self_attn_subsequence_mask = torch.from_numpy(subsequence_mask).byte()
        # dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).to(device) # [batch_size, tgt_len, tgt_len]
        ##########################################################################################
        
        
        # print(dec_self_attn_pad_mask.size())
        # print(dec_self_attn_subsequence_mask.size())
        # print("--"*50)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0).to(device) # [batch_size, tgt_len, tgt_len]
        ############################ Generate pad_mask ##########################################
        batch_size, len_q = dec_inputs.size()
        batch_size, len_k = enc_inputs.size()
        # eq(zero) is PAD token
        # seq_k_f = seq_k.to(torch.float64)
        # pad_attn_mask_new = torch.where(seq_k_f.data.eq(0), -1e9, 0).unsqueeze(1)
        pad_attn_mask = enc_inputs.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
        dec_enc_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k)
        # dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs) # [batc_size, tgt_len, src_len]
        ##########################################################################################

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size):
        # print("--------------Transformer Initing--------------")
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size).to(device)
        self.decoder = Decoder(tgt_vocab_size).to(device)
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).to(device)
    def forward(self, enc_inputs, dec_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        # print("--------------Transformer Forwarding--------------")
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        
        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # print(enc_outputs)
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs) # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1))