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

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
# ----------------------------------------------

def preprocess(num_examples):
    #读取平行语句对，规范格式
    with open('/root/PPTF/application/saved/raw_data/fra.txt', 'r',encoding='UTF-8') as f:
        text = f.read() 
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    out = ''
    for i, char in enumerate(text.lower()):
        if char in (',', '!', '.') and i > 0 and text[i-1] != ' ':
            out += ' '
        out += char
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) >= 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' ')) 
    # print(source[0:6])
    # print(target[0:6])
    print("(文件读取完成，共读取{}个平行语句对)".format(len(source)-1))
    return source,target



def pad(line, max_len, padding_token):
    if len(line) > max_len:
        return line[:max_len]
    return line + [padding_token] * (max_len - len(line))

def build_vocab(tokens):
    tokens = [token for line in tokens for token in line]
    return d2l.data.base.Vocab(tokens, min_freq=3, use_special_tokens=True)

def build_array(lines, w2i, max_len, option):
    lines = [[w2i.get(w,w2i['<unk>'])for w in line] for line in lines]  
    if option==0:
        #enc_input
        pass
    elif option==1:
        lines = [[w2i['<bos>']] + line for line in lines] #dec_input
    else:
        lines = [line + [w2i['<eos>']] for line in lines] #dec_output
    
    array = torch.LongTensor([pad(line, max_len, w2i['<pad>']) for line in lines])
    return array  

def make_data(max_len, source, target): 
    # 读取静态词典
    src_w2i, _, target_w2i, _ = get_static_vocab() 
    enc_inputs = build_array(source, src_w2i, max_len, 0)
    dec_inputs = build_array(target, target_w2i, max_len, 1)
    dec_outputs = build_array(target, target_w2i, max_len, 2) 
    return enc_inputs, dec_inputs, dec_outputs


class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs
  
    def __len__(self):
        return self.enc_inputs.shape[0]
  
    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]
    
# ----------------------------------------------
 
def sign_filter(list_with_sign):
    # 去除特殊标识<xx>
    clean_list=[]
    for word in list_with_sign:
        if word!='<pad>'and word!='<bos>'and word!='<eos>'and word!='<unk>':
            clean_list.append(word)
    return clean_list


def static_vocab(num_examples=177210):
    # ps.为了简便，把读取数据的路径写死在了 preprocess.py 的 preprocess 函数中
    
    source,target = preprocess(num_examples)
    src_vocab, tgt_vocab = build_vocab(source), build_vocab(target)

    src_w2i = src_vocab.token_to_idx
    src_i2w = {i: w for i, w in enumerate(src_vocab.idx_to_token)}
    
    w2i = tgt_vocab.token_to_idx
    i2w = {i: w for i, w in enumerate(tgt_vocab.idx_to_token)}
    
    # 存储词典文件
    np.save('save/vocab_saved/src_w2i.npy', src_w2i, allow_pickle='TRUE')
    np.save('save/vocab_saved/src_i2w.npy', src_i2w, allow_pickle='TRUE')  
    np.save('save/vocab_saved/tgt_w2i.npy', w2i, allow_pickle='TRUE')
    np.save('save/vocab_saved/tgt_i2w.npy', i2w, allow_pickle='TRUE') 
    print("(静态字典【存储】完成，相对路径为：save/vocab_saved)")
    

def get_static_vocab():
    src_w2i = np.load('/root/PPTF/application/saved/vocabulary/src_w2i.npy', allow_pickle='TRUE').item()
    src_i2w = np.load('/root/PPTF/application/saved/vocabulary/src_i2w.npy', allow_pickle='TRUE').item()  
    w2i = np.load('/root/PPTF/application/saved/vocabulary/tgt_w2i.npy', allow_pickle='TRUE').item()
    i2w = np.load('/root/PPTF/application/saved/vocabulary/tgt_i2w.npy', allow_pickle='TRUE').item()
    print("(静态字典读取完成。source字典大小：{}, target字典大小：{})".format(len(src_w2i),len(w2i)))
    return src_w2i, src_i2w, w2i, i2w


   