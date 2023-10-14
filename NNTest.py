'''
Author: ZJG
Date: 2022-07-14 15:46:27
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-10-14 10:36:49
'''
from cProfile import label
import TFCrypt
import TFCrypt.mpc as mpc
import TFCrypt.communicator as comm
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
from application.model_debug import*
from application.preprocess import*
import onnx
import copy
import io
from torch.onnx import OperatorExportTypes
import numpy as np

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
def train(epochs, inputs, labels, model, device,Lr,momen):
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=Lr, momentum=momen)
  model.to(device)
  for e in range(epochs):
    imgs = inputs.to(device)
    labels = labels.to(device)
    out = model(imgs)
#     print(out)
    loss = criterion(out, labels)
    optimizer.zero_grad() # if don't call zero_grad, the grad of each batch will be accumulated
    loss.backward()
    optimizer.step()
  torch.save(model, 'testMask.pth') # save net model and parameters
  
def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    # zero_tensor = torch.zeros(seq_k.size())
    pad_attn_mask = seq_k.data.eq(0)  # [batch_size, 1, len_k], False is masked
    result = pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]
    return result
def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # subsequence_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # Upper triangular matrix diagonal
    x = torch.ones(attn_shape)
    diagonal = 1
    l = x.shape[1]
    bs = x.shape[0]
    arange = torch.arange(l, device=x.device)
    mask = arange.expand(bs, l, l)
    arange = arange.unsqueeze(-1)
    if diagonal:
        arange = arange + diagonal
    mask = mask >= arange
    return mask * x
    
    
    # subsequence_mask = torch.ones(attn_shape)
    # subsequence_mask = torch.zeros(attn_shape)
    # subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask # [batch_size, tgt_len, tgt_len]

class NN(nn.Sequential):
    # network architecture:
    def __init__(self):
        super(NN, self).__init__()
        # self.mask_sub = mask_sub
        self.fc = nn.Linear(3, 4)
    def forward(self, attn):
        # attn_size = attn.size()
        # mask_ = self.mask.reshape(-1)
        # attn_ = attn.reshape(-1)
        # for i, value in enumerate(mask_):
        #     if value==1:
        #         attn_[i] = -1e9
        # attn = attn_.reshape(attn_size)
        # print(attn)
        batch_size, len_q = attn.size()
        
        
        ############################ Generate pad_mask ##########################################
        batch_size, len_q = attn.size()
        batch_size, len_k = attn.size()
        # eq(zero) is PAD token
        # seq_k_f = seq_k.to(torch.float64)
        # pad_attn_mask_new = torch.where(seq_k_f.data.eq(0), -1e9, 0).unsqueeze(1)
        pad_attn_mask = attn.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
        mask = pad_attn_mask.expand(batch_size, len_q, len_k)
        # mask = get_attn_pad_mask(attn, attn)
        ##########################################################################################
        
        ############################ Generate pad_mask ##########################################
        # attn_shape = [attn.size(0), attn.size(1), attn.size(1)]
        # # subsequence_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # Upper triangular matrix
        # # subsequence_mask = torch.ones(attn_shape)
        # sub_mask = torch.zeros(attn_shape)
        sub_mask = get_attn_subsequence_mask(attn)
        ##########################################################################################
        
        dec_self_attn_mask = torch.gt((mask + sub_mask), 0)
        # mask_sub = torch.randn(mask.size())
        # mask = mask+self.mask_sub
        # mask = mask.gt(0)
        # mask = torch.gt(mask, 0)
        # attn = attn.unsqueeze(1).expand(batch_size, len_q, len_q)
        attn = attn.expand(batch_size, len_q, len_q)
        # print(attn)
        
        # mask = torch.ones(attn.size())
        # mask = mask.to(torch.bool)
        # print(mask)
        # attn = attn.masked_fill_(mask, -1e9)
        # print(dec_self_attn_mask)
        attn_mask_value = torch.where(dec_self_attn_mask, -1e9, 0.)
        
        attn = attn+attn_mask_value
        # attn[mask>0] = -1e9
        # print(attn)
        result = self.fc(attn)
        # print(self.fc.weight)
        # print(result)
        return result
class MyModel(nn.Module):
    def __init__(self, size):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(size, 10)
        self.ln = nn.LayerNorm(512)
            
    def forward(self, x):
        y = self.fc(x)
        y = MyModel1()(y)
        z = torch.tensor([1, 2, 3])
        aa = torch.tensor([1, 2])
        bb = torch.tensor([1, 2])
        return torch.stack((aa,bb), 0)
class MyModel1(nn.Module):
    def __init__(self):
        super(MyModel1, self).__init__()
        self.ln = nn.LayerNorm(512)
            
    def forward(self, x):
        y = torch.randint(1, 10, (6, 10))
        w = torch.randint(1, 10, (10, 6))
        context = torch.matmul(y, w)
        return context
# @mpc.run_multiprocess(world_size=3)
def test():
    TFCrypt.init()
    mask_sub = torch.tensor([[821, 0.2, 3], [1.2, 100, 0], [0, 3.2, 1]])
    model = NN()
    model = torch.load('testMask.pth')
    input = torch.tensor([[821, 0.2, 3], [1.2, 100, 0], [0, 3.2, 1]])
    # dummy_input = torch.tensor([[821, 0, 3], [1.2, 100, 0], [0, 3.2, 1]])
    dummy_input = torch.randn(3, 3)
    private_model = TFCrypt.nn.from_pytorch(model, input).encrypt()
    private_inputs = TFCrypt.cryptensor(input)
    print("-"*50)
    predict_cry = private_model(private_inputs)
    predict = model(input)
    print(predict_cry.get_plain_text())
    print(predict)   
    if np.testing.assert_allclose(to_numpy(predict), predict_cry.get_plain_text(), rtol=1e-03, atol=1e-05) is None:
        print("Matched!!")
@mpc.run_multiprocess(world_size=3)  
def test_inf():
    # TFCrypt.init()
    x_enc = TFCrypt.cryptensor([2.0, .0, 21.])
    y_enc = TFCrypt.cryptensor([4.0, 7.2, 1.2])

    a, b = -1e9, 3 
    a_enc = TFCrypt.cryptensor(a)
    b_enc = TFCrypt.cryptensor([42, 21, 53])
    z_enc = TFCrypt.where(x_enc < y_enc, a_enc, b_enc)
    print("z:", z_enc.get_plain_text())
    
if __name__ == '__main__':
    test_inf()
    z = torch.tensor([1, 2, 3])
    theta_z = comm.get().gather(z, 0)
    print(theta_z)
    a = TFCrypt.cryptensor([1, 2, 3])
    b = TFCrypt.cryptensor([4, 5, 6])
    c = torch.tensor([7, 8, 9])
    d = a * c
    print(d.get_plain_text())
    a = TFCrypt.cryptensor([1, 2, 3], precision=0)
    print(a)

    network = torch.load('myDemo.pth')
    dummy_input = torch.tensor([2, 3, 4])
    private_model = TFCrypt.nn.from_pytorch(network, dummy_input).encrypt()
    # private_input = TFCrypt.cryptensor([1])
    # input = torch.tensor([0, -2])
    # private_input = TFCrypt.mpc.mpc.MPCTensor.from_shares(input, precision=0)
    private_input = TFCrypt.cryptensor([1], precision=0)
    result = private_model(private_input)
    print("Successful!")
    model = MyModel(512)
    dummy_input = torch.randn(10, 512, device="cpu")
    input_names = ["x"]
    output_names = ["y"]

    private_model = from_pytorch(model, dummy_input)

    opset_version 选择范围：[7,15]
    torch.onnx.export(
        model,
        dummy_input,
        "my_model.onnx",
        input_names=input_names,
        output_names=output_names,
        opset_version=10
    )

    num_examples=100
    source,target=preprocess(num_examples)
    enc_inputs, dec_inputs, dec_outputs, src_vocab_size, tgt_vocab_size, src_vocab, tgt_vocab = make_data(20, source, target)
    # 加载预训练模型
    model = Transformer(src_vocab_size, tgt_vocab_size).to(device)
    model.load_state_dict(torch.load("application/saved/model-0.3051304519176483.pt"))
    model.eval()

    dummy_enc_input = torch.randint(0, 10,(1,20)).to(device)
    dummy_dec_input = torch.randint(0, 10,(1,4)).to(device)


    # opset_version 选择范围：[7,15]
    torch.onnx.export(model, (dummy_enc_input, dummy_dec_input) ,"application/transformer.onnx",
        input_names=['enc_input','dec_input'],
        output_names=['output'],
        opset_version=9
    )
    mask_sub = torch.randn(3, 3)
    mask = torch.tensor([[1, 1, 0], [0, 1, 0], [0, 1, 1]])
    network = NN()
    inputs = torch.tensor([[0, 0.2, 3], [1.2, 100, 0], [0, 3.2, 1]])
    label = torch.tensor([[1, 2, 0, 2], [1, 0, 1, 0], [0, 2, 1, 1]])
    train(10, inputs, label, network, 'cpu', 0.001, 0.9)

    # Test
    test()
    
    # print("successfully!")
    
    
    
    
    # TFCrypt.init()
    # xx = TFCrypt.cryptensor([1, 2, 4, 5])
    # yy = TFCrypt.cryptensor([3, 2, 1, 6])
    # result = xx >= yy
    # print(result.get_plain_text())
    # print(np.inf)
    # model = onnx.load('/root/PPTF/application/transformer.onnx')
    # print(model.graph.node)