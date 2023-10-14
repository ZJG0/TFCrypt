'''
Author: ZJG
Date: 2022-08-31 14:41:08
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-10-14 10:29:00
'''
import TFCrypt
from application.model_debug import*
from application.preprocess import*
import TFCrypt.communicator as comm
import onnxruntime
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def greedy_decoder(model, enc_input, start_symbol):
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    terminal = False
    next_symbol = start_symbol
    while not terminal:   
        dec_input = torch.cat([dec_input.detach(),torch.tensor([[next_symbol]],dtype=enc_input.dtype).to(device)],-1)
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[-1]
        next_symbol = next_word
        if next_symbol == tgt_vocab.eos:
            terminal = True
        #print(next_word)            
    return dec_input
    
def construct_private_model(model, src_vocab_size, tgt_vocab_size, device, enc_input_size, dec_input_size):
    """Encrypt and validate trained model for multi-party setting."""
    # get rank of current process
    rank = comm.get().get_rank()
    dummy_enc_input = torch.randint(0, 10, (1, enc_input_size[0]))
    dummy_dec_input = torch.randint(0, 10, dec_input_size)
    # party 0 always gets the actual model; remaining parties get dummy model
    if rank == 0:
        model_upd = model
    else:
        model_upd = Transformer(src_vocab_size, tgt_vocab_size).to(device)
    model_upd.eval()
    private_model = TFCrypt.nn.from_pytorch(model_upd, (dummy_enc_input, dummy_dec_input)).encrypt(src=0)
    return private_model

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    

if __name__ == '__main__':
    TFCrypt.init()
    num_examples=100
    source,target=preprocess(num_examples)
    path = "/root/PPTF/application/saved/model-withMasked.pt"

    enc_inputs, dec_inputs, dec_outputs, src_vocab_size, tgt_vocab_size, src_vocab, tgt_vocab = make_data(20, source, target)
    # print(src_vocab_size)
    # print(tgt_vocab_size)

    src_w2i = src_vocab.token_to_idx
    src_i2w = {i: w for i, w in enumerate(src_vocab.idx_to_token)}

    i2w = {i: w for i, w in enumerate(tgt_vocab.idx_to_token)}


    # 加载预训练模型
    model = Transformer(src_vocab_size, tgt_vocab_size).to(device)

    save_model = torch.load(path)
    model_dict = model.state_dict()
    state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
    # print(state_dict.keys()) # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    # model.load_state_dict(torch.load("application/saved/model-withMasked.pt"))
    model.eval()
    # model = model.encoder
    
    # outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
    # 加载测试数据
    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 100)

    # Test
    enc_inputs, _, _ = next(iter(loader))
    predict_total = []
    predict_onnx_total = []

    # greedy_dec_input = greedy_decoder(model, enc_inputs[0].view(1, -1), start_symbol=tgt_vocab.bos)
    
    # private_model = construct_private_model(model, src_vocab_size, tgt_vocab_size, device, enc_inputs[0].size(), greedy_dec_input.size())
    # ort_session = onnxruntime.InferenceSession("/root/PPTF/application/transformer.onnx")

    # predict = model(enc_inputs[0].view(1, -1), greedy_dec_input)

    # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(enc_inputs[0].view(1, -1)), ort_session.get_inputs()[1].name: to_numpy(greedy_dec_input)}
    # ort_outs = ort_session.run(None, ort_inputs)
    # # predict_onnx = torch.tensor(predict_onnx)
    # # predict_onnx = predict_onnx.view(-1, predict_onnx.size(-1))
    # # compare accuracy
    # print('################################Result:')
    # print(to_numpy(predict))
    # print('################################Onnx:')
    # print(ort_outs[0])
    # np.testing.assert_allclose(to_numpy(predict), ort_outs[0], rtol=1e-03, atol=1e-05)
           








    for i in tqdm(range(len(enc_inputs))):
    # for i in tqdm(range(70, 85)):
        # print([src_i2w[n.item()] for n in enc_inputs[i].numpy()])
        #print( '->', enc_inputs[i]) #输出 enc_input 的 tensor 

        greedy_dec_input = greedy_decoder(model, enc_inputs[i].view(1, -1), start_symbol=tgt_vocab.bos)
        # private_model = construct_private_model(model, src_vocab_size, tgt_vocab_size, device, enc_inputs[i].size(), greedy_dec_input.size())
        ort_session = onnxruntime.InferenceSession("/root/PPTF/application/transformer.onnx")
        # print(enc_inputs[i].view(1, -1).size())
        # print(greedy_dec_input.size())
        # print(enc_inputs[i])
        # greedy_dec_input = torch.tensor([[ 1,  9, 3, 3]])
        # enc_inputs[i] = torch.tensor([ 18,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0])

        enc_private_input = TFCrypt.cryptensor(enc_inputs[i].view(1, -1), precision=0)
        dec_private_input = TFCrypt.cryptensor(greedy_dec_input, precision=0)
        print(enc_inputs[i])
        print("Inference")
        # predict1 = private_model(enc_private_input, dec_private_input)
        # predict1 = predict1.get_plain_text()
        # predict = predict.data.max(1, keepdim=True)[1]
        print("ONNX Inference")
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(enc_inputs[i].view(1, -1)), ort_session.get_inputs()[1].name: to_numpy(greedy_dec_input)}
        predict_onnx = ort_session.run(None, ort_inputs)


        predict = model(enc_inputs[0].view(1, -1), greedy_dec_input)

        print(predict)
        # print(greedy_dec_input)
        # print(predict1)
        # np.testing.assert_allclose(to_numpy(predict1), predict_onnx[0], rtol=1e-03, atol=1e-05)


        # predict_onnx = predict_onnx.data.max(1, keepdim=True)[1]
        # print("Inference:")
        # if predict.shape[0] != 1:
        #     #print(dec_outputs[i]) #输出 dec_output 的 tensor
        #     print([src_i2w[n.item()] for n in enc_inputs[i].numpy()])
        #     print('->', [i2w[n.item()] for n in predict.squeeze()])
        #     print()
            
        # print("ONNX Inference:")
        # if predict_onnx.shape[0] != 1:
        #     #print(dec_outputs[i]) #输出 dec_output 的 tensor
        #     print([src_i2w[n.item()] for n in enc_inputs[i].numpy()])
        #     print('->', [i2w[n.item()] for n in predict_onnx.squeeze()])
        #     print()

        # predict_total.append([i2w[n.item()] for n in predict.squeeze()])
        # predict_onnx_total.append([i2w[n.item()] for n in predict_onnx.squeeze()])
    # predict_total = np.array(predict_total).reshape(-1).tolist()
    # predict_onnx_total = np.array(predict_onnx_total).reshape(-1).tolist()
    # predict_size = len(predict_total)
    # sum_eq = 0
    # for i in range(predict_size):
    #     if predict_total[i]==predict_onnx_total[i]:
    #         sum_eq = sum_eq+1
    # print('Equal Percent: '+str(sum_eq/len(predict_total)*100)+'%')