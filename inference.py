'''
Author: ZJG
Date: 2022-08-31 14:41:08
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-10-14 10:28:53
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
        if next_symbol == w2i['<eos>']:
            terminal = True
        #print(next_word)            
    return dec_input
    
def construct_private_model(model, src_vocab_size, tgt_vocab_size, device, enc_input_size, dec_input_size):
    """Encrypt and validate trained model for multi-party setting."""
    # get rank of current process
    # rank = comm.get().get_rank()
    dummy_enc_input = torch.randint(0, 10, (1, enc_input_size[0]))
    dummy_dec_input = torch.randint(0, 10, dec_input_size)
    # # party 0 always gets the actual model; remaining parties get dummy model
    # if rank == 0:
    #     model_upd = model
    # else:
    #     model_upd = Transformer(src_vocab_size, tgt_vocab_size).to(device)
    # model_upd.eval()
    private_model = TFCrypt.nn.from_pytorch(model, (dummy_enc_input, dummy_dec_input))
    return private_model

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    

if __name__ == '__main__':
    num_examples=100
    max_len = 32
    batch_size = 64
    path = "/root/PPTF/application/saved/model-3.4043922424316406.pt"
    
    # path = "/root/PPTF/application/saved/0.30198728933357466_model.pth"
    
    source, target=preprocess(num_examples)
    
    enc_i, dec_i, dec_o = make_data(max_len, source, target)
    # src_w2i = src_vocab.token_to_idx
    # src_i2w = {i: w for i, w in enumerate(src_vocab.idx_to_token)}

    # w2i = tgt_vocab.token_to_idx
    # i2w = {i: w for i, w in enumerate(tgt_vocab.idx_to_token)}
    src_w2i, src_i2w, w2i, i2w = get_static_vocab()


    # 加载预训练模型
    model = Transformer(len(src_w2i), len(w2i)).to('cpu')
    save_model = torch.load(path, map_location='cpu')
    model_dict = model.state_dict()
    state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    model.eval()
    
    # outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
    # 加载测试数据
    loader = Data.DataLoader(MyDataSet(enc_i, dec_i, dec_o), batch_size)

    # Test
    # for x in range(len(loader)):
    enc_inputs, _, dec_outputs = next(iter(loader))
    predict_total = []
    predict_onnx_total = []
    t = time.perf_counter()
    for i in range(len(enc_inputs)):
        # for i in tqdm(range(70, 85)):
        # print([src_i2w[n.item()] for n in enc_inputs[i].numpy()])
        #print( '->', enc_inputs[i]) #输出 enc_input 的 tensor 

        greedy_dec_input = greedy_decoder(model, enc_inputs[i].view(1, -1), start_symbol= w2i['<bos>'])
        # private_model = construct_private_model(model, len(src_w2i), len(w2i), device, enc_inputs[i].size(), greedy_dec_input.size())
        # ort_session = onnxruntime.InferenceSession("/root/PPTF/application/transformer.onnx")
        # print(enc_inputs[i].view(1, -1).size())
        # print(greedy_dec_input.size())
        # print(enc_inputs[i])
        # greedy_dec_input = torch.tensor([[ 1,  9, 3, 3]])
        # enc_inputs[i] = torch.tensor([ 18,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0])

        # enc_private_input = TFCrypt.cryptensor(enc_inputs[i].view(1, -1), precision=0)
        # dec_private_input = TFCrypt.cryptensor(greedy_dec_input, precision=0)

        predict = model(enc_inputs[i].view(1, -1), greedy_dec_input)
        predict = predict.data.max(1, keepdim=True)[1]

        # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(enc_inputs[i].view(1, -1)), ort_session.get_inputs()[1].name: to_numpy(greedy_dec_input)}
        # predict_onnx = ort_session.run(None, ort_inputs)
        # predict_onnx = torch.tensor(predict_onnx)
        # print(enc_inputs[i])
        print("predict:")
        print(predict)
        # print("onnx:")
        # print(predict_onnx)
        
    cost = time.perf_counter() - t
    print('Time Cost = {}'.format(cost))
        # predict_onnx = predict_onnx.view(-1, predict_onnx.size(-1))
        # predict_onnx = predict_onnx.data.max(1, keepdim=True)[1]
        # print("Inference:")
        # if predict.shape[0] != 1:
        #     #print(dec_outputs[i]) #输出 dec_output 的 tensor
        #     print([i2w[n.item()] for n in dec_outputs[i].numpy()])
        #     print('->', [i2w[n.item()] for n in predict.squeeze()])
        #     print()
            
    #     print("ONNX Inference:")
    #     if predict_onnx.shape[0] != 1:
    #         #print(dec_outputs[i]) #输出 dec_output 的 tensor
    #         print([src_i2w[n.item()] for n in enc_inputs[i].numpy()])
    #         print('->', [i2w[n.item()] for n in predict_onnx.squeeze()])
    #         print()

    #     predict_total.append([i2w[n.item()] for n in predict.squeeze()])
    #     predict_onnx_total.append([i2w[n.item()] for n in predict_onnx.squeeze()])
    # predict_total = np.array(predict_total).reshape(-1).tolist()
    # predict_onnx_total = np.array(predict_onnx_total).reshape(-1).tolist()
    # predict_size = len(predict_total)
    # sum_eq = 0
    # for i in range(predict_size):
    #     if predict_total[i]==predict_onnx_total[i]:
    #         sum_eq = sum_eq+1
    # print('Equal Percent: '+str(sum_eq/len(predict_total)*100)+'%')

        