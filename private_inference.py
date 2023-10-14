'''
Author: ZJG
Date: 2022-08-31 14:41:08
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-10-14 10:28:57
'''
import TFCrypt
from application.model_debug import*
from application.preprocess import*
import TFCrypt.communicator as comm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os
import psutil
# os.environ['GLOO_SOCKET_IFNAME'] = 'eth1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_network_bytes():
    # 获取所有网卡的网络流量统计信息
    net_io = psutil.net_io_counters(pernic=True)
    return net_io
def calculate_network_traffic(net_io_start, net_io_end):
    traffic = {}
    for interface, stats_start in net_io_start.items():
        stats_end = net_io_end.get(interface)
        if stats_end:
            bytes_sent = stats_end.bytes_sent - stats_start.bytes_sent
            bytes_received = stats_end.bytes_recv - stats_start.bytes_recv
            traffic[interface] = {
                'bytes_sent': bytes_sent,
                'bytes_received': bytes_received
            }
    return traffic
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
    rank = comm.get().get_rank()
    # dummy_enc_input = torch.randint(0, 10, (1, enc_input_size[0]))
    # dummy_dec_input = torch.randint(0, 10, dec_input_size)
    enc_input_size = enc_input_size.view(1,-1)
    # party 0 always gets the actual model; remaining parties get dummy model
    if rank == 0:
        model_upd = model
    else:
        model_upd = Transformer(len(src_w2i), len(w2i)).to(device)
    model_upd.eval()
    # private_model = TFCrypt.nn.from_pytorch(model_upd, (dummy_enc_input, dummy_dec_input)).encrypt(src=0)
    private_model = TFCrypt.nn.from_pytorch(model_upd, (enc_input_size, dec_input_size)).encrypt(src=0)
    return private_model
    

if __name__ == '__main__':
    TFCrypt.init()
    
    num_examples=3
    max_len = 32
    batch_size = 64
    
    # path = "/root/PPTF/application/saved/model-tiny.pt"  # tiny 
    path = "/root/PPTF/application/saved/model-medium.pt" # medium
    # path = "/root/PPTF/application/saved/0.30198728933357466_model.pth" # Base
    # path = "/root/PPTF/application/saved/model-large.pt"  # large
    
    source, target=preprocess(num_examples)
    
    enc_i, dec_i, dec_o = make_data(max_len, source, target)
    # src_w2i = src_vocab.token_to_idx
    # src_i2w = {i: w for i, w in enumerate(src_vocab.idx_to_token)}

    # w2i = tgt_vocab.token_to_idx
    # i2w = {i: w for i, w in enumerate(tgt_vocab.idx_to_token)}
    src_w2i, src_i2w, w2i, i2w = get_static_vocab()


    # 加载预训练模型
    model = Transformer(len(src_w2i), len(w2i)).to('cpu')
    
    
    # save_model = torch.load(path, map_location='cpu')
    # save_model = save_model['model_state_dict']
    # model_dict = model.state_dict()
    # state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
    # model_dict.update(state_dict)
    # model.load_state_dict(model_dict)

    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    
    # outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
    # 加载测试数据
    loader = Data.DataLoader(MyDataSet(enc_i, dec_i, dec_o), batch_size)

    # Test
    # enc_inputs, _, _ = next(iter(loader))
    smooth = SmoothingFunction()
    total_bleu=[]
    t = time.perf_counter()
    net_io_before = get_network_bytes()
    for enc_inputs, dec_inputs, dec_outputs in loader:
        for i in range(len(enc_inputs)):
            # print([src_i2w[n.item()] for n in enc_inputs[i].numpy()])

            greedy_dec_input = greedy_decoder(model, enc_inputs[i].view(1, -1), start_symbol= w2i['<bos>'])
            # private_model = construct_private_model(model, len(src_w2i), len(w2i), device, enc_inputs[i].size(), greedy_dec_input.size())
            private_model = construct_private_model(model, len(src_w2i), len(w2i), device, enc_inputs[i], greedy_dec_input)
            # print(enc_inputs[i])
            # sys.exit(0)
            # print("#####################################################################################")
            # greedy_dec_input = torch.tensor([[ 1,  9, 3, 3]])
            # enc_inputs[i] = torch.tensor([ 18,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0])

            enc_private_input = TFCrypt.cryptensor(enc_inputs[i].view(1, -1), precision=0)
            dec_private_input = TFCrypt.cryptensor(greedy_dec_input, precision=0)

            predict = private_model(enc_private_input, dec_private_input)
            predict = predict.get_plain_text()
            
            # print(predict)
        
            # predict = model(enc_inputs[i].view(1, -1), greedy_dec_input)
            predict = predict.data.max(1, keepdim=True)[1]
            
            if predict.shape[0] != 1:
                predict_list=sign_filter([i2w[n.item()] for n in predict.squeeze()])
                # print('->',predict_list)
                target_list=sign_filter([i2w[n.item()] for n in dec_outputs[i].numpy()])
                # print('target:',target_list)
                bleu_score = sentence_bleu([target_list], predict_list, smoothing_function=smooth.method1)
                # print("bleu得分：",bleu_score)
                print("The "+str(i)+" Inference Done!")
                total_bleu.append(bleu_score)
    cost = time.perf_counter() - t
    net_io_after = get_network_bytes()
    traffic_diff = calculate_network_traffic(net_io_before, net_io_after)
    # print('There are '+str(len(total_bleu))+' sentences.')
    bleu = sum(total_bleu) / len(total_bleu)
    
    # print('TOTAL BLEU SCORE = {}'.format(bleu))
    print('Time Cost = {}'.format(cost))
    print(traffic_diff)