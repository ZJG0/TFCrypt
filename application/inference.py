'''
Author: ZJG
Date: 2022-08-31 14:41:08
LastEditors: ZJG
LastEditTime: 2022-09-01 08:30:45
'''
import TFCrypt
from model import*
from preprocess import*
import TFCrypt.communicator as comm


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
    
def construct_private_model(model, src_vocab_size, tgt_vocab_size, device):
    """Encrypt and validate trained model for multi-party setting."""
    # get rank of current process
    rank = comm.get().get_rank()
    dummy_enc_input = torch.randint(0, 10,(1,20))
    dummy_dec_input = torch.randint(0, 10,(1,4))
    # party 0 always gets the actual model; remaining parties get dummy model
    if rank == 0:
        model_upd = model
    else:
        model_upd = Transformer(src_vocab_size, tgt_vocab_size).to(device)
    private_model = TFCrypt.nn.from_pytorch(model_upd, (dummy_enc_input, dummy_dec_input)).encrypt(src=0)
    return private_model
    

if __name__ == '__main__':
    TFCrypt.init()
    num_examples=100
    source,target=preprocess(num_examples)

    enc_inputs, dec_inputs, dec_outputs, src_vocab_size, tgt_vocab_size, src_vocab, tgt_vocab = make_data(20, source, target)
    # print(src_vocab_size)
    # print(tgt_vocab_size)

    src_w2i = src_vocab.token_to_idx
    src_i2w = {i: w for i, w in enumerate(src_vocab.idx_to_token)}

    w2i = tgt_vocab.token_to_idx
    i2w = {i: w for i, w in enumerate(tgt_vocab.idx_to_token)}


    # 加载预训练模型
    model = Transformer(src_vocab_size, tgt_vocab_size).to(device)
    model.load_state_dict(torch.load("application/saved/model-0.29741358757019043.pt"))
    model.eval()
    
    outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
    # 加载测试数据
    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 10, True)

    # Test
    enc_inputs, _, _ = next(iter(loader))
    for i in range(len(enc_inputs)):
        print([src_i2w[n.item()] for n in enc_inputs[i].numpy()])
        #print( '->', enc_inputs[i]) #输出 enc_input 的 tensor 
        private_model = construct_private_model(model, src_vocab_size, tgt_vocab_size, device)

        greedy_dec_input = greedy_decoder(model, enc_inputs[i].view(1, -1), start_symbol=tgt_vocab.bos)

        print(enc_inputs[i].view(1, -1).size())
        print(greedy_dec_input.size())

        predict, _, _, _ = model(enc_inputs[i].view(1, -1), greedy_dec_input)
        predict = predict.data.max(1, keepdim=True)[1]
        if predict.shape[0] != 1:
            #print(dec_outputs[i]) #输出 dec_output 的 tensor
            print('->', [i2w[n.item()] for n in predict.squeeze()])
            print()