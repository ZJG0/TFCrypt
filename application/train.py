'''
Author: ZJG
Date: 2022-08-31 18:41:56
LastEditors: ZJG
LastEditTime: 2022-11-16 17:07:14
'''
from model_debug import*
from preprocess import*
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    num_examples=1000
    max_len = 32
    batch_size = 64
    source,target=preprocess(num_examples)
    src_w2i, src_i2w, w2i, i2w = get_static_vocab()
    enc_i, dec_i, dec_o = make_data(max_len, source, target)
    
    # enc_inputs, dec_inputs, dec_outputs, src_vocab_size, tgt_vocab_size, src_vocab, tgt_vocab = make_data(20, source, target)
    loader = Data.DataLoader(MyDataSet(enc_i, dec_i, dec_o), batch_size)
    
    model = Transformer(len(src_w2i), len(w2i)).to('cpu')
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.99)
    
    print("处理数据数量：",len(source)-1)
    
    tic = time.time()
    for epoch in range(1000):
        l_sum, num_sum = 0.0, 0.0
        for enc_inputs, dec_inputs, dec_outputs in loader:
            '''
            enc_inputs: [batch_size, src_len]
            dec_inputs: [batch_size, tgt_len]
            dec_outputs: [batch_size, tgt_len]
            '''
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            # outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            outputs = model(enc_inputs, dec_inputs)
            loss = criterion(outputs, dec_outputs.view(-1))
            l_sum += loss
            num_sum+=1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch {0:4d}, loss= {1:.3f}, time {2:.1f} sec".format(epoch, (l_sum/num_sum), time.time()-tic))
        if(epoch==10):
            torch.save(model.state_dict(), 'application/saved/model-{0}.pt'.format(l_sum/num_sum))
            sys.exit(0)
        tic = time.time()


if __name__ == '__main__':
    train()