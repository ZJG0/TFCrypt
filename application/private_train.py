'''
Author: ZJG
Date: 2022-08-31 18:41:56
LastEditors: ZJG
LastEditTime: 2022-10-13 17:08:11
'''
import TFCrypt
import torch.nn as nn
import torch
from model import*
from preprocess import*
import TFCrypt.communicator as comm
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_mpc_model(local_model: torch.nn.Module, src_vocab_size, tgt_vocab_size, device):
    rank = comm.get().get_rank()
    dummy_enc_input = torch.randint(0, 10, (101, 20))
    dummy_dec_input = torch.randint(0, 10, (101, 20))
    if rank == 0:
        model_upd = local_model
    else:
        model_upd = Transformer(src_vocab_size, tgt_vocab_size).to(device)
    model_upd.eval()
    private_model = TFCrypt.nn.from_pytorch(model_upd, (dummy_enc_input, dummy_dec_input)).encrypt(src=0)
    return private_model
    
def train_mpc(dataloader: DataLoader, model: TFCrypt.nn.Module, loss: TFCrypt.nn.Module, lr: float, tgt_vocab_size):
    total_loss = None
    count = len(dataloader)

    model.train()
    for enc_inputs, dec_inputs, dec_outputs in dataloader:
        
        private_enc_inputs = TFCrypt.cryptensor(enc_inputs, precision=0)
        private_dec_inputs = TFCrypt.cryptensor(dec_inputs, precision=0)
        dec_outputs = F.one_hot(dec_outputs, num_classes=tgt_vocab_size) 
        dec_outputs = dec_outputs.view(-1, tgt_vocab_size)
        private_dec_outputs = TFCrypt.cryptensor(dec_outputs, precision=0)

        outputs = model(private_enc_inputs, private_dec_inputs)
        print(outputs.size())
        print(private_dec_outputs.view(-1).size())
        loss_val = loss(outputs, private_dec_outputs)

        model.zero_grad()
        loss_val.backward()
        model.update_parameters(lr)

        if total_loss is None:
            total_loss = loss_val.detach()
        else:
            total_loss += loss_val.detach()

    total_loss = total_loss.get_plain_text().item()
    return total_loss / count

# def validate_mpc(dataloader: DataLoader, model: TFCrypt.nn.Module, loss: TFCrypt.nn.Module):
#     model.eval()
#     outs = []
#     true_ys = []
#     total_loss = None
#     count = len(dataloader)
#     for enc_inputs, dec_inputs, dec_outputs in tqdm(dataloader, file=sys.stdout):
#         private_enc_inputs = TFCrypt.cryptensor(enc_inputs, precision=0)
#         private_dec_inputs = TFCrypt.cryptensor(dec_inputs, precision=0)
#         private_dec_outputs = TFCrypt.cryptensor(dec_outputs, precision=0)

#         outputs = model(private_enc_inputs, private_dec_inputs)
#         loss_val = loss(outputs, private_dec_outputs.view(-1))

#         outs.append(outputs)
#         true_ys.append(ys)

#         if total_loss is None:
#             total_loss = loss_val.detach()
#         else:
#             total_loss += loss_val.detach()

#     total_loss = total_loss.get_plain_text().item()

#     all_out = TFCrypt.cat(outs, dim=0)
#     all_prob = all_out.sigmoid()
#     all_prob = all_prob.get_plain_text()
#     pred_ys = torch.where(all_prob > 0.5, 1, 0).tolist()
#     pred_probs = all_prob.tolist()

#     true_ys = TFCrypt.cat(true_ys, dim=0)
#     true_ys = true_ys.get_plain_text().tolist()

#     return total_loss / count, precision_score(true_ys, pred_ys), recall_score(true_ys, pred_ys), \
#            roc_auc_score(true_ys, pred_probs)


# def train():
#     num_examples=100
#     source,target=preprocess(num_examples)
    
#     enc_inputs, dec_inputs, dec_outputs, src_vocab_size, tgt_vocab_size, src_vocab, tgt_vocab = make_data(20, source, target)
#     loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 128, True)
    
#     model = Transformer(src_vocab_size, tgt_vocab_size).to(device)
#     criterion = nn.CrossEntropyLoss(ignore_index=0)
#     optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.99)
    
#     print("处理数据数量：",len(source)-1)
    
#     tic = time.time()
#     for epoch in range(1000):
#         l_sum, num_sum = 0.0, 0.0
#         for enc_inputs, dec_inputs, dec_outputs in loader:
#             '''
#             enc_inputs: [batch_size, src_len]
#             dec_inputs: [batch_size, tgt_len]
#             dec_outputs: [batch_size, tgt_len]
#             '''
#             enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
#             # outputs: [batch_size * tgt_len, tgt_vocab_size]
#             # outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
#             outputs = model(enc_inputs, dec_inputs)
#             loss = criterion(outputs, dec_outputs.view(-1))
#             l_sum += loss
#             num_sum+=1
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#         print("epoch {0:4d}, loss= {1:.3f}, time {2:.1f} sec".format(epoch, (l_sum/num_sum), time.time()-tic))
#         if(epoch==50):
#             torch.save(model.state_dict(), 'application/saved/model-{0}.pt'.format(l_sum/num_sum))
#             sys.exit(0)
#         tic = time.time()

def main():
    epochs = 50
    batch_size = 32
    lr = 1e-3
    eval_every = 1

    TFCrypt.init()
    num_examples=100
    source,target=preprocess(num_examples)
    
    enc_inputs, dec_inputs, dec_outputs, src_vocab_size, tgt_vocab_size, src_vocab, tgt_vocab = make_data(20, source, target)
    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 16)

    model = Transformer(src_vocab_size, tgt_vocab_size).to(device)
    mpc_model = make_mpc_model(model, src_vocab_size, tgt_vocab_size, device)
    mpc_loss = TFCrypt.nn.CrossEntropyLoss()

    for epoch in tqdm(range(epochs)):
        train_loss = train_mpc(loader, mpc_model, mpc_loss, lr, tgt_vocab_size)
        print(f"epoch: {epoch}, train loss: {train_loss}")

        # if epoch % eval_every == 0:
        #     validate_loss, p, r, auc = validate_mpc(test_dataloader, mpc_model, mpc_loss)
        #     print(f"epoch: {epoch}, validate loss: {validate_loss}, precision: {p}, recall: {r}, auc: {auc}")

if __name__ == '__main__':
    main()