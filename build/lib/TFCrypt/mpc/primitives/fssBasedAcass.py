'''
Author: ZJG0 1173939915@qq.com
Date: 2023-07-26 17:47:07
LastEditors: ZJG0 1173939915@qq.com
LastEditTime: 2023-08-04 08:43:47
FilePath: /PPTF/crypten/mpc/primitives/fssBasedAccess.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from scipy.fftpack import ss_diff
import TFCrypt
import TFCrypt.communicator as comm
import torch
import random
from TFCrypt.common.util import count_wraps
from TFCrypt.config import cfg
from .arithmetic import ArithmeticSharedTensor
from .fss import DCF, GroupElement
RING_LEN = 16
seed=135124324
def rightShift(A,a):
    # return A
    # arr = A.detach().numpy().tolist()
    # print(A.shape[0])
    arr = [s for s in range(A.shape[0])]
    # print(len(arr))
    for i in range(a):
        arr.insert(0,arr.pop())
    # return torch.tensor(arr)
    
    return arr

def preprocess(m):
    rank = comm.get().get_rank()
    if rank == 0:
        ri_0 = random.randint(0, 1)
        
        dcf = DCF(ring_len = RING_LEN)

        a = GroupElement(ri_0,RING_LEN)
        b = GroupElement(1,RING_LEN)

        k0, _ = dcf.keyGen(seed,a,b)
        x= GroupElement(ri_0,RING_LEN)
        # x.selfPrint()
        t0 = dcf.eval(0,x,k0, m)
        return ri_0, t0, t0, t0
    if rank == 1:
        ri_1 = random.randint(0, 1)
        
        dcf = DCF(ring_len = RING_LEN)

        a = GroupElement(ri_1,RING_LEN)
        b = GroupElement(1,RING_LEN)
        x= GroupElement(ri_1,RING_LEN)
        _, k1 = dcf.keyGen(seed,a,b)
        t1 = dcf.eval(1,x,k1, m)
        return ri_1, t1, t1, t1
    if rank == 2:
        ri_2 = random.randint(0, 1)
        
        dcf = DCF(ring_len = RING_LEN)

        a = GroupElement(ri_2,RING_LEN)
        b = GroupElement(1,RING_LEN)
        x= GroupElement(ri_2,RING_LEN)
        _, k2 = dcf.keyGen(seed,a,b)
        t1 = dcf.eval(1,x,k2, m)
        return t1, t1
 
def preprocess2(m):
    t2 = torch.zeros(m)
    t2[3] = 1
    t3 = torch.zeros(m)
    t3[3] = 1
    return  t2, t3
# def convert(t_xor):
    
#     return t_ari
def fssBasedAccess(array, index):
    if comm.get().get_world_size() != 3:
        raise NotImplementedError(
            "fssBasedAccess is only implemented for world_size == 3."
        )

    rank = comm.get().get_rank()

    c = ArithmeticSharedTensor.PRZS(array.size(), device=array.device)

    array_size = array.size()
    m = array.size()[0]
    n = array.size()[1]
    if rank == 0:
        ri, t1, t2, t3 = preprocess(m)
        
        zeta_0 = torch.randn(m, n).long()
        arrayblind_0 = array + zeta_0
        arrayblind_0 = arrayblind_0.long()
        
        # comm.get().send(arrayblind_0, 1, 1)
        comm.get().send(zeta_0, 2, 1)
        
        
        
        # arrayblind_1 = torch.zeros([m, n]).long()
        # arrayblind_1 = comm.get().recv(arrayblind_1, 1, 6)
        arrayblind = array #+ arrayblind_1
        
        indexBlind_0 = index - ri
        indexBlind_0 = torch.tensor([indexBlind_0]).long()
        
        comm.get().send(indexBlind_0, 1, 3)
        comm.get().send(indexBlind_0, 2, 4)
        
        
        indexBlind_1 = torch.zeros(1).long()
        indexBlind_1 = comm.get().recv(indexBlind_1, 1, 7)
        indexBlind_1 = indexBlind_1[0].data
        indexBlind = indexBlind_0 + indexBlind_1
        
        rotation = int((indexBlind%m)[0].data)
        
        iii = rightShift(t1, rotation)
        newt1 = t1[iii]
        # iii = rightShift(t3, rotation)
        newt3 = t3[iii]
        
        gamma_0 = torch.zeros(n).long()
        gamma_0 = comm.get().recv(gamma_0, 2, 9)
        gamma_0 = gamma_0[0].data
        embed_1 = torch.sum(torch.mul(arrayblind, newt1.reshape(m, 1))) - torch.sum(torch.mul(zeta_0, (newt3-newt1).reshape(m, 1))) + gamma_0
        
        
        return torch.zeros(n).long()
    
    elif rank == 1:
        ri, t1, t2, t3 = preprocess(m)
        
        zeta_1 = torch.randn(m, n).long()
        
        
        # arrayblind_0 = torch.zeros([m, n]).long()
        # arrayblind_0 = comm.get().recv(arrayblind_0, 0, 1)
        
        arrayblind_1 = array + zeta_1
        arrayblind_1 = arrayblind_1.long()
        # comm.get().send(arrayblind_1, 0, 6)
        comm.get().send(zeta_1, 2, 5)
        
        
        
        arrayblind = array #+ arrayblind_0
        
        indexBlind_0 = torch.zeros(1).long()
        indexBlind_0 = comm.get().recv(indexBlind_0, 0, 3)
        indexBlind_0 = indexBlind_0[0].data
        
        indexBlind_1 = index - ri
        indexBlind_1 = torch.tensor([indexBlind_1]).long()
        
        
        indexBlind = indexBlind_0 + indexBlind_1
        # print('hjdhashdjahsldhalskh')
        rotation = int((indexBlind%m)[0].data)
        # print(t1.size())
        # print(rotation)
        iii = rightShift(t1, rotation)
        newt1 = t1[iii]
        # iii = rightShift(t2, rotation)
        newt2 = t2[iii]
        
        comm.get().send(indexBlind_1, 0, 7)
        comm.get().send(indexBlind_1, 2, 8)
        
        # print('hjkdhalhdklalkdkskl')
        gamma_1 = torch.zeros(n).long()
        gamma_1 = comm.get().recv(gamma_1, 2, 10)
        gamma_1 = gamma_1[0].data
        
        embed_1 = torch.sum(torch.mul(arrayblind, newt1.reshape(m, 1))) - torch.sum(torch.mul(zeta_1, (newt2-newt1).reshape(m, 1))) + gamma_1
        # print("--------------------------------------------------------test----------------------------------------------------------")
        
        
        return torch.zeros(n).long()
        
        
    elif rank == 2:
        t2, t3 = preprocess(m)
        
        zeta_0 = torch.zeros([m, n]).long()
        zeta_0 = comm.get().recv(zeta_0, 0, 1)
        
        zeta_1 = torch.zeros([m, n]).long()
        zeta_1 = comm.get().recv(zeta_1, 1, 5)
        
        # zeta_0 = torch.randn(m, n).long()
        # zeta_1 = torch.randn(m, n).long()
        
        indexBlind_0 = torch.zeros(1).long()
        indexBlind_0 = comm.get().recv(indexBlind_0, 0, 4)
        indexBlind_0 = indexBlind_0[0].data
        
        indexBlind_1 = torch.zeros(1).long()
        indexBlind_1 = comm.get().recv(indexBlind_1, 1, 8)
        indexBlind_1 = indexBlind_1[0].data
        indexBlind = indexBlind_0 + indexBlind_1
        
        rou = torch.randn(n).long()
        
        rotation = int(indexBlind%m)
        
        iii = rightShift(t2, rotation)
        newt2 = t2[iii]
        # iii = rightShift(t3, rotation)
        newt3 = t3[iii]
        
        gamma_0 = (-1)*torch.sum(torch.mul(zeta_0, newt3.reshape(m, 1))) + rou
        gamma_1 = (-1)*torch.sum(torch.mul(zeta_1, newt2.reshape(m, 1))) - rou
        gamma_0 = gamma_0.long()
        gamma_1 = gamma_1.long()
        
        comm.get().send(gamma_0, 0, 9)
        comm.get().send(gamma_1, 1, 10)
        return torch.zeros(n).long()