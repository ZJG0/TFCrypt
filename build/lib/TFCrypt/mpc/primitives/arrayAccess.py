'''
Author: ZJG
Date: 2022-07-05 16:00:32
LastEditors: ZJG0 1173939915@qq.com
LastEditTime: 2023-07-28 10:58:31
'''
from scipy.fftpack import ss_diff
import TFCrypt
import TFCrypt.communicator as comm
import torch
import random
from TFCrypt.common.util import count_wraps
from TFCrypt.config import cfg
from .arithmetic import ArithmeticSharedTensor



def rightShift(A,a):
    # return A
    # arr = A.detach().numpy().tolist()
    arr = [s for s in range(A.shape[0])]
    for i in range(a):
        arr.insert(0,arr.pop())
    # return torch.tensor(arr)
    return arr
# def reverse(arr,start,end):
#     while start<end:
#         temp = arr[start]
#         arr[start] = arr[end]
#         arr[end] = temp
#         start += 1
#         end -= 1
 
# def rightShift(arr,k):
#     if not isinstance(arr, list):
#         is_list = False
#         arr = arr.detach().numpy().tolist()
#     if arr == None:
#         print("Paramter Invalid!")
#         return
#     lens = len(arr)
#     # print(arr)
#     k %= lens
#     reverse(arr,0,lens-k-1)
#     reverse(arr,lens-k,lens-1)
#     reverse(arr,0,lens-1)
#     # print(arr)
#     if not is_list:
#         return torch.tensor(arr)
#     else:
#         return arr

def where(condition, input, other):
    """
    Return a tensor of elements selected from either `input` or `other`, depending
    on `condition`.
    """
    # if TFCrypt.is_encrypted_tensor(condition):
    #     return condition * input + (1 - condition) * other
    # elif torch.is_tensor(condition):
    #     condition = condition.float()
    # if not torch.is_tensor(other):
    #     condition = TFCrypt.cryptensor(condition)
    #     input = TFCrypt.cryptensor(input)
    # # print(type(condition))
    # # print(type(input))
    # # print(type(other))
    # return input * condition + other * (1 - condition)
    # print('-----------------------------------------------')
    # if TFCrypt.is_encrypted_tensor(other):
    #     print(other.get_plain_text())
    if TFCrypt.is_encrypted_tensor(condition):
        return condition * input + (1-condition) * other
    elif torch.is_tensor(condition):
        # print(condition)
        condition = condition.float()
    return input * condition + other * (1-condition)


def SecureArrayAccess(array, index):
    if comm.get().get_world_size() != 3:
        raise NotImplementedError(
            "SecureArrayAccess is only implemented for world_size == 3."
        )

    rank = comm.get().get_rank()

    c = ArithmeticSharedTensor.PRZS(array.size(), device=array.device)

    array_size = array.size()
    m = array.size()[0]
    n = array.size()[1]

    '''
    Communication Pairs Table:
    r1: 0-->1       0
    r3: 0-->2       1
    array: 0-->1    2
    index: 0-->1    3
    array: 1-->2    4
    h: 1-->2        5
    '''

    if rank == 0:
        # generate r1 and r3
        r1 = random.randint(0, 10000)
        r3 = random.randint(0, 10000)
        r1_tensor = torch.tensor([r1])
        r3_tensor = torch.tensor([r3])
        ##################### Debug #########################
        # print('array:')
        # print(array)
        #####################################################
        #* calculate <a'>_1
        iii = rightShift(array, r1%m)
        array = array[iii]
        ##################### Debug #########################
        # print('array shifted r1 result:')
        # print(array)
        # print('r1:')
        # print(r1%m)
        #####################################################
        array = array.long() + c.share.long()
        ##################### Debug #########################
        # print('c:')
        # print(c.share)
        # print('array added result:')
        # print(array)
        # print('r3:')
        # print(r3%m)
        #####################################################
        #* calculate <a''>_1
        iii = rightShift(array, r3%m)
        array = array[iii]
        ##################### Debug #########################
        # print('array shifted r3 result:')
        # print(array)
        #####################################################
        #* calculate <h>_1
        index = (index+r1+r3)%m
        index_tensor = torch.tensor([index]).long()
        ##################### Debug #########################
        # print('<h>_1:')
        # print(index_tensor)
        #####################################################

        #* send data
        comm.get().send(r1_tensor, 1, 0)
        comm.get().send(r3_tensor, 2, 1)
        a = array.clone()
        comm.get().send(a, 1, 2)
        comm.get().send(index_tensor, 1, 3)
        
        # return result's share
        return torch.zeros(n).long()

    elif rank == 1:
        ##################### Debug #########################
        # print('array:')
        # print(array)
        #####################################################
        #* recv <a>_2
        array_2 = torch.zeros(array_size).long()
        array_2 = comm.get().recv(array_2, 2, 6)
        ##################### Debug #########################
        # print('array_2 from P3:')
        # print(array_2)
        #####################################################
        #* recv I_2
        index_2 = torch.zeros(1).long()
        index_2 = comm.get().recv(index_2, 2, 7)
        ##################### Debug #########################
        # print('I_2 from P3:')
        # print(index_2)
        #####################################################
        #* recv r1
        r1_tensor = torch.zeros(1).long()
        r1_tensor = comm.get().recv(r1_tensor, 0, 0)
        ##################### Debug #########################
        # print('r1 from P1:')
        # print(r1_tensor)
        #####################################################
        #* recv <a''>_2
        a = torch.zeros(size=array_size, dtype=torch.int64).long()
        a = comm.get().recv(a, 0, 2)
        ##################### Debug #########################
        # print('<a..>_2 from P1:')
        # print(a)
        #####################################################
        #* recv <h>_1
        index_tensor = torch.zeros(1).long()
        h1_tensor = comm.get().recv(index_tensor, 0, 3)
        ##################### Debug #########################
        # print('<h>_1 from P1:')
        # print(h1_tensor)
        #####################################################

        r1 = r1_tensor[0].data
        h1 = h1_tensor[0].data
        index_2 = index_2[0].data
        #* calculate <h>_1 + I_2
        h = (index +index_2 + h1)%m
        ##################### Debug #########################
        # print('h:')
        # print(h)
        #####################################################
        #* calculate <a'>_2
        array_1 = array + array_2
        ##################### Debug #########################
        # print('array1 + array2:')
        # print(array_1)
        #####################################################
        iii = rightShift(array_1, r1%m)
        array_1 = array_1[iii]
        
        array_1 = array_1.long() + c.share.long()
        ##################### Debug #########################
        # print('<a.>_2:')
        # print(array_1)
        #####################################################
        #* send data
        h_tensor = torch.tensor([h]).long()
        comm.get().send(array_1, 2, 4)
        comm.get().send(h_tensor, 2, 5)
        return a[h]

    elif rank == 2:
        ##################### Debug #########################
        # print('array:')
        # print(array)
        #####################################################
        #* send <a>_3
        array_2 = array.clone()
        comm.get().send(array_2, 1, 6)
        #* send I_3
        index_2 = torch.tensor([index]).long()
        comm.get().send(index_2, 1, 7)
        #* recv r3
        r3_tensor = torch.zeros(1).long()
        r3_tensor = comm.get().recv(r3_tensor, 0, 1)
        ##################### Debug #########################
        # print('r3 from P1')
        # print(r3_tensor)
        #####################################################
        #* recv <a'>_2
        array_1 = torch.zeros(array_size).long()
        array_1 = comm.get().recv(array_1, 1, 4)
        ##################### Debug #########################
        # print('<a.>_2 from P2')
        # print(array_1)
        #####################################################
        #* recv h
        h_tensor = torch.zeros(1).long()
        h_tensor = comm.get().recv(h_tensor, 1, 5)
        ##################### Debug #########################
        # print('h from P2')
        # print(h_tensor)
        #####################################################

        h = h_tensor[0].data
        r3 = r3_tensor[0].data

        a = array_1.long() + c.share.long()
        iii = rightShift(a, r3%m)
        a = a[iii]
        ##################### Debug #########################
        # print('<a..>_3')
        # print(a)
        #####################################################
        return a[h]