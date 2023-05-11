import numpy as np
import torch
#冒号区分的是索引，逗号区分的是维度，。。。。用来代替全索引长度

a = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a)
print(f"a[:,1]", a[...,1:])
print(f"a[0,:]", a[:,0])
print(a[1,::])
print("torch.randn{}:".format(torch.randn(20,20)))

t=torch.tensor([[1,2,3],[4,5,6]])
t_new = t.transpose(0,1)
print("墨迹阿婆:{}".format(t_new.is_contiguous()))
print(f"墨迹阿婆:{(t_new.is_contiguous())}")