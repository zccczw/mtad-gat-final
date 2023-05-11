import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.nn as nn
from STTNS.STTNs import STTNSNet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from STTNS.utils import get_adjacency_matrix, SlidingWindowDataset, create_data_loaders

"""
parameters:
A:邻接矩阵
in_channels:输入通道信息，只有速度信息，所以通道为1
embed_size:Transformer通道数
time_num:一天时间间隔数量
num_layers:堆叠层数
T_dim=12:输入时间维度。输入前一小时数据，所以60min/5min = 12
output_T_dim=3:输出时间维度。预测未来15,30,45min速度
heads = 1
"""

if __name__ == '__main__':
    """
    model = STTNS()
    criterion
    optimizer
    for i in rang(epochs):
        out, _ = model(args)


    """
    days = 62  # 用10天的数据进行训练
    val_day = 3  # 3天验证

    train_num = 288 * days  # 间隔5min一个数据， 一天288个5min
    val_num = 288 * days
    row_num = train_num + val_num

    # dataset
    v = np.load("PEMS08/PEMS08.npz")['data']
    adj = get_adjacency_matrix('PEMS08/PEMS08.csv', v.shape[1])  # 邻接矩阵
    # print(v.shape) : [T, N]

    adj = np.array(adj)
    adj = torch.tensor(adj, dtype=torch.float32).cuda()

    v = np.array(v)[:, :, 0]
    v = torch.tensor(v, dtype=torch.float32)
    # 最终 v shape:[N, T]。  N=25, T=row_num
    # print(v.shape)

    node = v.shape[1]
    k = 5
    in_channels = 1
    embed_size = 128
    time_num = 288
    num_layers = 1
    T_dim = 12  # 12*5 = 60, 输入前一个小时数据
    output_T_dim = 3  # 预测后15min数据
    heads = 1
    epochs = 50
    dropout = 0.2
    forward_expansion = 4
    bs = 16
    val_split = 0.2
    shuffle_dataset = True
    init_lr = 0.001

    train_dataset = SlidingWindowDataset(v, T_dim, horizon=output_T_dim)
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, bs, val_split, shuffle_dataset)

    model = STTNSNet(node, k, in_channels, embed_size, time_num, num_layers,
                     T_dim, output_T_dim, heads, dropout, forward_expansion)
    model.cuda()

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params += param
    print(f"Total Trainable Params: {total_params}")
    """
    for i in range(epochs):
        pass

    """
    #   ----训练部分----
    # t表示遍历到的具体时间
    pltx = []
    plty = []
    print(f"Training model for {epochs} epochs..")
    train_start = time.time()
    t = 0
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        forecast_b_losses = []
        recon_b_losses = []

        for x, y in train_loader:
            x = x.cuda()
            x = x.unsqueeze(1).permute(0, 1, 3, 2)
            y = y.cuda().permute(0, 2, 1)
            optimizer.zero_grad()
            # x shape:[bs, 1, N, T_dim]
            # y shape:[bs, N, output_T_dim]

            out = model(x, t)
            loss = criterion(out, y)

            if t % 100 == 0:
                print("MAE loss:", loss)

            # 常规操作
            loss.backward()
            optimizer.step()

            pltx.append(t)
            plty.append(loss.detach().cpu().numpy())
            t += 1

    plt.plot(pltx, plty, label="STTN train")
    plt.title("ST-Transformer train")
    plt.xlabel("t")
    plt.ylabel("MAE loss")
    plt.legend()
    plt.show()

    # 保存模型
    torch.save(model, "model.pth")

import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
