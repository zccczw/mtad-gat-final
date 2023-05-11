import torch
import torch.nn as nn
from STTNS.GCN_models import GCN
from STTNS.One_hot_encoder import One_hot_encoder
from STTNS.layers import graph_constructor
import numpy as np


# model input shape:[1, N, T]
# model output shape:[N, T]
class STTNSNet(nn.Module):
    def __init__(self, node, k, in_channels, embed_size, time_num,
                 num_layers=1, heads=1, dropout=0.2, forward_expansion=4):
        self.num_layers = num_layers
        super(STTNSNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, embed_size, 1)
        self.transformer = Transformer(embed_size, heads, k, node, time_num, dropout, forward_expansion)
        self.conv2 = nn.Conv2d(embed_size, in_channels, 1)

    def forward(self, x):
        # input x:[B, C, N, T]
        # 通道变换
        x = self.conv1(x)  # [B, embed_size, N, T]

        x = x.permute(0, 2, 3, 1)  # [B, N, T, embed_size]
        x = self.transformer(x, x, x, self.num_layers)  # [B, N, T, embed_size]

        # 预测时间T_dim，转换时间维数
        x = x.permute(0, 3, 1, 2) # [B, embed_size, N, T]
        x = self.conv2(x)

        out = x.squeeze(1)

        return out

#自注意力
class Transformer(nn.Module):
    def  __init__(self, embed_size, heads, k, node, time_num, dropout, forward_expansion):
        super(Transformer, self).__init__()
        self.sttnblock = STTNSNetBlock(embed_size, heads, k, node, time_num, dropout, forward_expansion)

    def forward(self, query, key, value, num_layers):
        q, k, v = query, key, value
        for i in range(num_layers):
            out = self.sttnblock(q, k, v)
            q, k, v = out, out, out
        return out


# model input:[N, T, C]
# model output[N, T, C]
class STTNSNetBlock(nn.Module):
    def __init__(self, embed_size, heads, k, node, time_num, dropout, forward_expansion):
        super(STTNSNetBlock, self).__init__()
        self.SpatialTansformer = STransformer(embed_size, heads, k, node, dropout, forward_expansion)
        self.TemporalTransformer = TTransformer(embed_size, heads, time_num, dropout, forward_expansion)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        out1 = self.norm1(self.SpatialTansformer(query, key, value) + query)
        out2 = self.dropout(self.norm2(self.TemporalTransformer(out1, out1, out1) + out1))

        return out2

#融合们g在这里
# model input:[N, T, C]
# model output:[N, T, C]
class STransformer(nn.Module):
    def __init__(self, embed_size, heads, k, node, dropout, forward_expansion):
        super(STransformer, self).__init__()
        self.node_idx = torch.arange(node).cuda()
        self.gen_adj = graph_constructor(node, k, embed_size)
        # self.D_S = nn.Parameter(adj)
        self.embed_linear = nn.Linear(node, embed_size)
        # self.embed_linear = nn.Embedding(node, embed_size)
        self.attention = SSelfattention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        # 调用GCN
        self.gcn = GCN(embed_size, embed_size * 2, embed_size, dropout)
        self.norm_adj = nn.InstanceNorm2d(1)  # 对邻接矩阵归一化

        self.dropout = nn.Dropout(dropout)
        self.fs = nn.Linear(embed_size, embed_size)
        self.fg = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value):
        # Spatial Embedding 部分
        B, N, T, C = query.shape
        adj = self.gen_adj(self.node_idx)

        D_S = self.embed_linear(nn.Parameter(adj))
        D_S = D_S.expand(T, N, C)
        D_S = D_S.permute(1, 0, 2)

        # GCN 部分
        X_G = torch.Tensor(0, N, T, C).cuda()
        adj = adj.unsqueeze(0).unsqueeze(0)
        adj = self.norm_adj(adj)
        adj = adj.squeeze(0).squeeze(0)

        for i in range(B):   #B是query的shape
            X_Gt = torch.Tensor(query.shape[1], 0, query.shape[3]).cuda()
            for t in range(query.shape[2]):
                o = self.gcn(query[i, :, t, :], adj)
                o = o.unsqueeze(1)  # shape [N, 1, C]
                X_Gt = torch.cat((X_Gt, o), dim=1)
            X_Gt = X_Gt.unsqueeze(0)
            X_G = torch.cat((X_G, X_Gt), dim=0)

        # spatial transformer
        query = query + D_S
        value = value + D_S
        key = key + D_S
        attn = self.attention(value, key, query)  # [N, T, C]
        M_s = self.dropout(self.norm1(attn + query))
        feedforward = self.feed_forward(M_s)
        U_s = self.dropout(self.norm2(feedforward + M_s))

        # 融合
        g = torch.sigmoid(self.fs(U_s) + self.fg(X_G))
        out = g * U_s + (1 - g) * X_G

        return out


#时间局部注意力，投射到高维空间，捕获，根据变化的高维信号进行建模
# model input:[N,T,C]
# model output:[N,T,C]
class SSelfattention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SSelfattention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.per_dim = embed_size // heads
        self.values = nn.Linear(self.per_dim, self.per_dim, bias=False)
        self.queries = nn.Linear(self.per_dim, self.per_dim, bias=False)
        self.keys = nn.Linear(self.per_dim, self.per_dim, bias=False)
        self.fc = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query):
        B, N, T, C = query.shape
        query = values.reshape(B, N, T, self.heads, self.per_dim)
        keys = keys.reshape(B, N, T, self.heads, self.per_dim)
        values = values.reshape(B, N, T, self.heads, self.per_dim)

        # q, k, v:[N, T, heads, per_dim]
        queries = self.queries(query)
        keys = self.keys(keys)
        values = self.values(values)

        # spatial self-attention       空间注意力模块  ,爱因斯坦求和约定
        attn = torch.einsum("bqthd, bkthd->bqkth", (queries, keys))  # [N, N, T, heads]
        attention = torch.softmax(attn / (self.embed_size ** (1 / 2)), dim=1)
                            #ij,jk->i,k,其改变的是维度的形状
        out = torch.einsum("bqkth,bkthd->bqthd", (attention, values))  # [N, T, heads, per_dim]
        out = out.reshape(B, N, T, self.heads * self.per_dim)  # [N, T, C]

        out = self.fc(out)

        return out


# input[N, T, C]
class TTransformer(nn.Module):
    def __init__(self, embed_size, heads, time_num, dropout, forward_expansion):
        super(TTransformer, self).__init__()
        # Temporal embedding One hot
        self.time_num = time_num
        self.one_hot = One_hot_encoder(embed_size, time_num).cuda()  # temporal embedding选用one-hot方式 或者
        self.temporal_embedding = nn.Embedding(time_num, embed_size).cuda()  # temporal embedding选用nn.Embedding

        self.attention = TSelfattention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
            #对应局部注意力部分的relu激活函数
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # q, k, v：[N, T, C]
        B, N, T, C = query.shape

        # D_T = self.one_hot(t, N, T)  # temporal embedding选用one-hot方式 或者
        D_T = self.temporal_embedding(torch.arange(0, T).cuda())  # temporal embedding选用nn.Embedding
        D_T = D_T.expand(N, T, C)

        # TTransformer
        x = D_T + query
        attention = self.attention(x, x, x)
        M_t = self.dropout(self.norm1(attention + x))
        feedforward = self.feed_forward(M_t)
        U_t = self.dropout(self.norm2(M_t + feedforward))

        out = U_t + x + M_t

        return out


class TSelfattention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TSelfattention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.per_dim = self.embed_size // heads
        self.queries = nn.Linear(self.per_dim, self.per_dim)
        self.keys = nn.Linear(self.per_dim, self.per_dim)
        self.values = nn.Linear(self.per_dim, self.per_dim)
        self.fc = nn.Linear(embed_size, embed_size)

    def forward(self, value, key, query):
        # q, k, v:[N, T, C]
        B, N, T, C = query.shape

        # q, k, v:[N,T,heads, per_dim]
        keys = key.reshape(B, N, T, self.heads, self.per_dim)
        queries = query.reshape(B, N, T, self.heads, self.per_dim)
        values = value.reshape(B, N, T, self.heads, self.per_dim)

        keys = self.keys(keys)
        values = self.values(values)
        queries = self.queries(queries)

        # compute temperal self-attention
        attnscore = torch.einsum("bnqhd, bnkhd->bnqkh", (queries, keys))  # [N, T, T, heads]
        attention = torch.softmax(attnscore / (self.embed_size ** (1 / 2)), dim=2)

        out = torch.einsum("bnqkh, bnkhd->bnqhd", (attention, values))  # [N, T, heads, per_dim]
        out = out.reshape(B, N, T, self.embed_size)
        out = self.fc(out)

        return out
