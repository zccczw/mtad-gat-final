import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from entmax import entmax_bisect
from torch_geometric.nn import GATv2Conv

class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)  #做嵌入
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a)) #领接矩阵
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)  #构造掩码层
        mask.fill_(float('0'))
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        edge_indexes = [
            [],
            []
        ]
        l = len(adj)
        for i in range(0, l):
            for j in range(0, l):
                if adj[i][j] > 0:
                    edge_indexes[0].append(j)
                    edge_indexes[1].append(i)
        return edge_indexes

    def fullA(self, idx):               #如果没有定义，则定义
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        return adj


class ConvLayer(nn.Module):
    """1-D Convolution layer to extract high-level features of each time-series input   1-D卷积层，提取每个时间序列输入的高级特征
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param kernel_size: size of kernel to use in the convolution operation      用于卷积运算的核的大小
    """

    def __init__(self, n_features, kernel_size=7):
        super(ConvLayer, self).__init__()
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)  # 用常数值填充输入张量边界。
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_features,
                              kernel_size=kernel_size)  # 对由多个输入平面组成的输入信号应用一维卷积。
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.padding(x)
        x = self.relu(self.conv(x))
        return x.permute(0, 2, 1)  # Permute back


class FeatureAttentionLayer(nn.Module):
    """Single Graph Feature/Spatial Attention Layer     单图特征/空间注意层
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely activation function     leakyRely激活函数中使用的负斜率
    :param embed_dim: embedding dimension (output dimension of linear transformation)       嵌入维数(线性变换输出维数)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer      是否要在注意力层中加入偏见项
    """

    def __init__(self, n_features, window_size, k, dropout, alpha, embed_dim=None, use_gatv2=True, use_bias=True):
        super(FeatureAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.k = k
        self.dropout = dropout
        self.embed_dim = embed_dim if embed_dim is not None else window_size
        self.use_gatv2 = use_gatv2
        self.num_nodes = n_features
        self.use_bias = use_bias

        # Because linear transformation is done after concatenation in GATv2        因为线性变换是在GATv2连接之后进行的
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * window_size
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = window_size
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)  # 对传入的数据应用线性变换::math: ' y = xA^T + b
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))  # 返回一个填满未初始化数据的张量。张量的形状由变量的大小来定义。
        nn.init.xavier_uniform_(self.a.data,
                                gain=1.414)  # 用来初始化卷积核, 目的是为了使得每一层的方差都尽可能相等, 使网络中的信息更好地流动. 则将每一层权重初始化为如下范围内的均匀分布

        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(n_features, k))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, adj):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For feature attention we represent a node as the values of a particular feature across all timestamps

        x = x.permute(0, 2, 1)
        blocks = x.repeat(1, self.num_nodes, 1)  # Right-side of the matrix      沿着指定的维度重复这个张量。size : 在每个维度上重复这个张量的次数
        l = []
        for i, _ in enumerate(adj):
            for j, val in enumerate(_):
                if val > 0:
                    l.append(i * self.num_nodes + j)
        vj = blocks[:, l, :]
        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu 线性变换应用在拼接后，注意层应用在leakyrelu后
        if self.use_gatv2:
            a_input = self._make_attention_input(x, adj, vj, self.k)  # (b, k, k, 2*window_size)
            a_input = self.leakyrelu(self.lin(a_input))  # (b, k, k, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)  # (b, k, k, 1)

        # Original GAT attention
        else:
            Wx = self.lin(x)  # (b, k, k, embed_dim)
            a_input = self._make_attention_input(Wx)  # (b, k, k, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)  # (b, k, k, 1)  扩展维度，也就是到4维

        if self.use_bias:
            e += self.bias

        # Attention weights
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)

        # Computing new node features using the attention
        h = self.sigmoid(torch.matmul(attention, vj))

        return h.permute(0, 2, 1)

    def _make_attention_input(self, v, a,vj , k):
        """Preparing the feature attention mechanism.                                   准备特征注意机制。
        Creating matrix with all possible combinations of concatenations of node.       创建包含节点连接的所有可能组合的矩阵。
        Each node consists of all values of that node within the window                 每个节点由窗口中该节点的所有值组成
            v1 || v1,
            ...
            v1 || vK,
            v2 || v1,
            ...
            v2 || vK,
            ...
            ...
            vK || v1,
            ...
            vK || vK,
        """

        K = self.num_nodes

        blocks_repeating = v.repeat_interleave(k, dim=1)  # Left-side of the matrix     重复张量和输入的形状相同，除了沿着给定的轴。

        blocks_alternating = vj

        combined = torch.cat((blocks_repeating, blocks_alternating),
                             dim=2)  # (b, K*K, 2*window_size)在给定维中连接给定序列的seq张量。所有张量必须具有相同的形状（连接维度中除外）或为空。

        if self.use_gatv2:
            return combined.view(v.size(0), K, k, 2 * self.window_size)
        else:
            return combined.view(v.size(0), K, k, 2 * self.embed_dim)


class TemporalAttentionLayer(nn.Module):
    """Single Graph Temporal Attention Layer        单图时间注意层
    :param n_features: number of input features/nodes
    :param window_size: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation) 嵌入维数(线性变换输出维数)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer

    """

    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_gatv2=True, use_bias=True):
        super(TemporalAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.use_gatv2 = use_gatv2
        self.embed_dim = embed_dim if embed_dim is not None else n_features
        self.num_nodes = window_size
        self.use_bias = use_bias

        # Because linear transformation is performed after concatenation in GATv2       因为在GATv2中连接后进行线性变换
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * n_features
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = n_features
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)    #使用xavier防止梯度爆炸和梯度消失

        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(window_size, window_size))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For temporal attention a node is represented as all feature values at a specific timestamp

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        if self.use_gatv2:
            a_input = self._make_attention_input(x)  # (b, n, n, 2*n_features)
            a_input = self.leakyrelu(self.lin(a_input))  # (b, n, n, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)  # (b, n, n, 1)

        # Original GAT attention
        else:
            Wx = self.lin(x)  # (b, n, n, embed_dim)
            a_input = self._make_attention_input(Wx)  # (b, n, n, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)  # (b, n, n, 1)

        if self.use_bias:
            e += self.bias  # (b, n, n, 1)

        # Attention weights
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)

        h = self.sigmoid(torch.matmul(attention, x))  # (b, n, k)

        return h

    def _make_attention_input(self, v):
        """Preparing the temporal attention mechanism.
        Creating matrix with all possible combinations of concatenations of node values:
            (v1, v2..)_t1 || (v1, v2..)_t1
            (v1, v2..)_t1 || (v1, v2..)_t2

            ...
            ...

            (v1, v2..)_tn || (v1, v2..)_t1
            (v1, v2..)_tn || (v1, v2..)_t2

        """

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.n_features)
        else:
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)


class DualAttention(nn.Module):
    def __init__(self, item_dim, pos_dim, n_items, n_pos, w, atten_way='dot', decoder_way='bilinear', dropout=0,
                 activate='relu'):
        super(DualAttention, self).__init__()
        self.item_dim = item_dim
        self.pos_dim = pos_dim
        dim = item_dim + pos_dim
        self.dim = dim
        self.n_items = n_items
        self.embedding = nn.Embedding(n_items + 1, item_dim, padding_idx=0, max_norm=1.5, dtype=torch.float32)
        self.pos_embedding = nn.Embedding(n_pos, pos_dim, padding_idx=0, max_norm=1.5, dtype=torch.float32)
        self.atten_way = atten_way
        self.decoder_way = decoder_way

        self.atten_w0 = nn.Parameter(torch.Tensor(1, dim))
        self.atten_w1 = nn.Parameter(torch.Tensor(dim, dim))
        self.atten_w2 = nn.Parameter(torch.Tensor(dim, dim))
        self.atten_bias = nn.Parameter(torch.Tensor(dim))
        self.w_f = nn.Linear(2 * dim, item_dim)

        self.dropout = nn.Dropout(dropout)

        self.self_atten_w1 = nn.Linear(dim, dim)
        self.self_atten_w2 = nn.Linear(dim, dim)

        self.LN = nn.LayerNorm(dim)
        self.LN2 = nn.LayerNorm(item_dim)
        self.is_dropout = True
        self.attention_mlp = nn.Linear(dim, dim)

        self.alpha_w = nn.Linear(dim, 1)
        self.w = w

        if activate == 'relu':
            self.activate = F.relu
        elif activate == 'selu':
            self.activate = F.selu

        self.initial_()

    def initial_(self):

        init.normal_(self.atten_w0, 0, 0.05)
        init.normal_(self.atten_w1, 0, 0.05)
        init.normal_(self.atten_w2, 0, 0.05)
        init.constant_(self.atten_bias, 0)
        init.constant_(self.attention_mlp.bias, 0)
        init.constant_(self.embedding.weight[0], 0)
        init.constant_(self.pos_embedding.weight[0], 0)

    def forward(self, x, pos):
        self.is_dropout = True
        x_embeddings = self.embedding(x)  # B,seq,dim
        pos_embeddings = self.pos_embedding(pos)  # B, seq, dim
        mask = (x != 0).float()  # B,seq
        x_ = torch.cat((x_embeddings, pos_embeddings), 2)  # B seq, 2*dim
        x_s = x_[:, :-1, :]  # B, seq-1, 2*dim
        alpha_ent = self.get_alpha(x=x_[:, -1, :], number=0)
        m_s, x_n = self.self_attention(x_, x_, x_, mask, alpha_ent)
        alpha_global = self.get_alpha(x=m_s, number=1)
        global_c = self.global_attention(m_s, x_n, x_s, mask, alpha_global)  # B, 1, dim
        h_t = global_c
        result = self.decoder(h_t, m_s)
        return result

    def get_alpha(self, x=None, number=None):
        if number == 0:
            alpha_ent = torch.sigmoid(self.alpha_w(x)) + 1
            alpha_ent = self.add_value(alpha_ent).unsqueeze(1)
            alpha_ent = alpha_ent.expand(-1, 285, -1)  # 这里不一样
            return alpha_ent
        if number == 1:
            alpha_global = torch.sigmoid(self.alpha_w(x)) + 1
            alpha_global = self.add_value(alpha_global)
            return alpha_global

    def add_value(self, value):
        mask_value = (value == 1).float()
        value = value.masked_fill(mask_value == 1, 1.0001)  # 这里不一样
        return value

    def self_attention(self, q, k, v, mask=None, alpha_ent=1):
        if self.is_dropout:
            q_ = self.dropout(self.activate(self.attention_mlp(q)))
        else:
            q_ = self.activate(self.attention_mlp(q))
        scores = torch.matmul(q_, k.transpose(1, 2)) / math.sqrt(self.dim)
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, q.size(1), -1)
            scores = scores.masked_fill(mask == 0, -np.inf)
        alpha = entmax_bisect(scores, alpha_ent, dim=-1)
        att_v = torch.matmul(alpha, v)  # B, seq, dim
        if self.is_dropout:
            att_v = self.dropout(self.self_atten_w2(self.activate(self.self_atten_w1(att_v)))) + att_v
        else:
            att_v = self.self_atten_w2(self.activate(self.self_atten_w1(att_v))) + att_v
        att_v = self.LN(att_v)
        c = att_v[:, -1, :].unsqueeze(1)
        x_n = att_v[:, :-1, :]
        return c, x_n

#全局注意力层
    def global_attention(self, target, k, v, mask=None, alpha_ent=1):
        alpha = torch.matmul(
            torch.relu(k.matmul(self.atten_w1) + target.matmul(self.atten_w2) + self.atten_bias),
            self.atten_w0.t())  # (B,seq,1)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            mask = mask[:, :-1, :]
            alpha = alpha.masked_fill(mask == 0, -np.inf)
        alpha = entmax_bisect(alpha, alpha_ent, dim=1)
        c = torch.matmul(alpha.transpose(1, 2), v)  # (B, 1, dim)
        return c

    def decoder(self, global_c, self_c):
        if self.is_dropout:
            c = self.dropout(torch.selu(self.w_f(torch.cat((global_c, self_c), 2))))
        else:
            c = torch.selu(self.w_f(torch.cat((global_c, self_c), 2)))
        c = c.squeeze()
        l_c = (c / torch.norm(c, dim=-1).unsqueeze(1))
        l_emb = self.embedding.weight[1:-1] / torch.norm(self.embedding.weight[1:-1], dim=-1).unsqueeze(1)
        z = self.w * torch.matmul(l_c, l_emb.t())

        return z

    def predict(self, x, pos, k=20):
        self.is_dropout = False
        x_embeddings = self.embedding(x)  # B,seq,dim
        pos_embeddings = self.pos_embedding(pos)  # B, seq, dim
        mask = (x != 0).float()  # B,seq
        x_ = torch.cat((x_embeddings, pos_embeddings), 2)  # B seq, 2*dim
        x_s = x_[:, :-1, :]  # B, seq-1, 2*dim
        alpha_ent = self.get_alpha(x=x_[:, -1, :], number=0)
        m_s, x_n = self.self_attention(x_, x_, x_, mask, alpha_ent)
        alpha_global = self.get_alpha(x=m_s, number=1)
        global_c = self.global_attention(m_s, x_n, x_s, mask, alpha_global)  # B, 1, dim
        h_t = global_c
        result = self.decoder(h_t, m_s)
        rank = torch.argsort(result, dim=1, descending=True)
        return rank[:, 0:k]


class GRULayer(nn.Module):
    """Gated Recurrent Unit (GRU) Layer     单图时间注意层
    :param in_dim: number of input features
    :param hid_dim: hidden size of the GRU
    :param n_layers: number of layers in GRU
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(GRULayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.gru = nn.GRU(in_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        out, h = self.gru(x)
        out, h = out[-1, :, :], h[-1, :, :]  # Extracting from last layer
        return out, h


class RNNDecoder(nn.Module):
    """GRU-based Decoder network that converts latent vector into output        基于GRU的解码网络，将潜在向量转换为输出
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(RNNDecoder, self).__init__()
        self.in_dim = in_dim
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.rnn = nn.GRU(in_dim, hid_dim, n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        decoder_out, _ = self.rnn(x)
        return decoder_out


class ReconstructionModel(nn.Module):
    """Reconstruction Model         重建模型
    :param window_size: length of the input sequence
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param in_dim: number of output features
    :param dropout: dropout rate
    """

    def __init__(self, window_size, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(ReconstructionModel, self).__init__()
        self.window_size = window_size
        self.decoder = RNNDecoder(in_dim, hid_dim, n_layers, dropout)  # 基于GRU的解码网络，将潜在向量转换为输出
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        # x will be last hidden state of the GRU layer
        h_end = x
        h_end_rep = h_end.repeat_interleave(self.window_size, dim=1).view(x.size(0), self.window_size, -1)

        decoder_out = self.decoder(h_end_rep)
        out = self.fc(decoder_out)
        return out


class Forecasting_Model(nn.Module):
    """Forecasting model (fully-connected network)     预测模型(全连通网络)
    :param in_dim: number of input features
    :param hid_dim: hidden size of the FC network
    :param out_dim: number of output features
    :param n_layers: number of FC layers
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(Forecasting_Model, self).__init__()
        layers = [nn.Linear(in_dim, hid_dim)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))

        layers.append(nn.Linear(hid_dim, out_dim))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
            x = self.dropout(x)
        return self.layers[-1](x)

def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i * edge_num:(i + 1) * edge_num] += i * node_num

    return batch_edge_index.long()