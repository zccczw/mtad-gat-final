import torch
import torch.nn as nn

from STTNS.STTNs import STTNSNet
from modules import (
    GRULayer,
    Forecasting_Model
)#从modules.py文件中导入 模型


class MTAD_GAT(nn.Module):
    """ MTAD-GAT model class.

    :param n_features: 输入特征数
    :param window_size: 输入序列的长度
    :param out_dim: 要输出的特征数
    :param kernel_size: 用于一维卷积的内核大小
    :param feat_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in feat-oriented GAT layer
    :param time_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in time-oriented GAT layer
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param gru_n_layers: number of layers in the GRU layer
    :param gru_hid_dim: hidden dimension in the GRU layer
    :param forecast_n_layers: number of layers in the FC-based Forecasting Model
    :param forecast_hid_dim: hidden dimension in the FC-based Forecasting Model
    :param recon_n_layers: number of layers in the GRU-based Reconstruction Model
    :param recon_hid_dim: hidden dimension in the GRU-based Reconstruction Model
    :param dropout: dropout rate
    :param alpha: negative slope used in the leaky rely activation function

    """

    def __init__(
            self,
            n_features,
            window_size,
            out_dim,
            k,
            embed_size,
            in_channels,
            gru_n_layers=1,
            gru_hid_dim=150,
            forecast_n_layers=1,
            forecast_hid_dim=150,
            dropout=0.2
    ):
        super(MTAD_GAT, self).__init__()
        self.sttn = STTNSNet(n_features, k, in_channels, embed_size, window_size, dropout=dropout)
        self.gru = GRULayer(2 * n_features, gru_hid_dim, gru_n_layers, dropout)
        self.forecasting_model = Forecasting_Model(gru_hid_dim, forecast_hid_dim, out_dim, forecast_n_layers, dropout)

        self.feature_idx = torch.arange(n_features).cuda()
        self.temporal_idx = torch.arange(window_size).cuda()

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        b, n, k = x.shape

        data = x.unsqueeze(1).permute(0, 1, 3, 2)  # (b, 1, k, n)
        h = self.sttn(data)
        """
        使用 unsqueeze 函数将维度 1 添加到输入张量的形状中，从而创建一个新的张量 data，
        其形状为 (batch_size, 1, num_features, window_size)。
        然后使用 permute 函数将通道维度与特征维度交换，使得数据格式符合卷积层输入的规范。
        """
        h_cat = torch.cat([x, h.permute(0, 2, 1)], dim=2)  # (b, n, 3k)
        """
        接下来，使用一个自注意力机制层 self.sttn（可能是一个带 scale_dot_product_attention 的神经网络层）处理数据 data，
        然后将原始输入张量 x 与得到的自注意力输出张量拼接起来，形成一个新的形状为 (batch_size, window_size, 3*num_features) 的张量 h_cat。
        
        此后，通过一个全连接的 GRU 网络对这个 h_cat 张量进行序列处理并计算出最后时间戳的隐状态。隐状态再通过 view 函数重塑为形状为 (batch_size, -1) 
        的二维张量 h_end，作为预测模型 forecasting_model 的输入，生成最终的预测结果 predictions。
        """


        _, h_end = self.gru(h_cat)
        h_end = h_end.view(x.shape[0], -1)  # Hidden state for last timestamp      batch_size * gru_hid_dim

        predictions = self.forecasting_model(h_end)

        return predictions
