import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math


class graph_convolution(nn.Module):
    """Spatiotemporal graph convolution operation
    时空图卷积操作
    
    Performs graph convolution in both temporal and spatial domains using learnable
    adjacency matrices and feature transformation.
    使用可学习的邻接矩阵和特征变换在时间和空间域进行图卷积。

    Args:
        in_features (int): Input feature dimension C 输入特征维度C
        out_features (int): Output feature dimension C' 输出特征维度C'
        node_n (int): Number of graph nodes N 图节点数量N
        seq_len (int): Sequence length T 序列长度T
        bias (bool): Whether to use bias 是否使用偏置
    """
    def __init__(self, in_features, out_features, node_n=21, seq_len=40, bias=True):
        super().__init__()
        
        # A_t ∈ R^{T×T}: temporal adjacency matrix 时间邻接矩阵
        self.temporal_graph_weights = Parameter(torch.FloatTensor(seq_len, seq_len))
        # W ∈ R^{d×d'}: feature transform matrix 特征变换矩阵
        self.feature_weights = Parameter(torch.FloatTensor(in_features, out_features))
        # A_s ∈ R^{N×N}: spatial adjacency matrix 空间邻接矩阵
        self.spatial_graph_weights = Parameter(torch.FloatTensor(node_n, node_n))

        if bias:
            self.bias = Parameter(torch.FloatTensor(seq_len))

        self.reset_parameters()

    def reset_parameters(self):
       stdv = 1. / math.sqrt(self.spatial_graph_weights.size(1))
       self.feature_weights.data.uniform_(-stdv, stdv)
       self.temporal_graph_weights.data.uniform_(-stdv, stdv)
       self.spatial_graph_weights.data.uniform_(-stdv, stdv)
       if self.bias is not None:
           self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """Forward pass of spatiotemporal graph convolution
        时空图卷积的前向传播
        
        Steps:
        1. X × A_t: Apply temporal graph convolution
           时间图卷积
        2. W × X: Transform features
           特征变换
        3. A_s × X: Apply spatial graph convolution
           空间图卷积
           
        Shape:
            input (B,C,N,T) -> output (B,C',N,T)
            where B=batch, C=channels, N=nodes, T=time steps
        """
        # 1. Temporal convolution 时间卷积 (B,C,V,T)->(B,C,V,T)
        y = torch.matmul(input, self.temporal_graph_weights)
        
        # 2. Feature transform 特征变换 (B,C,V,T)->(B,C',V,T)
        y = torch.matmul(y.permute(0, 3, 2, 1), self.feature_weights)
        
        # 3. Spatial convolution 空间卷积 (B,C',V,T)->(B,C',V,T)
        y = torch.matmul(self.spatial_graph_weights, y).permute(0, 3, 2, 1).contiguous()

        if self.bias is not None:
            y += self.bias
            
        return y

class residual_graph_convolution(nn.Module):
    """Residual graph convolution block with Layer Normalization and Dropout
    带Layer Normalization和Dropout的图卷积残差块
    
    Structure:
        GCN->LayerNorm->Tanh->Dropout->Add
        
    This block applies graph convolution with residual connection.
    Layer normalization and dropout are used to stabilize training
    and prevent overfitting.
    
    该模块应用图卷积并有残差连接。使用Layer Normalization来稳定训练，
    使用Dropout来防止过拟合。

    Args:
        features (int): Feature dimension 特征维度
        node_n (int): Number of nodes 节点数量
        seq_len (int): Sequence length 序列长度
        bias (bool): Whether to use bias 是否使用偏置
        p_dropout (float): Dropout probability dropout概率
    """
    def __init__(self, features, node_n=21, seq_len=40, bias=True, p_dropout=0.3):
        super().__init__()

        # GCN layer GCN层
        self.gcn = graph_convolution(features, features, node_n, seq_len, bias)
        # Layer normalization 层归一化
        self.ln = nn.LayerNorm([features, node_n, seq_len], elementwise_affine=True)
        # Activation function 激活函数
        self.act_f = nn.Tanh()
        # Dropout layer dropout层
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        
        y = self.gcn(x)
        y = self.ln(y)
        y = self.act_f(y)
        y = self.dropout(y)
        
        return y + x
    
class graph_convolution_network(nn.Module):
    """Spatiotemporal Graph Convolutional Network with Residual Connections
    带残差连接的时空图卷积网络

    This network processes spatiotemporal graph data through multiple stages:
    1. Initial feature extraction using graph convolution
    2. Feature refinement through a series of residual blocks
    3. Feature aggregation across temporal dimension
    
    The network maintains spatiotemporal relationships while learning 
    hierarchical features through residual connections.
    
    该网络通过多个阶段处理时空图数据：
    1. 使用图卷积进行初始特征提取
    2. 通过一系列残差块优化特征
    3. 在时间维度上聚合特征
    
    网络在学习分层特征的同时保持时空关系。

    Args:
        in_features (int): Input feature dimension 输入特征维度
        latent_features (int): Hidden feature dimension 隐藏特征维度
        node_n (int): Number of nodes 节点数量
        seq_len (int): Sequence length 序列长度
        p_dropout (float): Dropout probability dropout概率
        residual_gcns_num (int): Number of residual blocks 残差块数量
    """
    def __init__(self, in_features, latent_features, node_n=21, seq_len=40, p_dropout=0.3, residual_gcns_num=1):
        super().__init__()

        self.residual_gcns_num = residual_gcns_num
        self.seq_len = seq_len

        # Initial GCN layer 初始GCN层
        self.start_gcn = graph_convolution(in_features, latent_features, node_n, seq_len)

        # Stack of residual blocks 残差块堆叠
        self.residual_gcns = []
        for i in range(residual_gcns_num):
            self.residual_gcns.append(residual_graph_convolution(latent_features, node_n, seq_len*2, p_dropout))
        self.residual_gcns = nn.ModuleList(self.residual_gcns)


    def forward(self, x):
        """Forward pass through the spatiotemporal GCN
        时空图卷积网络的前向传播
        
        Steps:
        1. Initial feature extraction using GCN 使用GCN进行初始特征提取
        2. Temporal dimension expansion for finer granularity 扩展时间维度以获得更细粒度的特征
        3. Feature refinement through residual blocks 通过残差块优化特征
        4. Feature selection along temporal dimension 在时间维度上选择特征

        Shape:
            Input: (B,C,N,T) 
            Output: (B,C',N,T')
            where:
                B = batch size
                C,C' = input/output channels
                N = number of nodes
                T,T' = input/output time steps
        """
        # 1. Initial GCN 初始GCN
        y = self.start_gcn(x)

        # 2. Double temporal dimension 加倍时间维度
        y = torch.cat((y, y), dim=3)
        
        # 3. Process through residual blocks 通过残差块处理
        for i in range(self.residual_gcns_num):
            y = self.residual_gcns[i](y)

        # 4. Take first half of temporal dimension 取时间维度的前半部分
        y = y[:, :, :, :self.seq_len]
        
        return y