import torch
import math
from torch import nn
import torch.nn.functional as F

class Base_Self_Attention(nn.Module):
    """Base Self-Attention Mechanism 基础自注意力机制模块
    Base class for both temporal and spatial attention.
    时序和空间注意力的基类。

    Args:
        latent_dim (int): Dimensionality of the input features. 输入特征的维度。
        num_head (int): Number of attention heads. 注意力头的数量。
        dropout (float): Dropout rate. Dropout比率。
    """
    def __init__(self, latent_dim, num_head, dropout):
        super().__init__()
        self.num_head = num_head
        # Layer normalization 层归一化
        self.norm = nn.LayerNorm(latent_dim)
        # Define Q,K,V linear transformations without bias 定义Q,K,V三个线性变换，不使用偏置项
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward Propagation 前向传播函数

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, D) 输入张量
                B: Batch size 批次大小
                N: Sequence length (T) or spatial locations (S) 序列长度或空间位置数
                D: Feature dimension 特征维度

        Returns:
            torch.Tensor: Output tensor of shape (B, N, D)
        """
        B, N, D = x.shape   # Batch, Sequence/Spatial dimension, Feature dimension
        H = self.num_head   # Number of attention heads 注意力头数量

        # 1. Query transformation Query变换
        # Shape transformation 维度变换: [B, N, D] -> [B, N, 1, D]
        query = self.query(self.norm(x)).unsqueeze(2)

        # 2. Key transformation Key变换
        # Shape transformation 维度变换: [B, N, D] -> [B, 1, N, D]
        key = self.key(self.norm(x)).unsqueeze(1)

        # 3. Reshape for multi-head attention 重塑为多头注意力格式
        # Shape transformation 维度变换: [B, N, 1/1, D] -> [B, N, H, D/H]
        query = query.view(B, N, H, -1)  
        key = key.view(B, N, H, -1)      

        # 4. Compute attention scores 计算注意力分数
        # Input shapes 输入形状:
        #   query: [B, N, H, D/H]
        #   key:   [B, N, H, D/H]
        # Output shape 输出形状: [B, N, N, H]
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)

        # 5. Apply softmax and dropout 应用softmax和dropout
        # Shape remains 形状保持: [B, N, N, H]
        weight = self.dropout(F.softmax(attention, dim=2))

        # 6. Value transformation 值变换
        # Shape transformation 维度变换: [B, N, D] -> [B, N, H, D/H]
        value = self.value(self.norm(x)).view(B, N, H, -1)

        # 7. Compute weighted sum 计算加权和
        # Input shapes 输入形状:
        #   weight: [B, N, N, H]
        #   value:  [B, N, H, D/H]
        # Output shape 输出形状: [B, N, D]
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, N, D)

        # 8. Add residual connection 添加残差连接
        y = x + y
        return y

class Temporal_Self_Attention(Base_Self_Attention):
    """Temporal Self-Attention Mechanism 时序自注意力机制
    Inherits from BaseAttention. Processes temporal sequences.
    继承自BaseAttention。处理时序序列。
    
    This class uses the base self-attention implementation as temporal attention
    operates on sequences in the same way as standard self-attention.
    此类使用基础自注意力实现，因为时序注意力在序列上的操作方式与标准自注意力相同。
    
    The sequence dimension N represents time steps T in this context.
    序列维度N在此上下文中表示时间步长T。
    """
    pass

class Spatial_Self_Attention(Base_Self_Attention):
    """Spatial Self-Attention Mechanism 空间自注意力机制
    Inherits from BaseAttention. Processes spatial features.
    继承自BaseAttention。处理空间特征。
    
    This class uses the base self-attention implementation as spatial attention
    operates on spatial locations in the same way as standard self-attention.
    此类使用基础自注意力实现，因为空间注意力在空间位置上的操作方式与标准自注意力相同。
    
    The sequence dimension N represents spatial locations S in this context.
    序列维度N在此上下文中表示空间位置S。
    """
    pass
 
class Base_Cross_Attention(nn.Module):
    """Base Cross-Attention Mechanism 基础交叉注意力机制模块
    Base class for both temporal and spatial cross-attention.
    时序和空间交叉注意力的基类。

    Args:
        latent_dim (int): Dimensionality of the primary input features (x). 主输入特征维度。
        mode_dim (int): Dimensionality of the context features (xf). 上下文特征维度，可以与latent_dim不同。
        num_head (int): Number of attention heads. 注意力头的数量。
        dropout (float): Dropout rate. Dropout比率。
    
    Note:
        latent_dim 和 mode_dim 可以不同，这允许处理不同模态或不同维度的特征之间的交互。
        The latent_dim and mode_dim can be different, allowing interaction between features
        of different modalities or dimensions.
    """
    def __init__(self, latent_dim, mode_dim, num_head, dropout):
        super().__init__()
        self.num_head = num_head
        # Layer normalization 层归一化
        self.norm = nn.LayerNorm(latent_dim)
        # Modal normalization 模态归一化
        self.mode_norm = nn.LayerNorm(mode_dim)
        # Define Q,K,V linear transformations without bias 定义Q,K,V三个线性变换，不使用偏置项
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, xf):
        """Forward Propagation 前向传播函数

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, D) 输入张量
                B: Batch size 批次大小
                N: Sequence length (T) or spatial locations (S) 序列长度或空间位置数
                D: Feature dimension 特征维度
            xf (torch.Tensor): Context tensor of shape (B, M, L) 上下文张量
                M: Context length (T) or spatial locations (S) 上下文长度或空间位置数
                L: Feature dimension 特征维度

        Returns:
            torch.Tensor: Output tensor of shape (B, N, D)
        """
        B, N, D = x.shape   # Batch, Sequence/Spatial dimension, Feature dimension
        M = xf.shape[1]     # Context length/Spatial dimension 上下文长度/空间维度
        H = self.num_head   # Number of attention heads 注意力头数量

        # 1. Query transformation Query变换
        # Shape transformation 维度变换: [B, N, D] -> [B, N, 1, D]
        query = self.query(self.norm(x)).unsqueeze(2)

        # 2. Key transformation Key变换
        # Shape transformation 维度变换: [B, M, L] -> [B, 1, M, L]
        key = self.key(self.mode_norm(xf)).unsqueeze(1)

        # 3. Reshape for multi-head attention 适应多头注意力机制
        query = query.view(B, N, H, -1)  # [B, N, 1, D] -> [B, N, H, D/H]
        key = key.view(B, M, H, -1)      # [B, 1, M, L] -> [B, M, H, L/H]

        # 4. Compute attention scores 计算注意力得分
        # Shape transformation 维度变换: [B, N, H, D/H] x [B, M, H, L/H] -> [B, N, M, H]
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)

        # 5. Apply softmax and dropout 应用softmax和dropout
        # Shape remains 形状保持: [B, N, M, H]
        weight = self.dropout(F.softmax(attention, dim=2))  # [B, N, M, H]

        # 6. Compute context vector 计算上下文向量
        # Shape transformation [B, M, L] -> [B, M, H, L/H]
        value = self.value(self.mode_norm(xf)).view(B, M, H, -1)

        # 7. Compute output
        # Shape transformation 维度变换: [B, N, M, H] x [B, M, H, L/H] -> [B, N, D]
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, N, D)
        y = x + y  # Residual connection 残差连接
        return y
    

class Temporal_Cross_Attention(Base_Cross_Attention):
    """Temporal Cross-Attention Mechanism 时序交叉注意力机制
    Inherits from BaseCrossAttention. Processes temporal sequences.
    继承自BaseCrossAttention。处理时序序列。
    
    This class enables attention between two temporal sequences, where:
    此类实现了两个时序序列之间的注意力机制，其中：
    - N represents the primary sequence length T1
    - N表示主要序列长度T1
    - M represents the context sequence length T2
    - M表示上下文序列长度T2
    
    Useful for relating events across different time scales or temporal contexts.
    适用于关联不同时间尺度或时序上下文中的事件。
    """
    pass

class Spatial_Cross_Attention(Base_Cross_Attention):
    """Spatial Cross-Attention Mechanism 空间交叉注意力机制
    Inherits from BaseCrossAttention. Processes spatial features.
    继承自BaseCrossAttention。处理空间特征。
    
    This class enables attention between two sets of spatial features, where:
    此类实现了两组空间特征之间的注意力机制，其中：
    - N represents the primary spatial locations S1
    - N表示主要空间位置S1
    - M represents the context spatial locations S2
    - M表示上下文空间位置S2
    
    Useful for relating features across different spatial regions or resolutions.
    适用于关联不同空间区域或分辨率的特征。
    """
    pass