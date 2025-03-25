import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EdgeAwareBiLSTM(nn.Module):
    """BiLSTM边缘判别器，专门分析TAD边界特征"""
    def __init__(self, input_dim=64, hidden_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 降维投影 - 减少计算量
        self.projection = nn.Linear(input_dim, hidden_dim)
        
        # 双向LSTM - 序列边缘分析
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        
        # 边缘评分网络 - 评估边界质量
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features, regions=None, hic_matrix=None):
        """
        分析边缘特征，返回边界评分和边界建议
        
        Args:
            features: 特征张量 [B, C, H, W] 或 [B, L, D]
            regions: 可选的区域列表 [(start, end, type), ...]
            hic_matrix: 可选的Hi-C矩阵
            
        Returns:
            edge_scores: 边缘评分
            boundary_adj: 边界调整建议
        """
        # 处理特征形状
        if features.dim() == 4:  # [B, C, H, W]
            B, C, H, W = features.shape
            features_flat = features.permute(0, 2, 3, 1).reshape(B, H*W, C)
        else:
            features_flat = features  # 已经是 [B, L, D] 格式
        
        # 降维
        features_proj = self.projection(features_flat)  # [B, L, hidden_dim]
        
        # 通过BiLSTM处理序列
        lstm_out, _ = self.bilstm(features_proj)  # [B, L, hidden_dim*2]
        
        # 计算所有位置的边缘评分
        batch_size, seq_len, _ = lstm_out.shape
        edge_scores = torch.zeros(batch_size, seq_len, device=features.device)
        
        for i in range(seq_len):
            edge_scores[:, i] = self.edge_scorer(lstm_out[:, i]).squeeze(-1)
        
        # 为边界加强评分信号
        edge_scores[:, 0] = edge_scores[:, 0] * 1.2  # 增强左边界
        edge_scores[:, -1] = edge_scores[:, -1] * 1.2  # 增强右边界
        
        # 生成边界调整建议 (简单的梯度估计)
        boundary_adj = torch.zeros_like(edge_scores)
        for b in range(batch_size):
            for i in range(1, seq_len-1):
                # 如果当前位置评分低于邻居，建议向邻居方向调整
                if edge_scores[b, i] < edge_scores[b, i-1]:
                    boundary_adj[b, i] = -1  # 向左调整
                elif edge_scores[b, i] < edge_scores[b, i+1]:
                    boundary_adj[b, i] = 1  # 向右调整
        
        return edge_scores, boundary_adj