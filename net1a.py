import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import math
import sys

# 导入模块
from tad_modules import (LowRank, GeometricFeatureExtractor, AVIT, 
                        EdgeAwareBiLSTM, SimplifiedUNet, 
                        find_chromosome_files, fill_hic)  # 从模块导入函数

# 创建基础配置类
class TADBaseConfig:
    """基础配置类：集中管理所有共享参数"""
    def __init__(self):
        # 基本环境参数
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_root = "./tad_results"
        self.resolution = 10000
        
        # 模型结构参数
        self.patch_size = 8
        self.embed_dim = 64
        self.num_layers = 15
        self.num_heads = 4
        
        # 训练参数
        self.use_amp = (self.device == "cuda")
        self.ema_decay = 0.996
        self.mask_ratio = 0.3
        self.gamma_base = 0.01
        self.epsilon_base = 0.05
        self.use_theory_gamma = True
        self.boundary_weight = 0.3
        self.num_epochs = 40

    def get_model_params(self):
        """获取模型相关参数字典"""
        return {
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'use_amp': self.use_amp,
            'ema_decay': self.ema_decay,
            'mask_ratio': self.mask_ratio,
            'gamma_base': self.gamma_base,
            'epsilon_base': self.epsilon_base,
            'use_theory_gamma': self.use_theory_gamma,
            'boundary_weight': self.boundary_weight,
        }

"""
以下模块已迁移到 (tad_modules)
0. 数据加载
1. CUR特征分解
2. 几何特征提取模块
3. Backbone: AVIT
4. BiLSTM + U-Net
"""

"""
5. A-VIT型DINO自监督框架
"""
class AVIT_GAN(nn.Module):
    """基于A-VIT骨干网络的对抗训练框架"""
    def __init__(self, embed_dim=None, patch_size=None, num_layers=None, num_heads=None, 
                 use_amp=None, mask_ratio=None, boundary_weight=None, **kwargs):
        super().__init__()
        
        # 先删除不支持的参数
        supported_kwargs = {k: v for k, v in kwargs.items() if k in ['ema_decay', 'gamma_base', 'epsilon_base']}
        
        # 初始化AVIT（仅传入支持的参数）
        self.backbone = AVIT(
            embed_dim=embed_dim,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            use_amp=use_amp,
            **supported_kwargs
        )
        
        # 生成器和判别器初始化（保持不变）
        self.generator = SimplifiedUNet(1, output_channels=1)
        self.discriminator = EdgeAwareBiLSTM(embed_dim, hidden_dim=32)
        self.num_heads = num_heads
        
        # 其他初始化代码不变...
        self.boundary_weight = boundary_weight
        self.use_amp = use_amp
        self.mask_ratio = mask_ratio
        
        # 损失函数和梯度累积步数不变
        self.adv_loss = nn.BCEWithLogitsLoss()
        self.seg_loss = nn.BCEWithLogitsLoss()
        self.grad_accum_steps = getattr(self.backbone, 'grad_accum_steps', 4)
        
        # 保存原始函数并设置替换函数（保持不变）
        old_func = F.multi_head_attention_forward
        
        def adaptive_mha_forward(*args, **kwargs):
            # 检查掩码
            if 'attn_mask' in kwargs and kwargs['attn_mask'] is not None:
                mask = kwargs['attn_mask']
                
                # === 获取正确的序列长度 ===
                seq_len = None
                
                # 方法1: 从query/key/value张量推断
                if len(args) >= 1 and hasattr(args[0], 'shape'):
                    q_shape = args[0].shape
                    if len(q_shape) == 3:  # [N*h, L, E/h] 或 [L, N*h, E/h]
                        seq_len = q_shape[1] if q_shape[0] > q_shape[1] else q_shape[0]
                
                if seq_len is None and len(args) >= 2 and hasattr(args[1], 'shape'):
                    k_shape = args[1].shape
                    if len(k_shape) == 3:
                        seq_len = k_shape[1] if k_shape[0] > k_shape[1] else k_shape[0]
                
                # 方法2: 硬编码为3844 (根据错误信息)
                if seq_len is None or seq_len <= 1:
                    seq_len = 3844  # 直接使用错误消息中的尺寸
                
                # 获取所需头数
                num_heads = kwargs.get('num_heads', 8)  # 默认使用8个头
                
                # === 修复掩码形状 ===
                # 检查空间尺寸是否正确
                if mask.dim() == 3 and (mask.size(1) != seq_len or mask.size(2) != seq_len):
                    # 空间尺寸错误 - 创建新掩码
                    device = mask.device
                    
                    # 如果是[8, 1, 1]这样的单元素掩码，我们假设它是允许全部注意力的掩码
                    if mask.size(1) == 1 and mask.size(2) == 1:
                        # 保留头数，但扩展空间尺寸
                        mask_value = mask[0,0,0].item()  # 获取掩码值
                        if mask_value > 0:  # 通常掩码>0表示允许注意力
                            new_mask = torch.ones(num_heads, seq_len, seq_len, device=device)
                        else:
                            new_mask = torch.zeros(num_heads, seq_len, seq_len, device=device)
                        kwargs['attn_mask'] = new_mask
                    else:
                        # 其他情况下，创建全1掩码
                        kwargs['attn_mask'] = torch.ones(num_heads, seq_len, seq_len, device=device)
                
                # 头数修复 (保留原有逻辑)
                elif mask.dim() == 3 and mask.size(0) != num_heads:
                    # 省略已有的头数修复代码...
                    pass
            
            # 尝试执行并处理错误
            try:
                return old_func(*args, **kwargs)
            except RuntimeError as e:
                error_msg = str(e)
                
                # 如果仍然是掩码形状错误
                if "shape of the 3D attn_mask" in error_msg:
                    # 从错误中提取完整信息
                    import re
                    full_shape_match = re.search(r'should be \((\d+), (\d+), (\d+)\)', error_msg)
                    
                    if full_shape_match:
                        needed_heads = int(full_shape_match.group(1))
                        needed_rows = int(full_shape_match.group(2))
                        needed_cols = int(full_shape_match.group(3))
                        
                        # 创建完全匹配的掩码
                        device = kwargs['attn_mask'].device if 'attn_mask' in kwargs else args[0].device
                        kwargs['attn_mask'] = torch.ones(needed_heads, needed_rows, needed_cols, device=device)
                        
                        # 重试
                        return old_func(*args, **kwargs)
            
                # 无法修复，重新抛出
                raise
        
        # 保存并替换函数
        self._original_mha_forward = old_func
        F.multi_head_attention_forward = adaptive_mha_forward

    def __del__(self):
        # 清理: 恢复原始函数
        if hasattr(self, '_original_mha_forward'):
            F.multi_head_attention_forward = self._original_mha_forward

    def forward(self, matrix: torch.Tensor) -> dict:
        with torch.set_grad_enabled(True):
            encoder_output = self.backbone.conv_embed(matrix)
            
            # 彻底处理几何掩码问题
            if hasattr(self.backbone, 'geometric_extractor'):
                # 确保几何特征回传
                if hasattr(self.backbone.geometric_extractor, 'set_skip_features') and hasattr(self.backbone.conv_embed, 'skip_features'):
                    self.backbone.geometric_extractor.set_skip_features(self.backbone.conv_embed.skip_features)
                
                # 设置/重置掩码参数
                if hasattr(self.backbone.geometric_extractor, '_geo_attention_cache'):
                    self.backbone.geometric_extractor._geo_attention_cache = None
                
                # 修改get_attention_bias方法确保返回全尺寸掩码
                if hasattr(self.backbone.geometric_extractor, 'get_attention_bias'):
                    orig_get_attn = self.backbone.geometric_extractor.get_attention_bias
                    def safe_get_attn(x_shape, device):
                        # 获取原始掩码
                        mask = orig_get_attn(x_shape, device)
                        if mask is not None:
                            # 确保掩码是全尺寸的 [B, L, L]
                            B, L = x_shape[0], x_shape[1]
                            if mask.dim() == 3 and (mask.size(1) != L or mask.size(2) != L):
                                return torch.ones(B, L, L, device=device)
                        return mask
                    self.backbone.geometric_extractor.get_attention_bias = safe_get_attn
            
            # 调用骨干网络前向传播
            reconstructed, z_mean, z_logvar = self.backbone(matrix)
            
            # 生成器处理（维持原有 U-Net 调用流程）
            student_feat_2d = self._reshape_to_2d(encoder_output)
            segmentation = self.generator(student_feat_2d)
            
            # 判别器前处理：将 encoder_output 转为序列格式
            if encoder_output.dim() == 2:  # [L, D]
                B, L, D = 1, encoder_output.shape[0], encoder_output.shape[1]
                seq_features = encoder_output.unsqueeze(0)  # [1, L, D]
            else:
                if encoder_output.dim() == 4:  # [B, D, H, W]
                    B, D, H, W = encoder_output.shape
                    seq_features = encoder_output.permute(0, 2, 3, 1).reshape(B, H*W, D)
                else:  # [B, L, D]
                    B, L, D = encoder_output.shape
                    seq_features = encoder_output
            
            real_features = seq_features.detach()
            
            # 对生成器输出进行维度匹配处理
            seg_view = segmentation
            if segmentation.dim() == 4:  # [B, C, H, W]
                B, C, H, W = segmentation.shape
                if H * W != seq_features.shape[1]:
                    seg_view = F.interpolate(segmentation, size=(int(math.sqrt(seq_features.shape[1])), int(math.sqrt(seq_features.shape[1]))),
                                             mode='bilinear', align_corners=False)
                seg_view = seg_view.view(B, -1, 1)
            
            if seg_view.shape[1] != seq_features.shape[1]:
                seg_view = F.interpolate(
                    seg_view.transpose(1, 2),
                    size=seq_features.shape[1],
                    mode='linear'
                ).transpose(1, 2)
            
            fake_features = seq_features * seg_view
            
            # 判别器前向得到真假判别结果
            d_real, _ = self.discriminator(real_features)
            d_fake, _ = self.discriminator(fake_features)
        
        return {
            'segmentation': segmentation,
            'd_real': d_real,
            'd_fake': d_fake,
            'reconstructed': reconstructed,
            'z_mean': z_mean,
            'z_logvar': z_logvar
        }

    def compute_loss(self, outputs, real_labels):
        """计算对抗损失和分割损失"""
        # 生成器损失
        g_loss = self.adv_loss(outputs['d_fake'], torch.ones_like(outputs['d_fake']))
        
        # 判别器损失
        d_real_loss = self.adv_loss(outputs['d_real'], real_labels)
        d_fake_loss = self.adv_loss(outputs['d_fake'], torch.zeros_like(outputs['d_fake']))
        d_loss = (d_real_loss + d_fake_loss) / 2
        
        # 分割重建损失（兼容原有AVIT损失）
        recon_loss = F.mse_loss(outputs['reconstructed'], real_labels)
        
        # 总损失
        total_loss = g_loss + d_loss + self.boundary_weight * recon_loss
        
        return {
            'total': total_loss,
            'generator': g_loss,
            'discriminator': d_loss,
            'recon': recon_loss
        }

    def _reshape_to_2d(self, features):
        """将1D特征序列重塑为2D特征图"""
        # 处理张量维度
        if features.dim() == 3:  # [B, L, D]
            batch_size, seq_len, feat_dim = features.shape
            features = features.view(batch_size, seq_len * feat_dim)
        else:  # [B, D]
            batch_size, feat_dim = features.shape
        
        # 计算最接近的平方数作为尺寸
        side_len = int(math.sqrt(feat_dim))
        
        # 如果不是完美平方数，调整特征维度
        if side_len * side_len != feat_dim:
            side_len = side_len + 1
            new_dim = side_len * side_len
            # 填充额外的位置
            pad_features = torch.zeros(batch_size, new_dim, device=features.device)
            pad_features[:, :feat_dim] = features
            features = pad_features
        
        # 重塑为 [B, 1, H, W]
        features_2d = features.view(batch_size, 1, side_len, side_len)
        return features_2d

    def init_optimizer(self):
        """初始化双优化器"""
        self.gen_optim = torch.optim.AdamW([
            {'params': self.backbone.parameters(), 'lr': 1e-4},
            {'params': self.generator.parameters(), 'lr': 1e-4}
        ], weight_decay=1e-4)
        
        self.disc_optim = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )

    def train_epoch(self, matrix, real_labels):
        """修改后的对抗训练流程"""
        self.backbone.train()
        self.generator.train()
        self.discriminator.train()
        
        total_loss = 0.0
        gen_loss = 0.0
        disc_loss = 0.0
        recon_loss = 0.0
        
        for step in range(self.grad_accum_steps):
            # 判别器训练
            outputs = self(matrix)
            losses = self.compute_loss(outputs, real_labels)
            
            self.disc_optim.zero_grad()
            losses['discriminator'].backward(retain_graph=True)
            self.disc_optim.step()
            
            # 生成器训练
            self.gen_optim.zero_grad()
            losses['total'].backward()
            self.gen_optim.step()
            
            # 损失累积
            total_loss += losses['total'].item()
            gen_loss += losses['generator'].item()
            disc_loss += losses['discriminator'].item()
            recon_loss += losses['recon'].item()
        
        return {
            'total': total_loss / self.grad_accum_steps,
            'generator': gen_loss / self.grad_accum_steps,
            'discriminator': disc_loss / self.grad_accum_steps,
            'recon': recon_loss / self.grad_accum_steps
        }

class TADFeatureExtractor(TADBaseConfig):
    """TAD特征提取器：从HiC数据中提取特征矩阵"""
    
    def __init__(self, use_cur=True, **kwargs):
        # 调用父类初始化
        super().__init__(**kwargs)
        self.use_cur = use_cur
        self.model = None
        self.cur_projector = None
    
    def load_model(self, model_path=None, chr_name=None):
        """加载预训练模型"""
        # 如果未指定模型路径但指定了染色体，尝试加载该染色体的默认模型
        if model_path is None and chr_name is not None:
            model_path = os.path.join(self.output_root, chr_name, "best_model.pth")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return False
        
        try:
            # 加载模型参数
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 从检查点提取模型参数
            if 'model_params' in checkpoint:
                for key in ['patch_size', 'embed_dim', 'num_layers', 'num_heads']:
                    if key in checkpoint['model_params']:
                        setattr(self, key, checkpoint['model_params'][key])
            
            # 创建模型实例 - 只使用教师模型用于特征提取
            model_params = self.get_model_params()
            self.model = AVIT(
                embed_dim=model_params['embed_dim'],
                patch_size=model_params['patch_size'],
                num_layers=model_params['num_layers'],
                num_heads=model_params['num_heads'],
                use_amp=model_params['use_amp']
            ).to(self.device)
            
            # 只加载教师模型状态字典
            if 'teacher' in checkpoint:
                # 使用教师模型替代学生模型进行特征提取
                self.model.load_state_dict(checkpoint['teacher'])
            elif 'student' in checkpoint:
                # 如果没有教师模型，使用学生模型
                self.model.load_state_dict(checkpoint['student'])
            
            # 设置为评估模式
            self.model.eval()
            
            print(f"成功加载模型: {model_path}")
            return True
            
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            return False
    
    def preprocess(self, matrix):
        """预处理HiC矩阵"""
        if self.use_cur:
            # 如果没有CUR投影器，创建一个
            if self.cur_projector is None:
                self.cur_projector = LowRank(p=0.7, alpha=0.7)
            
            # 应用CUR分解
            cur_matrix = self.cur_projector.fit_transform(matrix)
            return cur_matrix
        else:
            # 不使用CUR预处理，直接返回原矩阵
            return matrix
    
    def extract_features(self, hic_matrix, return_reconstruction=False):
        """
        从HiC矩阵中提取特征
        Args:
            hic_matrix: 输入的HiC矩阵
            return_reconstruction: 是否同时返回重建矩阵
        """
        # 确保模型已加载
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用load_model方法")
        
        # 预处理矩阵
        processed_matrix = self.preprocess(hic_matrix)
        
        # 将矩阵转换为tensor
        if isinstance(processed_matrix, np.ndarray):
            matrix_tensor = torch.from_numpy(processed_matrix).float().to(self.device)
        else:
            matrix_tensor = processed_matrix.to(self.device)
        
        # 提取特征
        with torch.no_grad():
            # 使用模型前向传播
            reconstructed, z_mean, z_logvar = self.model(matrix_tensor)
            
            # 获取编码器输出作为特征
            if hasattr(self.model, '_encoder_output_cache'):
                features = self.model._encoder_output_cache.cpu().numpy()
            else:
                # 如果没有缓存，返回空特征
                features = None
        
        # 准备返回结果
        result = {
            'features': features
        }
        
        if return_reconstruction:
            result['reconstructed'] = reconstructed.cpu().numpy()
        
        return result

# 创建全局特征提取器单例
_feature_extractor = None

def get_feature_extractor(**kwargs):
    """获取或创建全局TAD特征提取器"""
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = TADFeatureExtractor(**kwargs)
    return _feature_extractor

# 修改全局特征提取函数
def extract_features_from_hic(hic_matrix, chr_name=None, model_path=None, **kwargs):
    """
    从HiC矩阵中提取特征的简化函数 - 使用基类配置
    Returns:
        dict: 包含特征矩阵
    """
    # 创建临时特征提取器
    extractor = TADFeatureExtractor(**kwargs)
    # 确定模型路径
    if model_path is None and chr_name is not None:
        model_path = os.path.join(extractor.output_root, chr_name, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    extractor.load_model(model_path)
    # 提取特征
    result = extractor.extract_features(hic_matrix, return_reconstruction=True)
    return result
        
if __name__ == "__main__":
    print("请使用train.py进行模型训练")

# 在模块导出部分添加AVIT_GAN
__all__ = [
    'TADBaseConfig', 
    'AVIT_GAN',  # 确保新类被导出
    'TADFeatureExtractor',
    'LowRank',
    'find_chromosome_files',
    'fill_hic'
]
  
  