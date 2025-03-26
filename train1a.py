import torch
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from net1a import (
    TADBaseConfig,
    AVIT_GAN,
    LowRank,
    find_chromosome_files,
    fill_hic
)

class TADPipelineGAN(TADBaseConfig):
    """对抗训练流程（兼容原有数据加载）"""
    
    def __init__(self, resume_training=False, **kwargs):
        super().__init__(**kwargs)
        self.resume_training = resume_training
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Initialize components and checkpoint configurations"""
        self.adj_graph = None
        self.normalized_matrix = None
        self.avit_dino = self._create_model()  # 初始化模型实例
        torch.backends.cuda.checkpoint_activations = False
        torch.backends.cuda.checkpoint_layers_n = 10

    def load_data(self):
        """Data loading process: Load Hi-C data and return contact matrix"""
        hic_paths = find_chromosome_files(self.output_root)
        if not hic_paths:
            raise FileNotFoundError("Hi-C data files not found")
        hic_path = hic_paths[0]
        chr_dir = os.path.dirname(hic_path)
        self.chr_name = os.path.basename(chr_dir)
        print(f"[DEBUG] Current chromosome: {self.chr_name}")
        matrix = fill_hic(hic_path, self.resolution)
        print(f"[DEBUG] matrix_sum = {np.sum(matrix)}")
        return matrix

    def _create_model(self, patch_size=None):
        model_params = self.get_model_params()
        # 过滤掉AVIT_GAN不需要的参数
        valid_params = ['embed_dim', 'patch_size', 'num_layers', 'num_heads',
                      'use_amp', 'mask_ratio', 'boundary_weight']
        filtered_params = {k:v for k,v in model_params.items() if k in valid_params}
        model = AVIT_GAN(**filtered_params).to(self.device)
        model.init_optimizer()
        return model

    def train_model(self, cur_tensor):
        """修改后的对抗训练循环"""
        print("=== Starting Adversarial Training ===")
        
        # 确保模型已初始化
        if self.avit_dino is None:
            self.avit_dino = self._create_model()
        
        # 生成伪标签（兼容原有数据格式）
        with torch.no_grad():
            real_labels = cur_tensor.clone()
            real_labels[real_labels > 0] = 1  # 二值化标签

        # 训练循环
        with tqdm(range(self.num_epochs), desc="Adversarial Training") as pbar:
            for epoch in pbar:
                losses = self.avit_dino.train_epoch(cur_tensor, real_labels)
                pbar.set_postfix({
                    'total': f"{losses['total']:.4f}",
                    'gen': f"{losses['generator']:.4f}",
                    'disc': f"{losses['discriminator']:.4f}",
                    'recon': f"{losses['recon']:.4f}"
                })
                
                # 可视化与模型保存逻辑保持兼容
                if epoch % 5 == 0:
                    self._visualize_boundary_predictions(cur_tensor, epoch)

    def _visualize_boundary_predictions(self, matrix, epoch):
        """修改可视化函数以适配新输出"""
        try:
            chr_output_dir = os.path.join(self.output_root, self.chr_name)
            os.makedirs(chr_output_dir, exist_ok=True)
            
            with torch.no_grad():
                outputs = self.avit_dino(matrix)
                segmentation = outputs['segmentation'].cpu().numpy()
                
                if segmentation.ndim == 4:  # [B, C, H, W]
                    seg_np = segmentation[0, 0]  # 取第一个批次和通道
                else:
                    seg_np = segmentation
                
                # 可视化分割结果
                plt.figure(figsize=(10, 10))
                plt.imshow(seg_np, cmap='viridis')
                plt.colorbar()
                plt.title(f"Epoch {epoch}: GAN Segmentation")
                plt.tight_layout()
                plt.savefig(os.path.join(chr_output_dir, f"gan_seg_epoch{epoch}.png"))
                plt.close()
                
                # 尝试可视化真假判别结果
                if 'd_real' in outputs and 'd_fake' in outputs:
                    d_real = outputs['d_real'].cpu().numpy()
                    d_fake = outputs['d_fake'].cpu().numpy()
                    
                    plt.figure(figsize=(12, 5))
                    if d_real.ndim > 1:
                        d_real = d_real.mean(axis=0)
                    if d_fake.ndim > 1:
                        d_fake = d_fake.mean(axis=0)
                    
                    plt.plot(d_real, label='Real Score')
                    plt.plot(d_fake, label='Fake Score')
                    plt.legend()
                    plt.title(f"Epoch {epoch}: Discriminator Scores")
                    plt.ylabel("Score")
                    plt.xlabel("Position")
                    plt.tight_layout()
                    plt.savefig(os.path.join(chr_output_dir, f"gan_disc_epoch{epoch}.png"))
                    plt.close()
        except Exception as e:
            print(f"可视化过程发生错误: {str(e)}")

    def run(self):
        """Main process execution: Data loading -> CUR decomposition -> Model training and parameter saving"""
        matrix = self.load_data()
        projector = LowRank(p=0.7, alpha=0.7)
        cur_matrix = projector.fit_transform(matrix)
        print(f"[DEBUG] CUR matrix shape: {cur_matrix.shape}, sparsity: {np.mean(cur_matrix < 1e-6):.2%}")
        del matrix
        
        print(f"[DEBUG] CUR matrix shape: {cur_matrix.shape}, sparsity: {np.mean(cur_matrix < 1e-6):.2%}")
        
        cur_tensor = torch.from_numpy(cur_matrix).float().to(self.device)
        del cur_matrix
        
        self.train_model(cur_tensor)

def main():
    pipeline = TADPipelineGAN(resume_training=True)
    pipeline.run()
    print("Model training and parameter saving process completed")

if __name__ == "__main__":
    main()