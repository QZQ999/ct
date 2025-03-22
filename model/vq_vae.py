import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # Codebook: 存储离散嵌入向量的查找表
        self.codebook = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.codebook.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
        
    def forward(self, inputs):
        # 将输入展平为 [batch_size * height * width, embedding_dim]
        flat_inputs = inputs.view(-1, self.embedding_dim)
        
        # 计算输入与 codebook 中嵌入向量之间的距离
        distances = (torch.sum(flat_inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self.codebook.weight**2, dim=1)
                    - 2 * torch.matmul(flat_inputs, self.codebook.weight.t()))
        
        # 找到最近的嵌入向量（量化）
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.codebook(encoding_indices).view(inputs.shape)
        
        # 梯度停止：将量化后的值与输入分离
        quantized_stopped = inputs + (quantized - inputs).detach()
        
        # 计算损失
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)  # 编码器损失
        q_latent_loss = F.mse_loss(quantized, inputs.detach())  # 量化损失
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # 计算困惑度（perplexity）
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return loss, quantized_stopped, perplexity, encoding_indices

class VQVAE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VQVAE, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # [B, 1, H, W] -> [B, 32, H/2, W/2]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # [B, 32, H/2, W/2] -> [B, 64, H/4, W/4]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # [B, 64, H/4, W/4] -> [B, 128, H/8, W/8]
            nn.ReLU(),
            nn.Conv2d(128, embedding_dim, kernel_size=4, stride=2, padding=1),  # [B, 128, H/8, W/8] -> [B, embedding_dim, H/16, W/16]
            nn.ReLU()
        )
        
        # 量化层
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 128, kernel_size=4, stride=2, padding=1),  # [B, embedding_dim, H/16, W/16] -> [B, 128, H/8, W/8]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 128, H/8, W/8] -> [B, 64, H/4, W/4]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [B, 64, H/4, W/4] -> [B, 32, H/2, W/2]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # [B, 32, H/2, W/2] -> [B, 1, H, W]
            nn.Sigmoid()  # 输出范围为 [0, 1]
        )
        
    def forward(self, x):
        # 编码
        z = self.encoder(x)
        
        # 量化
        vq_loss, quantized, perplexity, _ = self.vq_layer(z)
        
        # 解码
        x_recon = self.decoder(quantized)
        
        # 总损失
        recon_loss = F.mse_loss(x_recon, x)  # 重构损失
        total_loss = recon_loss + vq_loss
        
        return total_loss, x_recon, perplexity

# 超参数
num_embeddings = 128  # 嵌入向量的数量
embedding_dim = 64    # 嵌入向量的维度
commitment_cost = 0.25  # 量化损失的权重

# 模型
model = VQVAE(num_embeddings, embedding_dim, commitment_cost)

# 示例输入
input_image = torch.randn(1, 1, 64, 64)  # [B, C, H, W]
total_loss, reconstructed_image, perplexity = model(input_image)

print(f"Total Loss: {total_loss.item()}, Perplexity: {perplexity.item()}")