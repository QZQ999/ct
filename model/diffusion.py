import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

# 1. 定义 Autoencoder（用于将图像压缩到潜在空间）
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(Autoencoder, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # [B, 1, 28, 28] -> [B, 32, 14, 14]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # [B, 32, 14, 14] -> [B, 64, 7, 7]
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, kernel_size=7, stride=1, padding=0),  # [B, 64, 7, 7] -> [B, latent_dim, 1, 1]
            nn.ReLU()
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=7, stride=1, padding=0),  # [B, latent_dim, 1, 1] -> [B, 64, 7, 7]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [B, 64, 7, 7] -> [B, 32, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # [B, 32, 14, 14] -> [B, 1, 28, 28]
            nn.Sigmoid()  # 输出范围为 [0, 1]
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon

# 2. 定义 Diffusion Model（在潜在空间上训练）
class DiffusionModel(nn.Module):
    def __init__(self, latent_dim=64, timesteps=1000):
        super(DiffusionModel, self).__init__()
        self.timesteps = timesteps
        
        # 定义噪声调度（noise schedule）
        self.betas = torch.linspace(1e-4, 0.02, timesteps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
        # 定义去噪网络（UNet）
        self.denoise_net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    
    def forward(self, z, t):
        # 预测噪声
        return self.denoise_net(z)
    
    def diffuse(self, z, t):
        # 添加噪声
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t])
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[t])
        noise = torch.randn_like(z)
        z_noisy = sqrt_alpha_bar * z + sqrt_one_minus_alpha_bar * noise
        return z_noisy, noise
    
    def train_step(self, z):
        # 随机选择时间步
        t = torch.randint(0, self.timesteps, (z.shape[0],))
        
        # 添加噪声
        z_noisy, noise = self.diffuse(z, t)
        
        # 预测噪声
        predicted_noise = self.forward(z_noisy, t)
        
        # 计算损失
        loss = F.mse_loss(predicted_noise, noise)
        return loss
    
    def sample(self, num_samples=1):
        # 从噪声中生成样本
        z = torch.randn(num_samples, 64)  # 假设潜在维度为 64
        for t in reversed(range(self.timesteps)):
            z = self.denoise_net(z)  # 去噪
        return z

# 3. 训练 Autoencoder
autoencoder = Autoencoder(latent_dim=64)
optimizer_ae = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)

# 加载 MNIST 数据集
transform = transforms.Compose([transforms.ToTensor()])
dataset = MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# 训练 Autoencoder
for epoch in range(5):  # 假设训练 5 个 epoch
    for x, _ in dataloader:
        optimizer_ae.zero_grad()
        x_recon = autoencoder(x)
        loss = F.mse_loss(x_recon, x)
        loss.backward()
        optimizer_ae.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# 4. 训练 Diffusion Model
diffusion_model = DiffusionModel(latent_dim=64)
optimizer_diff = torch.optim.Adam(diffusion_model.parameters(), lr=1e-4)

# 使用训练好的 Autoencoder 对数据集进行编码
latent_dataset = []
for x, _ in dataloader:
    with torch.no_grad():
        z = autoencoder.encode(x)
        latent_dataset.append(z)
latent_dataset = torch.cat(latent_dataset, dim=0)

# 训练 Diffusion Model
for epoch in range(10):  # 假设训练 10 个 epoch
    for z in latent_dataset:
        optimizer_diff.zero_grad()
        loss = diffusion_model.train_step(z)
        loss.backward()
        optimizer_diff.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# 5. 生成样本
with torch.no_grad():
    z_samples = diffusion_model.sample(num_samples=16)
    x_samples = autoencoder.decode(z_samples)

# 可视化生成的样本
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_samples[i].squeeze(), cmap="gray")
    ax.axis("off")
plt.show()