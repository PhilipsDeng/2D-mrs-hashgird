import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import yaml
from tqdm import tqdm
from modules.hash_modules import HashGrid
from modules.hash_render import Renderer
import matplotlib.pyplot as plt
import os
from PIL import Image
import torchvision.transforms as transforms


# 加载 config.yaml 文件
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


# 指定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 加载目标图像并转换为张量
def load_target_image(image_path, resolution):
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),  # 将图像转换为 [0, 1] 范围内的张量
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).to(device)  # 转换为张量并移动到指定设备
    return image_tensor


# 训练函数
def train(config):
    # 创建模型和优化器
    hashgrid = HashGrid(config['hashgrid_config']).to(device)
    renderer = Renderer.hash_render(config['renderer_config']).to(device)
    optimizer = optim.Adam([
        {'params': hashgrid.parameters()},
        {'params': renderer.parameters()}
    ], lr=config['lr'])

    # 创建训练数据
    x, y = torch.arange(config['resolution']) / (config['resolution'] - 1) * 2 - 1, torch.arange(config['resolution']) / (config['resolution'] - 1) * 2 - 1
    x, y = torch.meshgrid(x, y)
    inputs = torch.stack([x, y], dim=-1).to(device).float()
    target_img = load_target_image('data/001.png', config['resolution'])  # 加载目标图像

    # 训练循环
    for epoch in range(config['num_epochs']):
        for i in tqdm(range(config['resolution'])):
            input = inputs[:, i, :].float()
            target = target_img[:, i, :]
            target = torch.permute(target, (1, 0))

            # 前向传播
            output = hashgrid(input)
            rgb = renderer(output)

            # 计算损失
            loss = F.mse_loss(rgb, target)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Loss: {loss.item()}")

    # 保存模型
    os.makedirs('outputs', exist_ok=True)
    torch.save({
        'hashgrid_state_dict': hashgrid.state_dict(),
        'renderer_state_dict': renderer.state_dict(),
    }, os.path.join('outputs', 'model.pth'))


# 推理函数
@torch.no_grad()
def inference(config):
    hashgrid = HashGrid(config['hashgrid_config']).to(device)
    renderer = Renderer.hash_render(config['renderer_config']).to(device)

    checkpoint = torch.load(os.path.join('outputs', 'model.pth'))
    hashgrid.load_state_dict(checkpoint['hashgrid_state_dict'])
    renderer.load_state_dict(checkpoint['renderer_state_dict'])

    x, y = torch.arange(config['resolution']) / (config['resolution'] - 1) * 2 - 1, torch.arange(config['resolution']) / (config['resolution'] - 1) * 2 - 1
    x, y = torch.meshgrid(x, y)
    inputs = torch.stack([x, y], dim=-1).to(device).float()
    img = torch.zeros((config['resolution'], config['resolution'], 3), device=device)

    for i in tqdm(range(config['resolution'])):
        input = inputs[:, i, :].float()
        output = hashgrid(input)
        rgb = renderer(output)

        img[:, i, :] = rgb

    # 将 img 从 GPU 移动到 CPU，并转换为 NumPy 数组
    img = img.detach().cpu().numpy()
    img = np.flip(img, axis=0)  # 翻转图像以匹配原始图像的方向
    img = np.rot90(img, k=-1)  # 旋转图像以匹配原始图像的方向

    # 创建输出目录
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    # 保存图像到输出目录
    output_path = os.path.join(output_dir, 'output_image.png')
    plt.imsave(output_path, img)
    print(f"Inference result saved to {output_path}")


# 主函数
def main():
    config_path = 'config/config.yaml'
    config = load_config(config_path)
    train(config)
    print("Starting Infenrence...")
    inference(config)



if __name__ == "__main__":
    main()
