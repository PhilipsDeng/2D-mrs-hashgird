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
import time
torch.backends.cudnn.benchmark = True

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
        transforms.Lambda(lambda img: transforms.functional.crop(img, 0, 0, min(img.size), min(img.size))),  
        transforms.Resize((resolution, resolution)),  # 调整大小到指定分辨率
        transforms.ToTensor(),  # 将图像转换为 [0, 1] 范围内的张量
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).to(device)  # 转换为张量并移动到指定设备
    return image_tensor


# 训练函数
def train(config):
    # 创建模型和优化器
    dtype = getattr(torch, config['dtype'])
    hashgrid = HashGrid(config['hashgrid_config']).to(device).to(dtype)
    renderer = Renderer.hash_renderer(config['renderer_config']).to(device).to(dtype)
    optimizer = optim.AdamW([
        {'params': hashgrid.parameters()},
        {'params': renderer.parameters()}
    ], lr=config['lr'])

    # 创建训练数据
    x, y = torch.arange(config['resolution']) / (config['resolution'] - 1) * 2 - 1, torch.arange(config['resolution']) / (config['resolution'] - 1) * 2 - 1
    x, y = torch.meshgrid(x, y)
    inputs = torch.stack([x, y], dim=-1).to(device).to(dtype)
    target_img = load_target_image('data/001.png', config['resolution'])  
    target = torch.permute(target_img, (1, 2, 0)).reshape(-1, 3).to(device).to(dtype)

    # 训练
    for epoch in range(config['num_epochs']):
        start_time = time.time()
        input = inputs.view(-1, 2).to(device).to(dtype) # -1是自动计算维度大小的意思

        output = hashgrid(input)
        rgb = renderer(output)

        loss = F.mse_loss(rgb, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()

        end_time = time.time()
        elapsed_time = end_time - start_time  
        it_per_sec = 1 / elapsed_time  
        print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Loss: {loss.item()}, Speed: {it_per_sec:.2f} it/s")

        if epoch % 1000 == 0:
            os.makedirs('outputs', exist_ok=True)
            torch.save({
                'hashgrid_state_dict': hashgrid.state_dict(),
                'renderer_state_dict': renderer.state_dict(),}, os.path.join('outputs', 'model.pth'))
            print("#####################Model saved#####################")
            
    # 保存模型
    os.makedirs('outputs', exist_ok=True)
    torch.save({
        'hashgrid_state_dict': hashgrid.state_dict(),
        'renderer_state_dict': renderer.state_dict(),
    }, os.path.join('outputs', f"epoch_{config['num_epochs']}model.pth"))


# 推理
@torch.no_grad()
def dynamic_res_inference(config,res):
    torch.cuda.empty_cache()

    hashgrid = HashGrid(config['hashgrid_config']).to(device)
    renderer = Renderer.hash_renderer(config['renderer_config']).to(device)

    checkpoint = torch.load(os.path.join('outputs', f"epoch_{config['num_epochs']}model.pth"))
    hashgrid.load_state_dict(checkpoint['hashgrid_state_dict'])
    renderer.load_state_dict(checkpoint['renderer_state_dict'])
   
    x, y = torch.arange(res) / (res - 1) * 2 - 1, torch.arange(res) / (res - 1) * 2 - 1
    x, y = torch.meshgrid(x, y)
    img = torch.zeros((res, res, 3), device=device)

    if res <=2048:
        input = torch.stack([x, y], dim=-1).view(-1, 2).to(device).float()
        output = hashgrid(input)
        img = renderer(output).view(res, res, 3)
        img = img.detach().cpu().numpy()

    
    else:
        inputs = torch.stack([x, y], dim=-1).to(device).float()

        for i in tqdm(range(res)):
            input = inputs[:, i, :].float()
            output = hashgrid(input)
            rgb = renderer(output)
            img[:, i, :] = rgb
            torch.cuda.empty_cache()

        img = img.detach().cpu().numpy()


    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'output_image.png')
    plt.imsave(output_path, img)
    print(f"Inference result saved to {output_path}")


# 主函数
def main():
    config_path = 'config/config.yaml'
    config = load_config(config_path)
    train(config)
    print("Starting Infenrence...")
    resolution = int(input('Please input the resolution: '))
    dynamic_res_inference(config,res=resolution)



if __name__ == "__main__":
    main()
