import torch
from dataset import *



from collections import OrderedDict
def initialize_memory(obj_list):

    mid = []
    large = []
    patch = []
    for x in obj_list:
        mid.append((x, []))
        large.append((x, []))
        patch.append((x, []))
    mid_memory   = OrderedDict(mid)
    large_memory = OrderedDict(large)
    patch_memory = OrderedDict(patch)
    return mid_memory, large_memory, patch_memory


@torch.no_grad()
def memory(model, obj_list, dataset_dir, save_path, preprocess, transform, k_shot, few_shot_features,
           dataset_name, device, specific_image_paths=None):
    mid_memory, large_memory, patch_memory = initialize_memory(obj_list)

    # 如果指定了特定图片路径，则直接加载这些图片
    if specific_image_paths is not None:
        # 假设所有图片都属于 obj_list[0]（单类别场景）
        cls_name = obj_list[0]
        for img_path in specific_image_paths:
            # 加载图像
            from PIL import Image
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            # 提取特征
            large_scale_tokens, mid_scale_tokens, patch_tokens, class_tokens, large_scale, mid_scale = model.encode_image(
                img_tensor, 16)
            # 存储特征（注意：encode_image 返回的是 batch 维度=1 的 tensor）
            large_memory[cls_name].append(large_scale_tokens.squeeze(0))  # 去掉 batch 维
            mid_memory[cls_name].append(mid_scale_tokens.squeeze(0))
            patch_memory[cls_name].append(patch_tokens.squeeze(0))
        # 拼接所有支撑特征
        for class_name in obj_list:
            large_memory[class_name] = torch.cat(large_memory[class_name])
            mid_memory[class_name] = torch.cat(mid_memory[class_name])
            patch_memory[class_name] = torch.cat(patch_memory[class_name])
        return large_memory, mid_memory, patch_memory

    # 原有逻辑（使用 Dataset 随机抽取）
    for i in range(len(obj_list)):
        if dataset_name == 'mvtec':
            normal_data = MVTecDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                                       aug_rate=-1, mode='train', k_shot=k_shot, save_dir=save_path,
                                       obj_name=obj_list[i])
        elif dataset_name == 'visa':
            normal_data = VisaDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                                      mode='train', k_shot=k_shot, save_dir=save_path, obj_name=obj_list[i])
        normal_dataloader = torch.utils.data.DataLoader(normal_data, batch_size=1, shuffle=False)
        for index, items in enumerate(normal_dataloader):
            images = items['img'].to(device)
            cls_name = items['cls_name']
            # 注意：cls_name 是列表，但 batch_size=1
            large_scale_tokens, mid_scale_tokens, patch_tokens, class_tokens, large_scale, mid_scale = model.encode_image(
                images, 16)
            for class_name, tokens in zip(cls_name, large_scale_tokens):
                large_memory[class_name].append(tokens)
            for class_name, tokens in zip(cls_name, mid_scale_tokens):
                mid_memory[class_name].append(tokens)
            for class_name, tokens in zip(cls_name, patch_tokens):
                patch_memory[class_name].append(tokens)
    # 拼接
    for class_name in obj_list:
        large_memory[class_name] = torch.cat(large_memory[class_name])
        mid_memory[class_name] = torch.cat(mid_memory[class_name])
        patch_memory[class_name] = torch.cat(patch_memory[class_name])
    return large_memory, mid_memory, patch_memory