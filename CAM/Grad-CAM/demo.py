# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2023/4/20 11:06
# @Author : liumin
# @File : demo.py

import math
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Optional, List
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
from torch import Tensor
from matplotlib import cm
from torchvision.transforms.functional import to_pil_image

img_path = '/home/dell/img/1.JPEG'     # 输入图片的路径
save_path = '/home/dell/cam/CAM1.png'    # 类激活图保存路径

preprocess = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

net = models.vgg11_bn(pretrained=True).cuda()   # 导入模型
# net = models.mobilenet_v2(pretrained=True).cuda()   # 导入模型
print(net)


feature_map = []     # 建立列表容器，用于盛放输出特征图
def forward_hook(module, inp, outp):     # 定义hook
    feature_map.append(outp)    # 把输出装入字典feature_map
net.features.register_forward_hook(forward_hook)    # 对net.layer4这一层注册前向传播


grad = []     # 建立列表容器，用于盛放特征图的梯度
def backward_hook(module, inp, outp):    # 定义hook
    grad.append(outp)    # 把输出装入列表grad
net.features.register_full_backward_hook(backward_hook)    # 对net.features这一层注册反向传播


orign_img = Image.open(img_path).convert('RGB')    # 打开图片并转换为RGB模型
img = preprocess(orign_img)     # 图片预处理
img = torch.unsqueeze(img, 0)     # 增加batch维度 [1, 3, 224, 224]

out = net(img.cuda())     # 前向传播
cls_idx = torch.argmax(out).item()    # 获取预测类别编码
score = out[:, cls_idx].sum()    # 获取预测类别分数
net.zero_grad()
score.backward(retain_graph=True)    # 由预测类别分数反向传播


weights = grad[0][0].squeeze(0).mean(dim=(1, 2))    # 获得权重


grad_cam = (weights.view(*weights.shape, 1, 1) * feature_map[0].squeeze(0)).sum(0)


def _normalize(cams: Tensor) -> Tensor:
    """CAM normalization"""
    cams.sub_(cams.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1))
    cams.div_(cams.flatten(start_dim=-2).max(-1).values.unsqueeze(-1).unsqueeze(-1))

    return cams

grad_cam = _normalize(F.relu(grad_cam, inplace=True)).cpu()
mask = to_pil_image(grad_cam.detach().numpy(), mode='F')

def overlay_mask(img: Image.Image, mask: Image.Image, colormap: str = 'jet', alpha: float = 0.6) -> Image.Image:
    """Overlay a colormapped mask on a background image

    Args:
        img: background image
        mask: mask to be overlayed in grayscale
        colormap: colormap to be applied on the mask
        alpha: transparency of the background image

    Returns:
        overlayed image
    """

    if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError('img and mask arguments need to be PIL.Image')

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError('alpha argument is expected to be of type float between 0 and 1')

    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Image.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, 1:]).astype(np.uint8)
    # Overlay the image with the mask
    overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))

    return overlayed_img

result = overlay_mask(orign_img, mask)

result.show()

result.save(save_path)