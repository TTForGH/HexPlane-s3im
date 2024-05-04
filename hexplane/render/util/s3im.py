# from hexplane.render.util.ssim import SSIM
from hexplane.render.util.metric import rgb_ssim
import torch
import numpy as np


# class S3IM(torch.nn.Module):
#     r"""Implements Stochastic Structural SIMilarity(S3IM) algorithm.
#     It is proposed in the ICCV2023 paper  
#     `S3IM: Stochastic Structural SIMilarity and Its Unreasonable Effectiveness for Neural Fields`.

#     Arguments:
#         kernel_size (int): kernel size in ssim's convolution(default: 4)
#         stride (int): stride in ssim's convolution(default: 4)
#         repeat_time (int): repeat time in re-shuffle virtual patch(default: 10)
#         patch_height (height): height of virtual patch(default: 64)
#         patch_width (height): width of virtual patch(default: 64)
#     """
#     def __init__(self, kernel_size=4, stride=4, repeat_time=10, patch_height=64, patch_width=64, max_val=1.0, filter_size=11, filter_sigma=1.5):
#         super(S3IM, self).__init__()
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.repeat_time = repeat_time
#         self.patch_height = patch_height
#         self.patch_width = patch_width
#         self.max_val = max_val
#         self.filter_size = filter_size
#         self.filter_sigma = filter_sigma
#         # self.ssim_loss = SSIM(window_size=self.kernel_size, stride=self.stride)
#         self.ssim_loss = lambda src, tar: rgb_ssim(src, tar, self.max_val, self.filter_size, self.filter_sigma)
        
#     def forward(self, src_vec, tar_vec):
#         loss = 0.0
#         index_list = []
#         for i in range(self.repeat_time):
#             if i == 0:
#                 tmp_index = torch.arange(len(tar_vec))
#                 index_list.append(tmp_index)
#             else:
#                 ran_idx = torch.randperm(len(tar_vec))
#                 index_list.append(ran_idx)
#         res_index = torch.cat(index_list)
#         tar_all = tar_vec[res_index]
#         src_all = src_vec[res_index]
#         # tar_patch = tar_all.permute(1, 0).reshape(1, 3, self.patch_height, self.patch_width * self.repeat_time)
#         # src_patch = src_all.permute(1, 0).reshape(1, 3, self.patch_height, self.patch_width * self.repeat_time)
#         loss = 1 - self.ssim_loss(src_all, tar_all)
#         return loss


# class S3IM(torch.nn.Module):
#     def __init__(self, kernel_size=4, stride=4, repeat_time=10, patch_height=64, patch_width=64, max_val=1.0, filter_size=11, filter_sigma=1.5):
#         super(S3IM, self).__init__()
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.repeat_time = repeat_time
#         self.patch_height = patch_height
#         self.patch_width = patch_width
#         self.max_val = max_val
#         self.filter_size = filter_size
#         self.filter_sigma = filter_sigma

#         # 初始化 rgb_ssim 函数，用于计算 SSIM
#         self.ssim_loss = lambda x, y: rgb_ssim(x, y, max_val=self.max_val, window_size=self.filter_size, filter_sigma=self.filter_sigma, return_map=False)

#     def forward(self, src_vec, tar_vec):
#         """计算损失
#         src_vec: 源向量
#         tar_vec: 目标向量
#         """
#         # 创建索引列表来重新排序和洗牌数据
#         index_list = [torch.arange(len(tar_vec))]  # 初始索引
#         for _ in range(1, self.repeat_time):
#             index_list.append(torch.randperm(len(tar_vec)))
        
#         # 重新排序
#         res_index = torch.cat(index_list)
#         tar_all = tar_vec[res_index]
#         src_all = src_vec[res_index]

#         # 将张量整形为 [N, C, H, W] 形式
#         # tar_patch = tar_all.reshape(-1, 3, self.patch_height, self.patch_width * self.repeat_time)
#         # src_patch = src_all.reshape(-1, 3, self.patch_height, self.patch_width * self.repeat_time)
#         tar_patch = tar_all.permute(1, 0, 2, 3).reshape(1, 3, self.patch_height, self.patch_width * self.repeat_time)
#         src_patch = src_all.permute(1, 0, 2 ,3).reshape(1, 3, self.patch_height, self.patch_width * self.repeat_time)
#         # 计算 S3IM 损失
#         loss = 1 - self.ssim_loss(src_patch, tar_patch)
#         return loss



# class S3IM(torch.nn.Module):
#     def __init__(self, kernel_size=4, stride=4, repeat_time=50, patch_height=64, patch_width=500, max_val=1.0, filter_size=11, filter_sigma=1.5):
#         super(S3IM, self).__init__()
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.repeat_time = repeat_time
#         self.patch_height = patch_height
#         self.patch_width = patch_width
#         self.max_val = max_val
#         self.filter_size = filter_size
#         self.filter_sigma = filter_sigma
#         # 初始化 rgb_ssim 函数，用于计算 SSIM
#         self.ssim_loss = lambda x, y: rgb_ssim(x, y, max_val=self.max_val, filter_size=self.filter_size, filter_sigma=self.filter_sigma, return_map=False)

#     def forward(self, src_vec, tar_vec):
#         index_list = [torch.arange(len(tar_vec))]  # 初始索引
#         for _ in range(1, self.repeat_time):
#             index_list.append(torch.randperm(len(tar_vec)))

#         res_index = torch.cat(index_list)
#         tar_all = tar_vec[res_index]
#         src_all = src_vec[res_index]
        
#         # tar_patch = tar_all.reshape(-1, 3, self.patch_height, self.patch_width * self.repeat_time)
# #         src_patch = src_all.reshape(-1, 3, self.patch_height, self.patch_width * self.repeat_time)
#         tar_patch = tar_all.permute(1, 0, 2, 3).reshape(1, 3, self.patch_height, self.patch_width * self.repeat_time)
#         src_patch = src_all.permute(1, 0, 2 ,3).reshape(1, 3, self.patch_height, self.patch_width * self.repeat_time)

#         # 假设tar_all和src_all已经正确reshape到[N, C, H, W]
#         loss = 1 - self.ssim_loss(tar_patch, src_patch)

#         # 转换为NumPy数组
#         # loss = loss.detach().cpu().numpy()  # 确保在CPU上，并转换为numpy
#         return loss

# 更新 S3IM 类以使用上述的 rgb_ssim 函数
class S3IM:
    def __init__(self, repeat_time=10, patch_height=64, patch_width=64, max_val=1.0, filter_size=11, filter_sigma=1.5):
        self.repeat_time = repeat_time
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.max_val = max_val
        self.filter_size = filter_size
        self.filter_sigma = filter_sigma

    def forward(self, src, tar):
        loss = 0.0
        index_list = []
        for i in range(self.repeat_time):
            if i == 0:
                tmp_index = torch.arange(len(tar_vec))
                index_list.append(tmp_index)
            else:
                ran_idx = torch.randperm(len(tar_vec))
                index_list.append(ran_idx)
        res_index = torch.cat(index_list)
        tar_all = tar_vec[res_index]
        src_all = src_vec[res_index]
        tar_patch = tar_all.permute(1, 0).reshape(1, 3, self.patch_height, self.patch_width * self.repeat_time)
        src_patch = src_all.permute(1, 0).reshape(1, 3, self.patch_height, self.patch_width * self.repeat_time)
        # 假设src和tar已经是合适的numpy数组
        # 计算 S3IM 损失
        loss = 1 - rgb_ssim(tar_patch, src_patch, self.max_val, self.filter_size, self.filter_sigma)
        return loss

