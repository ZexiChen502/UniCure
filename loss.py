import torch.nn as nn
import torch
from geomloss import SamplesLoss


class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, normalize=False, **kwargs):
        """
        Args:
            kernel_type: 'rbf' or 'linear'
            kernel_mul: 多核带宽的乘数因子
            kernel_num: 核的数量
            fix_sigma: 是否使用固定的标准差 (如果为None则使用中位数/均值启发式)
            normalize: [新增] 是否在计算前对输入进行正态标准化
        """
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        # 修正原代码中的bug: 之前直接 self.fix_sigma = None 会忽略传入的参数
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type
        self.normalize = normalize  # 新增标志位

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        # 这里的unsqueeze和expand是为了生成距离矩阵
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))

        # 计算欧氏距离平方
        L2_distance = ((total0 - total1) ** 2).sum(2)

        if fix_sigma:
            bandwidth = fix_sigma
        else:
            # 如果没有指定sigma，使用均值启发式策略
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)

        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]

        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = torch.matmul(delta, delta.T)
        return loss

    def forward(self, source, target):

        if self.normalize:

            total = torch.cat([source, target], dim=0)

            mean = total.mean(dim=0, keepdim=True)
            std = total.std(dim=0, keepdim=True) + 1e-6  # 加一个小数值防止除以0

            # 3. 应用标准化
            source = (source - mean) / std
            target = (target - mean) / std

        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)

            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)

            return loss









