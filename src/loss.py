import torch.nn as nn
import torch


class MMDLoss(nn.Module):
    def __init__(self, kernel_type='linear', kernel_mul=2.0, kernel_num=1, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
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


class WeightedMSELoss(nn.Module):
    def __init__(self, epsilon=1e-3, penalty_alpha=0.0):
        super().__init__()
        self.epsilon = epsilon
        self.penalty_alpha = penalty_alpha

    def forward(self, outputs, targets, control):
        # 计算基因特异性权重
        gene_diff = torch.abs(targets - control)
        weights = (gene_diff / (gene_diff.mean(dim=1, keepdim=True) + self.epsilon)).detach()

        # 加权MSE
        mse_loss = torch.mean(weights * (outputs - targets) ** 2)

        # 可选：基因相关性惩罚项
        output_corr = torch.corrcoef(outputs.T)
        target_corr = torch.corrcoef(targets.T)
        output_corr = torch.nan_to_num(output_corr, nan=0.0)
        target_corr = torch.nan_to_num(target_corr, nan=0.0)
        corr_penalty = torch.mean((output_corr - target_corr) ** 2)

        return mse_loss + self.penalty_alpha * corr_penalty

