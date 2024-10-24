import torch
import torch.nn as nn


class IMUNormalizer(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(IMUNormalizer, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        """
        Computes statistics (mean and stdev) for two groups of features:
        - The first group consists of the first 3 features.
        - The second group consists of the next 3 features.
        """
        assert self.num_features >= 6, "num_features must be at least 6 to divide into two groups"
        
        # 选择从时间维度来计算均值和标准差
        dim2reduce = tuple(range(1, x.ndim - 1))  
        # acc
        self.mean_group1 = torch.mean(x[:, :, :3], dim=dim2reduce, keepdim=True).detach()
        self.stdev_group1 = torch.sqrt(torch.var(x[:, :, :3], dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
        # gyro
        self.mean_group2 = torch.mean(x[:, :, 3:], dim=dim2reduce, keepdim=True).detach()
        self.stdev_group2 = torch.sqrt(torch.var(x[:, :, 3:], dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x

        # Normalize the first group of features
        x[:, :, :3] = (x[:, :, :3] - self.mean_group1) / self.stdev_group1

        # Normalize the second group of features
        x[:, :, 3:] = (x[:, :, 3:] - self.mean_group2) / self.stdev_group2

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x

        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)

        # Denormalize the first group of features
        x[:, :, :3] = x[:, :, :3] * self.stdev_group1 + self.mean_group1

        # Denormalize the second group of features
        x[:, :, 3:] = x[:, :, 3:] * self.stdev_group2 + self.mean_group2

        return x

    