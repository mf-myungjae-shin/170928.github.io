---
layout:     post
title:      "Normalizing Flow - RealNVP 이론 과 구현까지 (2)"
date:       2021-7-19 00:00:00
author:     "MJ Shin"
tags:
    - Machine Learning
    - Normalizing Flow
    - ML
use_math: true
comments: true
sitemap :
  changefreq : daily
  priority : 1.0
---
## Normalizing Flow -> RealNVP Implementation & Theory

<p> 이 포스트부터는 실제로 CouplingLayer들이 어떻게 작동하는지를 살펴보도록 하겠습니다. </p>

<p> 우선, image input x를 가장 처음 변환하는 CheckerBoardCoupling의 코드입니다. </p>

```python

class CheckerboardCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, size, mask_config):
        """Initializes a CheckerboardCoupling.

        Args:
            in_out_dim: number of input and output features.
            mid_dim: number of features in residual blocks.
            size: height/width of features.
            mask_config: mask configuration (see build_mask() for more detail).
            hps: the set of hyperparameters.
        """
        super(CheckerboardCoupling, self).__init__()

        self.coupling = CheckerboardAdditiveCoupling(
                in_out_dim, mid_dim, size, mask_config)

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log of diagonal elements of Jacobian.
        """
        return self.coupling(x, reverse)

class CheckerboardAdditiveCoupling(AbstractCoupling):
    def __init__(self, in_out_dim, mid_dim, size, mask_config):
        """Initializes a CheckerboardAdditiveCoupling.
        Args:
            in_out_dim: number of input and output features.
            mid_dim: number of features in residual blocks.
            size: height/width of features.
            mask_config: mask configuration (see build_mask() for more detail).
            hps: the set of hyperparameters.
        """
        super().__init__(mask_config)
        
        self.mask = self.build_mask(size, config=mask_config).cuda()
        self.in_bn = nn.BatchNorm2d(in_out_dim)
        self.block = nn.Sequential(
            nn.ReLU(),
            ResidualModule(2*in_out_dim+1, mid_dim, in_out_dim, 
                self.res_blocks, self.bottleneck, self.skip, self.weight_norm))
        self.out_bn = nn.BatchNorm2d(in_out_dim, affine=False)

    def forward(self, x, reverse=False):
        """Forward pass.
        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log of diagonal elements of Jacobian.
        """
        [B, _, _, _] = list(x.size())
        mask = self.mask.repeat(B, 1, 1, 1)
        x_ = self.in_bn(x * mask)
        x_ = torch.cat((x_, -x_), dim=1)
        x_ = torch.cat((x_, mask), dim=1)     # 2C+1 channels
        shift = self.block(x_) * (1. - mask)

        log_diag_J = torch.zeros_like(x)     # unit Jacobian determinant
        # See Eq(3) and Eq(4) in NICE and Section 3.7 in real NVP
        if reverse:
            if self.coupling_bn:
                mean, var = self.out_bn.running_mean, self.out_bn.running_var
                mean = mean.reshape(-1, 1, 1, 1).transpose(0, 1)
                var = var.reshape(-1, 1, 1, 1).transpose(0, 1)
                x = x * torch.exp(0.5 * torch.log(var + 1e-5) * (1. - mask)) \
                    + mean * (1. - mask)
            x = x - shift
        else:
            x = x + shift
            if self.coupling_bn:
                if self.training:
                    _, var = self.batch_stat(x)
                else:
                    var = self.out_bn.running_var
                    var = var.reshape(-1, 1, 1, 1).transpose(0, 1)
                x = self.out_bn(x) * (1. - mask) + x * mask
                log_diag_J = log_diag_J - 0.5 * torch.log(var + 1e-5) * (1. - mask)
        return x, log_diag_J
```
