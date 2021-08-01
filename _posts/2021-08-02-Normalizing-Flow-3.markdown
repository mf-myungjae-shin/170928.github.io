---
layout:     post
title:      "Normalizing Flow - RealNVP 이론 과 구현까지 (3)"
date:       2021-08-02 00:00:00
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

<p> 먼저, image input x를 가장 처음 변환하는 CheckerBoardCoupling의 코드입니다. </p>

<p> 우선 앞선 코드에서, in_out_dim은 n_channel (3)이 할당됩니다. 이유는 32 x 32 x 3 의 image를 입력으로 받고 있기 때문입니다. 그리고 mask_config의 경우 normalizing-flow 2번 포스팅에서 마름모 모양의 sequence flow 그림처럼 데이터가 그대로 전달되거나 혹은 연산에 참여하는 것이 매 layer마다 섞이도록 하는 변수입니다. </p>

```python

class CheckerboardCoupling(nn.Module):
    def __init__(self, in_out_dim, size, mask_config):
        super().__init__()

        self.coupling = CheckerboardAdditiveCoupling(
                in_out_dim, size, mask_config)

    def forward(self, x, reverse=False):
        return self.coupling(x, reverse)
```

<p> batch x 1 (channel) x size x size 의 mask matrix를 만들때 mask_config에 따라 홀수 위치 혹은 짝수 위치의 값이 1 이되고 나머지는 0이 되도록 하는 matrix를 생성할 때 mod와 numpy arange 를 적절히 사용해서 만든 코드입니다. </p>

```python
def build_mask(self, size, config=1.):
        mask = np.arange(size).reshape(-1, 1) + np.arange(size)
        mask = np.mod(config + mask, 2)
        mask = mask.reshape(-1, 1, size, size)
        return torch.tensor(mask.astype('float32'))
```

<p> 위의 방법으로 생성된 마스크를 input x 와 곱해준 후 논문의 수식 s와 t에 대한 연산을 수행합니다. 그냥 수식 그대로 구현해주면 되는 간단한 부분이었습니다. 실제로 논문에서의 성능을 복구하기 위해서는 t와 s layer가 resnet을 사용하였으나 포스팅을 위해서 간단한 sequential conv2d를 사용하였습니다. </p>

```python
class CheckerboardAdditiveCoupling(AbstractCoupling):
    def __init__(self, in_out_dim, size, mask_config):
        super().__init__()
        
        self.mask = self.build_mask(size, config=mask_config).cuda()
        self.t = nn.Sequential(
                            nn.Conv2d(in_channels=in_out_dim, out_channels=in_out_dim, kernel_size=3, padding=1), 
                            nn.ReLU(), 
                            nn.Conv2d(in_channels=in_out_dim, out_channels=in_out_dim, kernel_size=3, padding=1),
                            nn.ReLU(), 
                            nn.Conv2d(in_channels=in_out_dim, out_channels=in_out_dim, kernel_size=3, padding=1)
                            )
        self.s = nn.Sequential(
                            nn.Conv2d(in_channels=in_out_dim, out_channels=in_out_dim, kernel_size=3, padding=1), 
                            nn.ReLU(), 
                            nn.Conv2d(in_channels=in_out_dim, out_channels=in_out_dim, kernel_size=3, padding=1),
                            nn.ReLU(), 
                            nn.Conv2d(in_channels=in_out_dim, out_channels=in_out_dim, kernel_size=3, padding=1)
                            )
    def forward(self, x, reverse=False):
        # x : 64 x 3 x 32 x 32
        # mask : 64 x 1 x 32 x 32
        [B, _, _, _] = list(x.size())
        mask = self.mask.repeat(B, 1, 1, 1)

        log_diag_J = torch.zeros_like(x) 
        if reverse:
            x_ = x * mask 
            s = self.s(x_) * (1. - mask)
            t = self.t(x_) * (1. - mask)
            x = x_ + (1. - self.mask) * (x * torch.exp(s) + t)
            return x, log_diag_J
        else:
            z = x
            _z = z * mask 
            s = self.s(_z) * (1. - mask)
            t = self.t(_z) * (1. - mask)
            z = (1. - self.mask) * (z - t) * torch.exp(-s) + _z
            log_diag_J -= torch.exp(self.s(_z))
            return z, log_diag_J
```

<p> 다음으로는 channel 축으로 masking을 수행하는 masked-convolution coupling layer 코드 부분입니다. </p>

```python
class ChannelwiseCoupling(nn.Module):
    def __init__(self, in_out_dim, size, mask_config):
        super().__init__()

        self.coupling = ChannelwiseAdditiveCoupling(
            in_out_dim, size, mask_config)

    def forward(self, x, reverse=False):
        return self.coupling(x, reverse)

def build_channel_mask(self, channel, size, config=1.):
        f = np.zeros(shape=(1, int(channel/2), size, size))
        g = np.ones(shape=(1, int(channel/2), size, size))
        if config:
            mask = np.concatenate([g, f], axis=1) 
        else:
            mask = np.concatenate([f, g], axis=1) 
        return torch.tensor(mask.astype('float32'))
```

<p> mask가 channel 축으로 input channel의 절반은 on, 나머지 절반은 off 형태로 channel wise masking을 수행하기 때문에 mask 생성 방식이 위에서 보았던 CheckerBoard를 위한 mask 생성 방법과는 차이가 있습니다.</p>

<p> 이 mask의 차이만 있을 뿐 나머지 부분에서의 차이는 없다고 생각하시면 될 것 같습니다. </p>

```python
class ChannelwiseAdditiveCoupling(AbstractCoupling):
    def __init__(self, in_out_dim, size, mask_config):
        super().__init__()
        self.t = nn.Sequential(
                            nn.Conv2d(in_channels=in_out_dim, out_channels=in_out_dim, kernel_size=3, padding=1), 
                            nn.ReLU(), 
                            nn.Conv2d(in_channels=in_out_dim, out_channels=in_out_dim, kernel_size=3, padding=1),
                            nn.ReLU(), 
                            nn.Conv2d(in_channels=in_out_dim, out_channels=in_out_dim, kernel_size=3, padding=1)
                            )
        self.s = nn.Sequential(
                            nn.Conv2d(in_channels=in_out_dim, out_channels=in_out_dim, kernel_size=3, padding=1), 
                            nn.ReLU(), 
                            nn.Conv2d(in_channels=in_out_dim, out_channels=in_out_dim, kernel_size=3, padding=1),
                            nn.ReLU(), 
                            nn.Conv2d(in_channels=in_out_dim, out_channels=in_out_dim, kernel_size=3, padding=1)
                            )
        self.mask = self.build_channel_mask(in_out_dim, size, mask_config).cuda()

    def forward(self, x, reverse=False):
        [B, _, _, _] = list(x.size())
        mask = self.mask.repeat(B, 1, 1, 1)

        log_diag_J = torch.zeros_like(x)    # unit Jacobian determinant
        if reverse:
            x_ = x * mask 
            s = self.s(x_) * (1. - mask)
            t = self.t(x_) * (1. - mask)
            x = x_ + (1. - mask) * (x * torch.exp(s) + t)
        else:
            z = x
            _z = z * mask 
            s = self.s(_z) * (1. - mask)
            t = self.t(_z) * (1. - mask)
            z = (1. - mask) * (z - t) * torch.exp(-s) + _z
            log_diag_J -= torch.exp(self.s(_z))
        return x, log_diag_J
```

<p> 이렇게 layer들을 구성한 후 학습 코드는 일반적인 neural network 학습 방법을 따르면 됩니다. </p>

```python
while epoch < args.max_epoch:
        print('Epoch %d:' % epoch)
        epoch += 1
        flow.train()
        for batch_idx, data in enumerate(train_loader, 1):
            optimizer.zero_grad()
            x, _ = data
            x = x.to(device)

            log_ll = flow(x).mean()

            loss = -log_ll
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"loss : {loss}")

        flow.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader, 1):
                x, _ = data
                x = x.to(device)

                log_ll = flow(x).mean()
                loss = -log_ll

            samples = flow.sample(args.sample_size)
            utils.save_image(utils.make_grid(samples),
                './samples/' + dataset + '/' + filename + '_ep%d.png' % epoch)
```

## 정리 

<p> 지금까지 3개의 포스팅 동안 normalizing flow 의 realNVP 를 cifar-10에서 적용하는 코드와 간단하게 이론에 대해서 정리해보았습니다. </p>

<p> 포스팅에서는 저처럼 완전 처음으로 접하는 경우 코드를 어떤식으로 구성하는게 좋은지에 대해서 생각해 보실 수 있도록 간단하게 구현을 하였기 때문에 논문과 같은 결과를 reproduce할 수는 없는 코드이지만 흐름을 이해하는데 도움이 될 수 있다고 생각합니다 ㅎㅎ </p>

<p> 혹시 위의 코드에서 논문과 동일하게 구현을 원하시는 분은 다음과 같은 부분을 신경쓰시면 될 것 같습니다. </p>

1. CheckerBoard와 ChannelWise 두 개의 CouplingLayer에서 input의 Order를 바꾸는 작업
2. t와 s로 정의한 neural network를 resnet과 같은 모델을 사용
3. det_J 계산

## 마무리
<p> 이런 저런 일들로 포스팅이 너무 오랫동안 걸려서 완성이 되었네요. 그래도 첫 포스팅으로 구현까지 하고 간단하게 코드를 바꾸는 작업등을 하면서 저도 많이 배우는 계기가 되었던 것 같습니다. 평소 computer vision쪽을 다루지 않아 혼자 공부하다보니 틀리는 내용이나 구현상에서도 오류가 많을 수 있습니다 ㅠㅠ 언제든지 메일로 의견주세요 ! </p>

<p> 감사합니다 :) </p>