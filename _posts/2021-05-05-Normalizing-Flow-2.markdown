---
layout:     post
title:      "Normalizing Flow - RealNVP 이론 과 구현까지 (2)"
date:       2021-5-05 00:00:00
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


<p>
이제 본격적으로 realNVP 논문의 내용을 살펴보도록 하겠습니다. 이론적인 부분보다 이번 포스팅에서는 간단한 realNVP 개념과 구현사항을 살펴보겠습니다.

realNVP 논문에서는 위의 1-1 invertible mapping function을 neural network로 구성하였으며, affine coupling layer 라고 정의하고 있습니다.

이 논문에서는 이 affine coupling layer가 시작이자 끝이라고 할 수 있습니다. 

<img src="https://github.com/170928/170928.github.io/blob/master/_images/normalizing_flow/figure8.PNG?raw=true">

normalizing flow에서의 중요한 점은 1-1 invertible mapping function을 Jacobian을 계산하기 효율적이면서 complex distribution을 잘 만들어 낼 수 있는 function을 만들어 내는 것입니다. 그리고 그 방법으로 "input 의 일부는 invert 하기 쉬운 방법으로 update되고 나머지는 complex way로 update 한다" 라고 논문에서는 언급합니다. 쉽게 설명하자면, 이전 포스팅에서의 z1 = f(z) 와 같은 과정에서 z가 5차원 이라고 가정하면 1, 2, 3 차원 까지는 쉬운 방법으로, 4, 5차원은 복잡한 방법으로 업데이트 하는 것입니다. 


<img src="https://github.com/170928/170928.github.io/blob/master/_images/normalizing_flow/figure9.PNG?raw=true">

그리고, 이러한 방법의 장점은 normalizing flow의 loss function을 계산하는 과정에서 (이전 포스팅 참조), Jacobian의 계산이 매우 효율적으로 이루어 질 수 있다는 것입니다. 위의 수식은 realNVP에서 말하는 affine coupling layer를 사용하였을 때 loss function이 되는 Jacobian Matrix이며 위의 행렬을 Jacobian을 계산하게 되면 아래와 같아짐을 알 수 있습니다.
$$exp[\sum_{j} s(x_{1:d})_j]$$

<img src="https://github.com/170928/170928.github.io/blob/master/_images/normalizing_flow/figure10.PNG?raw=true">

더해서, invertibility 측면에서 s 와 t를 적용하는 inference 과정과 유사하게 계산할 수 있다는 점이 큰 장점으로 제시하고 있습니다. 하지만 실제로 위의 Jacobian 과정에서 필요하지 않아서 그다지 의미는 없어보이네요 ㅎㅎ 

논문에서는 s 와 t 함수 또한 neural network로 구현하고 있으며, deep convolutional network 를 적용하고 있습니다. 자세한 내용은 구현을 살펴볼 때 보겠습니다. 
</p>

<p>
<img src="https://github.com/170928/170928.github.io/blob/master/_images/normalizing_flow/figure11.PNG?raw=true">

친절하게도 이 논문에서는 위 수식 (7) 번의 실제 구현시에 방법을 설명해주고 있습니다. input dimension의 1:d 와 d+1:D 를 masking 방법을 통해서 구현하였다는 것을 알려줍니다. 


<img src="https://github.com/170928/170928.github.io/blob/master/_images/normalizing_flow/figure12.png?raw=true">

최종적으로 realNVP를 사용한 모델 학습의 경우 위의 목적식을 가지고 학습을 수행하게 됩니다. 
</p>

<p>
이제 본격적으로 pytorch 구현과 함께 살펴보도록 하겠습니다. 이번 포스팅에서는 cifar-10을 기준으로 코드를 작성하였습니다.
</p>

> 참조 : https://github.com/fmu2/realNVP/blob/master/realnvp.py, https://github.com/xqding/RealNVP/blob/master/Real%20NVP%20Tutorial.ipynb


<p> </p>


<p>다음으로 Coupling Layer에 대해서 살펴보겠습니다.</p>

<img src="https://github.com/170928/170928.github.io/blob/master/_images/normalizing_flow/figure13.PNG?raw=true">

<p>RealNVP의 핵심이 되는 Coupling Layer의 경우 수식은 단순하지만, image에 적용할 때 생각보다 복잡합니다. 논문에서는 masking을 활용하는 "Masked convolution"을 사용한다고 서술하고 있으며, 이 방법에 대해서 그림과 함께 설명을 해주고 있지만 time-series 데이터와 강화학습만을 다루던 저에게는 생각보다 쉽게 와닿지 않았습니다.</p>

<p>우선 위의 그림에서 (좌)는 "spatial checkerboard pattern mask", (우)는 "channel-wise masking"으로 논문에서 정의하고 있습니다. 그리고 (좌)에서 (우)로 만드는 과정을 "squeezing operation"이라고 정의합니다. 이때 (좌)가 H x W x C 라면 (우)는 H/2 x W/2 x 4*C 가 되도록 하는 특징이 있습니다. </p>

<p> 이 두가지 masking과 squeezing operation의 적용 순서는 다음과 같습니다.

1. (좌) 의 mask 를 사용하여 Coupling layer forward를 3번 수행 .
2. squeezing operation 수행.
3. (우) 의 mask 를 사용하여 Coupling layer forward를 3번 수행.

이때, (좌)의 spatial checkerboard pattern mask 는 spatial coordinate의 합이 홀수인 경우 "1", 짝수인 경우 "0"으로 설정한 것입니다. 그리고, channel-wise mask는 channel 축으로 앞에서 절반에 "1", 나머지 절반에 "0"을 설정합니다. 
</p>

<p>논문에서 용어를 한번더 바꾸는데 squeezing operation을 수행하는 것을 "multi-scale architecture"에서 사용하기 때문에 "scale" 이라고 정의하고 사용합니다. </p>

<img src="https://github.com/170928/170928.github.io/blob/master/_images/normalizing_flow/figure14.PNG?raw=true">

<p>그리고 이 그림을 이해할 수 있게 설명해줍니다. 위에서 mask를 0과 1로 한 것을 한번 수행할때 마다 서로 바꾸어 줍니다. 1을 0으로 0을 1로! 이를 통해서 1번 coupling layer를 지나면서 x1:d의 경우 변하지 않았던 부분이 다음에는 변하게되도록 해줍니다. 이 그림의 +와 X 칸을 coupling layer라고 생각하시면 되며 시작사는 빈 마름모를 x1:d 와 xd+1:D라고 생각하시면 됩니다. </p>

<p>아래는 RealNVP의 init 부분입니다. 우선 cifar-10 이미지를 사용해서 학습하기 위해서 다음과 같이 이미지의 dimension, channel 수 그리고 중간 t와 s로 사용되는 residual block을 위한 r_dim 을 전달해주어야 합니다. 앞서 말씀드린대로 checkboard mask과 channel-wise mask를 사용해서 각각 3번의 coupling layer를 지나게 되므로 해당 과정을 nn.Modulelist로 한번에 수행할 수 있도록 구성하였습니다. </p>

```python
    class RealNVP(nn.Module):
        def __init__(self, prior, n_dim, n_channel, r_dim):
        super(RealNVP, self).__init__()
        self.prior = prior
        n_dim = n_dim         # image dimension 
        n_channel = n_channel # image channel 수
        base_dim = r_dim      # residual block 변환되었을때의 dimension

        # cifar-10 images는 32 x 32 x 3 
        # n_dim : 32      
        # n_channel : 3     
        # r_dim : 64    

        # SCALE 1: 3 x 32 x 32
        self.s1_ckbd = self.checkerboard(n_channel, r_dim, n_dim)
        self.s1_chan = self.channelwise(n_channel*4, r_dim)
        self.order_matrix_1 = self.order_matrix(n_channel).cuda()
        n_channel *= 2
        n_dim //= 2

        # SCALE 2: 6 x 16 x 16
        self.s2_ckbd = self.checkerboard(n_channel, r_dim, n_dim, final=True)
    
    def checkerboard(self, in_out_dim, mid_dim, size, final=False):
        if final:
            return nn.ModuleList([
                CheckerboardCoupling(in_out_dim, mid_dim, size, 1.),
                CheckerboardCoupling(in_out_dim, mid_dim, size, 0.),
                CheckerboardCoupling(in_out_dim, mid_dim, size, 1.),
                CheckerboardCoupling(in_out_dim, mid_dim, size, 0.)])
        else:
            return nn.ModuleList([
                CheckerboardCoupling(in_out_dim, mid_dim, size, 1.), 
                CheckerboardCoupling(in_out_dim, mid_dim, size, 0.),
                CheckerboardCoupling(in_out_dim, mid_dim, size, 1.)])
        
    def channelwise(self, in_out_dim, mid_dim):
        return nn.ModuleList([
                ChannelwiseCoupling(in_out_dim, mid_dim, 0.),
                ChannelwiseCoupling(in_out_dim, mid_dim, 1.),
                ChannelwiseCoupling(in_out_dim, mid_dim, 0.)])

```
<p> RealNVP의 class init은 위와같이 간단하게 구성이 가능합니다. 그러나, 실제 동작 코드는 조금 더 복잡합니다. </p>

<p>RealNVP의 흐름은 다음과 같습니다. 가장 큰 코드의 흐름은 cifar-10 의 image input x 가 들어오면, log_prob 함수를 호출해서 image x 가 normalizing flow가 추정한 distribution에서의 log-likelihood를 계산하는 것입니다.  </p>

```python
    def forward(self, x):
        """
        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input
        """
        return self.log_prob(x), weight_scale
```

<p> 이 포스트의 수식 설명 마지막 부분에서 log P(x)를 계산하기 위한 수식을 다시한번 살펴 볼 필요가 있습니다. 이떄 수식의 가장 끝에 보이는 Jacobian이 구현에서 가장 귀찮고 어려운 부분이 됩니다.  </p>

```python
    def log_prob(self, x):
        """Computes data log-likelihood.
    
        (See Eq(2) and Eq(3) in the real NVP paper.)
    
        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        z, log_diag_J = self.f(x)
        log_det_J = torch.sum(log_diag_J, dim=(1, 2, 3))
        log_prior_prob = torch.sum(self.prior.log_prob(z), dim=(1, 2, 3))
        return log_prior_prob + log_det_J
```

<p> 우리는 RealNVP class를 처음 정의할 때 3 층으로 구성된 CheckBoardCoupling과 3층으로 구성된 ChannelWiseCoupling을 정의하였었습니다. 이 Layer들은 각 층마다 image input x 의 latent z 와 Jacobian 을 계산합니다. </p>

<p> 이때, 해당 수식처럼 Jacobian들은 계속해서 더해주면 되므로, 아래의 f 함수에서도 "inc"는 계속해서 for문을 돌면서 더해지는 것을 볼 수 있습니다. 반면, z는 다음 layer의 입력으로 다시 들어가는 것을 알 수 있습니다. </p>

```python
    def f(self, x):
        """Transformation f: X -> Z 
        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor and log of diagonal elements of Jacobian.
        """
        z, log_diag_J = x, torch.zeros_like(x)
    
        # SCALE 1: 32(64) x 32(64)
        for i in range(len(self.s1_ckbd)):
            z, inc = self.s1_ckbd[i](z)
            log_diag_J = log_diag_J + inc
    
        z, log_diag_J = self.squeeze(z), self.squeeze(log_diag_J)
        for i in range(len(self.s1_chan)):
            z, inc = self.s1_chan[i](z)
            log_diag_J = log_diag_J + inc
        z, log_diag_J = self.undo_squeeze(z), self.undo_squeeze(log_diag_J)
    
        z, z_off_1 = self.factor_out(z, self.order_matrix_1)
        log_diag_J, log_diag_J_off_1 = self.factor_out(log_diag_J, self.order_matrix_1)
    
        # SCALE 2: 16(32) x 16(32)
        for i in range(len(self.s2_ckbd)):
            z, inc = self.s2_ckbd[i](z)
            log_diag_J = log_diag_J + inc
            
        z = self.restore(z, z_off_1, self.order_matrix_1)
        log_diag_J = self.restore(log_diag_J, log_diag_J_off_1, self.order_matrix_1)
    
        return z, log_diag_J
```
<p> 이 Jacobian을 계산하는 Layer들은 다음 포스트에서 자세히 살펴보겠습니다.  </p>
