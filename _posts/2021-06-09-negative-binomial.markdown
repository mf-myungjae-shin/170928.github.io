---
layout:     post
title:      "Negative Binomial Distribution (scipy.stats)"
date:       2021-6-09 00:00:00
author:     "MJ Shin"
tags:
    - Scipy
    - Distribution
    - Negative Distribution
    - Stats
use_math: true
sitemap :
  changefreq : daily
  priority : 1.0
---

<p> 다양한 확률분포들이 존재하고 실제 시계열 관련 업무에서 사용하고 있습니다. 그 중에서 최근 공부하고 정리가 필요하게 된 Negative Binomial Distribution에 대해서 정리하는 블로그를 쓰려고 합니다. </p>

<p> 우선 확률분포에 대해서 정확하게 정리하는게 필요한 것 같습니다. 확률 분포란, 어떤 확률 변수 (Random Variable) X가 가지는 확률을 수학적으로 모델링 한 것이라고 할 수 있습니다. 그리고, 이 때 이 확률 분포에 대해서 모델링을 한 함수에 대해서 만나게 되는 확률분포함수로 주로 pmf (probability mass function)과 pdf (probability density function)이 있습니다. </p>

<p> pmf (probability mass function)은 discrete random variable X가 특정값을 가질 확률을 의미합니다. 예를들어 주사위를 "한 번" 굴릴 때 나타나는 값에 대한 확률 변수 (random variable)이 X일 때, 이 확률 변수에 대응되는 확률 질량 함수 pmf는 f_{X}(x)=1/6 이 됩니다. 조금 더 복잡한 예제를 살펴봅시다! </p>

<p> random variable Q 가 2번 코인을 flip 해서 나오는 head의 수에 대한 random variable이라고 가정한 경우, Q는 0, 1, 2 가 가능합니다. 이때, 확률 질량 함수 f(0) = P(Q=0) = P(t, t) = 1/4, f(1) = P(Q=1) = P(h,t) + P(t,h) = 1/4 + 1/4 = 2/4 = 0.5, 그리고 f(2) = P(Q=2) = P(h, h) = 1/4 가 됩니다. 위의 주사위 예제보다는 감이 더 오는 예시라고 생각합니다. </p>

<p> pdf (probability density function)은 continuous random variable X가 특정값을 가질 확률을 의미합니다. pmf와의 차이는 random variable X 가 discrete한 것인지 continuous 한 것이지에 의해 결정됩니다. 그러나, 수학적으로 위의 pmf와 같이 정확히 X="특정 값" 일 때의 확률을 정의할 수 없습니다. 간단하게 생각하면, continuous random variable에서의 X="특정 값"은 무한대의 개념 epsilon이 도입되게 되어 X="특정 값" \in "특정 값 - epsilon" < X < 특정 값"이되며 epsilon이 무한히 0에 다가가게 될 때 무한한 값을 가지게 되는 그런 복잡한 이야기가 있습니다. 저는 이 부분까지는 다루지 않고 scipy.stats의 활용까지 살펴볼 예정이기 때문에 넘어가도록 하겠습니다. </p>

<p> 이 블로그의 목적인 negative binomial distribution을 이해하기 위해서는 우선 이항분포 (binomial distribution)에 대해서 이해 할 필요가 있습니다. 그리고 이 이항 분포를 이해하기 위해서는 베르누이 분포 (Bernoulli distribution)을 이해해야합니다. 베르누이 분포은 단순합니다. 단 1번의 실험에서 나오는 결과가 성공 or 실패라고 생각할 수 있을 때 (즉, label이 0 or 1인 binary label)의 확률 변수 (random variable) X의 분포입니다. 베르누이 분포에서 결과 p가 성공 or 실패할 확률 p 를 parameter로 입력하게 됩니다. </p>

<p> 그리고, 이 베르누이 분포는 아래와 같이 모델링 할 수 있습니다.</p>

<p>$$X~Bernoulli(p), P(X=x) = p^{x}(1-p)^{1-x}$$</p> 

<p>그리고 이 이항분포는 위의 베르누이 분포를 n번 반복할 때의 성공 횟수를 확률 변수 (random variable) X로 하는 분포입니다. 이때 n번의 반복은 베르누이 시행이 독립시행으로써 반복 될 때의 분포가 됩니다. </p>

<p>$$P(X=x) = {n \choose k}p^{x}(1-p)^{n-x}$$</p>

<p> 이제, negative binomial distribution에 대해서 생각해봅시다. negative binomial distribution은 r-th번째의 성공에 관한 확률분포입니다. 즉, 위의 베르누이 시행을 통해서 우리가 r 번째 성공을 거둘 떄 까지의 반복 시행한 횟수를 X 라는 random variable로써 정의 합니다. 베르누이 시행의 확률 p와 성공횟수 r를 parameter을 negative binomial distribution에서는 필요하게 되며 이때 pmf는 아래와 같아집니다.</p>

<p>$$P(X=x) = {x-1 \choose r-1} p^{r}(1-p)^{x-r}$$</p>

<p>위의 수식을 보면 x번째에서 r번째 성공을 해야하므로, x-1번째 까지는 r-1번의 성공을 거두어야 하며, x번째에서 r번째 성공을 하는 확률을 나타낸다는 것을 알 수 있습니다. </p>

<p>개념은 여기까지 살펴보면 모두 이해했다고 할 수 있습니다. 이제 scipy.stats.nbinom를 이용한 코드를 살펴보도록 하겠습니다. </p>

<p>우선 stats 를 사용하게 된 이유는 pmf를 사용해서 probability를 얻고자 할 때, standardized form이 아닌 distribution 의 shift form을 사용하고자 하였기 때문입니다. stats에서 사용하는 negative binomial distribution의 수식은 다음과 같습니다. </p>

<img src="https://github.com/170928/170928.github.io/blob/master/_images//distributions/negative/stats.PNG?raw=true">

<p>그리고, 이 수식에 기반하여 stats.nbinom은 다음과 같이 사용합니다. "nbinom.pmf(k, n, p, loc)" 이때, k는 실패 횟수, n은 성공횟수 (위에서 설명한 r), p는 성공이 발생할 확률 (in 베르누이 시행), 그리고 loc은 standardform을 shift하기 위한 변수입니다. loc은 nbinom.pmf(k - loc, n, p) = nbinom.pmf(k, n, p, loc) 라고 공식 document에서 설명하고 있습니다. 즉, 실패 횟수가 평균적으로 loc번 줄어든 상태에서 n번 성공하는데 필요한 횟수의 분포가 된다고 생각하시면 좀 편하게 이해하실 수 있습니다. 아무래도 n이 클 때 loc이 크면 n번을 성공하기 위해 필요한 횟수가 줄어들겠죠 ...?  </p>

<img src="https://github.com/170928/170928.github.io/blob/master/_images/distributions/negative/negative-bimonial.png?raw=true">

<p> 위의 그림은 n 번의 성공을 하기위해서 성공 확률 p가 0.25, 0.5, 0.75일 때 random variable X가 어떻게 분포하는지를 보여줍니다. 즉 X번 해야 n 번 성공하게 되는 분포가 어떤 확률로 존재하게 되는지를 보여줍니다. p가 낮을수록 X번의 평균이 증가한다는 것을 볼 수 있습니다. </p>

### Reference 
>1. https://sumniya.tistory.com/27 (통계관련 공부를 할때 자주 보는 블로그입니다! 정리를 엄청 깔끔하게 해두셨어요! 제 통계학 지식은 이 블로그에서 온 것 같습니다.)
>2. https://ko.wikipedia.org/wiki%ED%99%95%EB%A5%A0_%EC%A7%88%EB%9F%89_%ED%95%A8%EC%88%98
>3. https://www.statisticshowto.com/negative-binomial-experiment/
>4. https://www.geeksforgeeks.org/python-negative-binomial-discrete-distribution-in-statistics/