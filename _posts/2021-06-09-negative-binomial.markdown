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
sitemap :
  changefreq : daily
  priority : 1.0
---


<p> 다양한 확률분포들이 존재하고 실제 시계열 관련 업무에서 사용하고 있습니다. 그 중에서 최근 공부하고 정리가 필요하게 된 Negative Binomial Distribution에 대해서 정리하는 블로그를 쓰려고 합니다. </p>

<p> 우선 확률분포에 대해서 정확하게 정리하는게 필요한 것 같습니다. 확률 분포란, 어떤 확률 변수 (Random Variable) X가 가지는 확률을 수학적으로 모델링 한 것이라고 할 수 있습니다. 그리고, 이 때 이 확률 분포에 대해서 모델링을 한 함수에 대해서 만나게 되는 확률분포함수로 주로 pmf (probability mass function)과 pdf (probability density function)이 있습니다. </p>

<p> pmf (probability mass function)은 discrete random variable X가 특정값을 가질 확률을 의미합니다. 예를들어 주사위를 "한 번" 굴릴 때 나타나는 값에 대한 확률 변수 (random variable)이 X일 때, 이 확률 변수에 대응되는 확률 질량 함수 pmf는 f_{X}(x)=1/6 이 됩니다. 조금 더 복잡한 예제를 살펴봅시다! </p>

<p> random variable Q 가 2번 코인을 flip 해서 나오는 head의 수에 대한 random variable이라고 가정한 경우, Q는 0, 1, 2 가 가능합니다. 이때, 확률 질량 함수 f(0) = P(Q=0) = P(t, t) = 1/4, f(1) = P(Q=1) = P(h,t) + P(t,h) = 1/4 + 1/4 = 2/4 = 0.5, 그리고 f(2) = P(Q=2) = P(h, h) = 1/4 가 됩니다. 위의 주사위 예제보다는 감이 더 오는 예시라고 생각합니다. </p>

<p> pdf (probability density function)은 continuous random variable X가 특정값을 가질 확률을 의미합니다. pmf와의 차이는 random variable X 가 discrete한 것인지 continuous 한 것이지에 의해 결정됩니다. 그러나, 수학적으로 위의 pmf와 같이 정확히 X="특정 값" 일 때의 확률을 정의할 수 없습니다. 간단하게 생각하면, continuous random variable에서의 X="특정 값"은 무한대의 개념 epsilon이 도입되게 되어 X="특정 값" \in "특정 값 - epsilon" < X < 특정 값"이되며 epsilon이 무한히 0에 다가가게 될 때 무한한 값을 가지게 되는 그런 복잡한 이야기가 있습니다. 저는 이 부분까지는 다루지 않고 scipy.stats의 활용까지 살펴볼 예정이기 때문에 넘어가도록 하겠습니다. </p>

<p> </p>



### Reference 
>1. https://sumniya.tistory.com/27 (통계관련 공부를 할때 자주 보는 블로그입니다! 정리를 엄청 깔끔하게 해두셨어요)
>2. https://ko.wikipedia.org/wiki%ED%99%95%EB%A5%A0_%EC%A7%88%EB%9F%89_%ED%95%A8%EC%88%98
>3. 