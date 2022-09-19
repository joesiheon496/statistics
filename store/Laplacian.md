[생새우 초밥집](https://freshrimpsushi.github.io/posts/what-is-an-euclidean-space/)

[공부 블로그 사이트](https://angeloyeo.github.io/2019/08/28/laplacian.html)

## 라플라시안의 정의(Definition)
유클리드 공간에서 두 번 미분할 수 있는 스칼라 함수 f에 대하여 라플라시안(Laplacian)은 f에 대한 그레디언트의 발산으로 정의되며 수식으로는 다음과 같다.
$$\Delta f = \Delta ^2 f = \nabla \cdot \nabla f$$

> $\Delta$ : Delta
> $\nabla$ : nabla
> 유클리드 공간 : 우리가 살아가고 있는 3차원 공간을 포함해서 평면, 수직선은 물론 그 이상의 다차원 공간까지 표현하는 공간
> 스칼라 함수 : 입력이 하나에 출력이 하나

## 스칼라 함수의 기울기(gradient)와 발산(divergence)
라플라시안은 스칼라 함수에 대해서 $div(grad(f))$와 같이 gradient 연산을 먼저 취해준 뒤, 그것으로 출력되는 벡터장에 대해 divergence를 구한 것이다.

Gradient는 경사도를 이야기하는데 가파르게 올라가는 방향으로 벡터장이 형성된다.

![image](https://user-images.githubusercontent.com/56191064/190578043-3e12b1dc-bfa3-496e-afb6-464d0a6e0616.png)

![image](https://user-images.githubusercontent.com/56191064/190578021-4c8698c5-a5cb-4be4-aec0-fd232bc45460.png)

[참고](https://m.blog.naver.com/wideeyed/221029470554)

```python
import copy
import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

def numerical_gradient(f,x):
    delta_x = 1e-7 # 0.0000001
    gradient = np.zeros_like(x) # 같은 모양의 0으로 이루어딘 배열을 반환한다

    for i in range(x.size):
        _x1 = copy.deepcopy(x) # 새로운 복합 객체를 생성한 다음 원본에서 발견된 객체의 복사본을 재귀적(recursively)으로 삽입합니다.
        _x1[i] = x[i] + delta_x
        _y1 = f(_x1) # f(x+delta_x) 계산

        _x2 = copy.deepcopy(x)
        _x2[i] = x[i] - delta_x
        _y2 = f(_x2) # f(x-delta_x) 계산

        gradient[i] = (_y1-_y2)/(delta_x*2)
    
    return gradient

def numerical_gradient_batch(f,X):
    print("X: "+str(X))
    gradient = np.zeros_like(X)

    x = np.array([0.0,0.0])

    for i in range(X[0].size):
        for s in range(X.shape[0]):
            x[s] = X[s][i]
            # x[0] = X[0][i] <-- 1차원으로 flatten된 데이터를 하나씩 x[0]에 입력
            # x[1] = X[1][i] <-- 1차원으로 flatten된 데이터를 하나씩 x[1]에 입력
        _gradient = numerical_gradient(f,x)

        for s in range(X.shape[0]):
            gradient[s][i] = -_gradient[s]
            # gradient[0][i] = -_gradient[0]<-- 구해진 기울기 _gradient[0]를 부호로 바꿔 gradient 저장
            # gradient[1][i] = -_gradient[1]<-- 구해진 기울기 _gradient[1]를 부호로 바꿔 gradient 저장
            # 부호를 바꾸는 이유는 작아지는 방향을 알기 위해서이다
    print("gradient: "+str(gradient))
    return gradient

def f(x):
    return x[0]**2 - x[1]**2

x0 = np.arange(-5.0, 5.01, 0.25)
x1 = np.arange(-5.0, 5.01, 0.25)
X,Y = np.meshgrid(x0,x1) # vectorized의 값을 구하기 위해 2d coordinate array를 만든다

Xf = X.flatten() # 축소된 배열의 복사본을 1D로 반환
Yf = X.flatten() # 축소된 배열의 복사본을 1D로 반환
'''
x0 값이 [2,3]이고 x1값이 [5,6]라고 가정하면
X값은 [[2,3],[2,3]]
Y값은 [[5,5],[6,6]]
Xf값은 [2,3,2,3]
Yf값은 [5,5,6,6]
'''

# 수치 편미분 Quiver 그리기
numerical_gradient = numerical_gradient_batch(f,np.array([Xf,Yf]))
plt.quiver(Xf,Yf,numerical_gradient[0], numerical_gradient[1], color = "r", headwidth=3) # quiver(화살모음) 그리기
plt.xlabel('x0')
plt.ylabel('x1')

# 미분전 본래 함수 3차원 그래프 그리기
Z = X**2 - Y**2
fig1 = plt.figure()
ax1 = fig1.gca(projection = '3d')
surf1 = ax1.plot_surface(X,Y,Z, rstride = 1, cstride = 1, linewidth = 1,
                         cmap = cm.RdPu, antialiased = True)
fig1.colorbar(surf1, shrink = 0.5, aspect = 5)
ax1.view_init(elev = 30, azim = 60)
ax1.dist = 7
plt.draw()
plt.show()
```

divergence가 수렴하면 음의 값

divergence가 발산하면 양의 값
