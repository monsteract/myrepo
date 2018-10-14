Linear Regression
================
*__수정중__*

machine learning 분야에서 지도_supervised_ 학습의 방법중의 하나인 linear regression입니다. 이 방법은 주어진 데이터(x, y)에 대해 input과 output사이의 관계를 나타낼 수 있는 _함수_를 설정하는 방법입니다. 정확히 말하자면, 설정한 함수의 __*weight*__를 조정하면서 입력받은 데이터의 output값을 더 잘 예측하도록 하는 방법입니다.

예를 들어 볼까요? 상가건물의 가격을 예측할수 있는 machine learning프로그램을 만들어 봅시다.  데이터의 input인 $x$는 건물의 면적을 나타낼수 있을겁니다. 또, output인 $y$는 건물의 가격이 될겁니다. 

많은 상가 건물에 대한 자료를 수집하며 $x, y$에 관해 데이터를 모은 뒤 , 다음과 같은 함수를 생각해 볼수 있겠습니다.
$$
f(x) = w_0 + w_1x \\
y_i = w_0 + w_1x_i + \epsilon_i
$$
여기서 $f(x)$는 설정한 함수이며, 데이터의 output에 대한 예측값 입니다.







---

```python
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# make data frame reading csv file
df = pd.read_csv('/home/rak/Desktop/artificia_intelligience/exam_scores.csv')

#draw 3d plot for scores on scatter
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

cir = df['Circuit']
data = df['DataStructure']
mach = df['MachineIntelligence']
ax.scatter(cir, data, mach)

#set name for axis
ax.set_ylabel('data_structure')
ax.set_xlabel('circuit')
ax.set_zlabel('machine_intelligence')

#value settting
n = 1000000 #number of iteration
rate = 0.0001 #learning_rate
data_len = len(df['MachineIntelligence']) #length of data(score)
w = np.zeros(3).reshape(3, 1) #initialize regression weight

#define feature extractor
h = np.ones(data_len).reshape(data_len, 1)

#data 가공하기
x_data = np.matrix(df['DataStructure']).reshape(data_len, 1)
x_circuit = np.matrix(df['Circuit']).reshape(data_len,1)
y_out = np.matrix(df['MachineIntelligence']).reshape(data_len,1)

#H 만들기
H1 = h
H2 = np.multiply(h, x_circuit)
H3 = np.multiply(h, x_data)
H = np.hstack((H1, H2, H3))

#cost function 구하기

#calculate RSS function 
def average_rss(w):
    return np.dot((y_out - np.dot(H, w)).T, (y_out - np.dot(H, w)))/1000
#( (y-hw)t * (y-hw) )/1000

#analytic gradient of rss
def grad_rss(w):
    grad = (-2)*np.dot(H.T, (y_out - np.dot(H, w)))/1000
    return grad
#-2h_t(y-Hw)/1000

#gradient descent and determine weight
w_descent = np.zeros(3).reshape(3, 1)
plot_rss = average_rss(w_descent)
count = 0

while count < n:
    w_descent = w_descent - rate*grad_rss(w_descent)
    plot_rss = np.vstack((plot_rss, average_rss(w_descent)))
    count += 1

print(w_descent)   

###################################################################################################
## 여기까지 gradient descent 를 적용 하는 부분입니다. 
## w(t+1)에 w(t)와 stepsize * gradient를 뺀값을 대입, 반복하며 w가 결과적으로 minimum값으로 수렴하도록 했습니다.
## RSS function과 Gradient는 손으로 계산한 수식을 이용, 대입했습니다.
## 처음에 급격한 변화를 보이다가 갈수록 변화량이 적어집니다.

#plot rss for number of iteration
source = plot_rss.A1 #get data of Rss over iteration
fig2 = plt.figure() 
rss = fig2.add_subplot(111)
rss.plot(source)
rss.set_xscale('log') #for readable graph set x axis in logscale
fig2

#set normal vector & intersection for drawing plane
inter = [w_descent.A1[0]]
n_vec = [w_descent.A1[1], w_descent.A1[2]]

# create x,y(input)
xx, yy = np.meshgrid(range(101), range(101))

# calculate z(estimated output)
zz = n_vec[0] * xx + n_vec[1] * yy + inter

# plot the surface
ax.plot_wireframe(xx, yy, zz, rstride = 20, cstride = 20)
fig

#######closed form solution
w_close = np.linalg.inv(np.dot(H.T, H))*np.dot(H.T, y_out)
# w = (H_t * H)^-1 * H_t * y

print(w_close)
###########################################################################################
## 위에 나오는 식이 weight를 closed-form solution으로 구한값입니다.
## Gradient(RSS) = 0 임을 이용, w값을 linear algebra를 통해 직접 구한값입니다.
## 데이터의 갯수가 w의 갯수보다 많고, 데이터의 갯수가 1000개밖에 되지 않기 때문에 구하는데 무리가 없었습니다.
## gradient descent를 이용해 나온값과 아주 비슷합니다.

#draw 3d plot for scores on scatter
fig3 = plt.figure()
ax_c = fig3.add_subplot(111,projection='3d')

cir = df['Circuit']
data = df['DataStructure']
mach = df['MachineIntelligence'] ## in order to see the data and the plane together
ax_c.scatter(cir, data, mach)

ax_c.set_ylabel('data_structure')
ax_c.set_xlabel('circuit')
ax_c.set_zlabel('machine_intelligence')

inter_c = [w_close.A1[0]]
n_vec_c = [w_close.A1[1], w_close.A1[2]]

# create x,y
xxx, yyy = np.meshgrid(range(101), range(101))

# calculate corresponding z
zzz = n_vec_c[0] * xxx + n_vec_c[1] * yyy + inter_c

# plot the surface
ax_c.plot_wireframe(xxx, yyy, zzz, rstride = 20, cstride = 20)
fig3

#################################          result analyze            #################################  
## 이렇게 gradient descent와 closedform solution의 방식으로 weight를 구해봤습니다.
## 직접 값을 확인한 결과 두가지 방식으로 구한 weight가  비슷함을 볼수 있습니다.
## 또, 각각의 weight를 이용해 평면을 그려 확인해보니 도식해놓은 exam_score들과 경향이 비슷함을 알수있습니다.
## gradient descent를 적용하면서 loss fundtion의 변화량을 plot 해봤고, 처음에 많이 줄어들고 갈수록 변화량이 적어져서 
## logscale 을 이용해 매반복마다 Rss가 줄어드는 것을 확인했습니다.
```

```

```







