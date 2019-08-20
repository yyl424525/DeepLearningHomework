import math
import  numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#自定义3次函数形式
def func(x, a, b,c,d):
    return a*x*x*x+b*x*x+c*x+d

a=18 #学号前两位
b=40 #学号后两位
step=((2*math.pi-b)/a+b/a)/2000 #计算步长
x=np.arange(2001.)
y=np.arange(2001.)
for i in range(0,2001):
     x[i]=-b/a+i*step
     y[i]=math.cos(a*x[i]+b)
     print("第",i,"个采样点：(",x[i],",",y[i],")")
plt.scatter(x,y)
#显示样本点所得到的图
plt.show()

#非线性最小二乘法拟合
popt, pcov = curve_fit(func, x, y)
#获取popt里面是拟合系数
print(popt)
a = popt[0]
b = popt[1]
c = popt[2]
d = popt[3]
yvals = func(x,a,b,c,d) #拟合y值
print('popt:', popt)
print('系数a:', a)
print('系数b:', b)
print('系数c:', c)
print('系数d:', d)
print('系数pcov:', pcov)
print('系数yvals:', yvals)
#绘图
plot1 = plt.plot(x, y, 's',label='original values')
plot2 = plt.plot(x, yvals, 'r',label='polyfit values')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=4) #指定legend的位置右下角
plt.title('curve_fit')
plt.show()

