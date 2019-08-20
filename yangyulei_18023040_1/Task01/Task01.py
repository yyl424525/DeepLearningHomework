import cv2 as cv
import matplotlib.pyplot as plt

image_arr = cv.imread('test01.jpg')
b,g,r=cv.split(image_arr)  #cv读取的时候按bgr的顺序排列
print(image_arr)
img2=cv.merge([b,r,g])    #按brg顺序排列
cv.imshow('input_image', img2)  #按cv方式显示改变通道后的图片
cv.waitKey(0)
cv.destroyAllWindows()#删除建立的全部窗口，释放资源

plt.figure('brg')  #按matplotlib方式显示改变通道后的图片
plt.imshow(img2)
plt.show()


