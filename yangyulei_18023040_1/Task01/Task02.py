import cv2 as cv

image_arr = cv.imread('test01.jpg')
b,g,r=cv.split(image_arr)  #cv读取的时候按bgr的顺序排列
print(b.shape)

#红色矩形区域
# for i in range(18,49):
#    for j in range(2,43):
#
#       r[i][j]=255
#       g[i][j]=0
#       b[i][j]=0

#红色矩形框,边框3个像素
for i in range(18,49):
   for j in range(2,6):
      r[i][j]=255
      g[i][j]=0
      b[i][j]=0
   for j in range(39,43):
      r[i][j]=255
      g[i][j]=0
      b[i][j]=0

for i in range(18,21):
   for j in range(2,43):
      r[i][j]=255
      g[i][j]=0
      b[i][j]=0

for i in range(45,49):
   for j in range(2,43):
      r[i][j]=255
      g[i][j]=0
      b[i][j]=0

img2=cv.merge([b,g,r])    #按brg顺序排列
cv.imshow('input_image', img2)  #按cv方式显示改变通道后的图片
cv.waitKey(0)
cv.destroyAllWindows()#删除建立的全部窗口，释放资源


#保存图片
cv.imwrite("output.jpg",img2)








