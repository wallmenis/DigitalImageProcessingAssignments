import numpy as np
import cv2
#import pyjion; pyjion.enable()

rgb_img=cv2.imread("source.png")

cv2.imshow("source image", rgb_img)

bw=cv2.cvtColor(rgb_img,cv2.COLOR_BGR2GRAY)
print(bw.shape)
#makegrayscale(rgb_img)
blr3x3=cv2.blur(bw,(3, 3))
print("Made 3x3")

blr3x3median=cv2.medianBlur(bw,3)  # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html#median-filtering
print("Made 3x3 median")

blr9x9=cv2.blur(bw,(9, 9))
print("Made 9x9")

blr9x9median=cv2.medianBlur(bw,9)  # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html#median-filtering
print("Made 9x9 median")

blr15x15=cv2.blur(bw,(15, 15))
print("Made 15x15")

blr15x15median=cv2.medianBlur(bw,15)  # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html#median-filtering
print("Made 15x15 median")

laplace=cv2.Laplacian(blr3x3,-1,3)
#laplace=np.add(cv2.Sobel(bw,-1,dx=1,dy=0,ksize=3),cv2.Sobel(bw,-1,dx=0,dy=1,ksize=3))
print("Made laplacian")
finallap=np.add(laplace,blr3x3)

cv2.imshow("grayscale", bw)
cv2.imshow("blured 3x3 image", blr3x3)
cv2.imshow("blured 3x3 median image", blr3x3median)
cv2.imshow("laplacian image", laplace)
cv2.imshow("blured 9x9 image", blr9x9)
cv2.imshow("blured 9x9 median image", blr9x9median)
cv2.imshow("blured 15x15 median image", blr15x15median)
cv2.imshow("laplacian sharpened blured 3x3 image", finallap)

cv2.imwrite("blured_3x3_image_ex2.png", blr3x3)
cv2.imwrite("blured_3x3_median_image_ex2.png", blr3x3median)
cv2.imwrite("laplacian_image_ex2.png", laplace)
cv2.imwrite("blured_9x9_image.png", blr9x9)
cv2.imwrite("blured_9x9_median_image.png", blr9x9median)
cv2.imwrite("blured_15x15_median_image.png", blr15x15median)
cv2.imwrite("laplacian_sharpened_blured_3x3_image.png", finallap)

cv2.waitKey(0)