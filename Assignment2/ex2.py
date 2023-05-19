import cv2
import numpy as np

def add_no_under_overflow(im1,im2):
    final_image = np.zeros(im1.shape, dtype=np.uint16)
    final_image += im1
    final_image += im2
    final_image[final_image>255] = 255
    final_image[final_image<0] = 0
    final_image=np.array(final_image,dtype=np.uint8)
    return final_image

original=cv2.imread("house.tif")

laplacian=cv2.Laplacian(original,-1)
sobel=add_no_under_overflow(cv2.Sobel(original,-1,dx=1,dy=1,ksize=3),add_no_under_overflow(cv2.Sobel(original,-1,dx=1,dy=0,ksize=3),cv2.Sobel(original,-1,dx=0,dy=1,ksize=3)))
canny=cv2.Canny(original,0,150)

cv2.imshow("Original", original)
cv2.imshow("2 (horizontal and vertical) Sobel Filter", sobel)
cv2.imshow("Laplacian", laplacian)
cv2.imshow("Canny", canny)

cv2.waitKey(0)