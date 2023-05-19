import cv2
import numpy as np

def sub_no_under_overflow(im1,im2):
    final_image = np.zeros(im1.shape, dtype=np.uint16)
    final_image += im1
    final_image -= im2
    final_image[final_image>255] = 255
    final_image[final_image<0] = 0
    final_image=np.array(final_image,dtype=np.uint8)
    return final_image

original=cv2.imread("cameraman.tif")

kernel= np.ones((3,3))
#kernel= np.array(((0.0,1.0,0.0),(1.0,1.0,1.0),(0.0,1.0,0.0)))
print(kernel)
eroded=cv2.erode(original,kernel)

#final_image=original-eroded

final_image=sub_no_under_overflow(original,eroded)

print(final_image.shape)

cv2.imshow("Original", original)
cv2.imshow("Eroded", eroded)
cv2.imshow("Final Image", final_image)

cv2.waitKey(0)