import numpy as np
#import matplotlib.pyplot as plt
import cv2

def make_histogram_image(hist):
    hist_w = 512
    hist_h = 512
    bin_w = int(round( hist_w/256 ))
    histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    #cv2.normalize(hist, hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    for i in range(1, 256):
        cv2.line(histImage, ( bin_w*(i-1), hist_h - int(hist[i-1]) ),
                ( bin_w*(i), hist_h - int(hist[i]) ),
                ( 255, 0, 0), thickness=2)
    #cv2.imshow("Histogram", histImage)
    return histImage

def make_g_func(hist):
    sum=0
    #output=np.zeros((256, 1),np.uint8)
    output=hist.copy()
    #print(output)
    for i in range(0,256):
        sum = output[i] + sum
        output[i] = sum
        #print(sum)
        #print(hist[i])
    return np.floor(255*output/np.sum(hist))

def calc_histogram(img):
    output=np.zeros((256, 1))
    for i in img:
        output[i] = output[i] + 1
    return output

def set_histogram(img,gfunc):
    image=img.copy()
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            image[i][j]=gfunc[img[i][j]]
    return image

og=cv2.imread("source2.png")
guide=cv2.imread("source.png")
ogbw=cv2.cvtColor(og,cv2.COLOR_BGR2GRAY)
guidebw=cv2.cvtColor(guide,cv2.COLOR_BGR2GRAY)
ogbweq=cv2.equalizeHist(ogbw)


hist=cv2.calcHist([guidebw],[0], None, [256], (0,256), True)
cv2.normalize(hist, hist, alpha=0, beta=512, norm_type=cv2.NORM_MINMAX)
#histog=cv2.calcHist([ogbw],[0], None, [256], [0,256], False)
#cv2.normalize(histog, histog, alpha=0, beta=512, norm_type=cv2.NORM_MINMAX)
#histogeq=cv2.calcHist([ogbweq],[0], None, [256], [0,256], False)
#cv2.normalize(histogeq, histogeq, alpha=0, beta=512, norm_type=cv2.NORM_MINMAX)
print(hist.shape)

#histcustom=calc_histogram(guidebw)


gfunc=make_g_func(hist)

#print(hist)
#print(gfunc)

#print(gfunc[254])
final=set_histogram(ogbweq,gfunc)

histfinal=cv2.calcHist([final],[0], None, [256], [0,256], False)
cv2.normalize(histfinal, histfinal, alpha=0, beta=512, norm_type=cv2.NORM_MINMAX)

cv2.imwrite("Grayscale.png", ogbw)

cv2.imshow("Original", og)
cv2.imshow("Guide", guide)
cv2.imshow("Guide Grayscale", guidebw)
cv2.imshow("Original Grayscale", ogbw)
cv2.imshow("Original Grayscale Equalized Histogram", ogbweq) # https://docs.opencv.org/3.4/d4/d1b/tutorial_histogram_equalization.html
cv2.imshow("Final Image", final)
cv2.imwrite("final_image_ex3.png", final)
#print(histog)
#cv2.imshow("Histogram OG", make_histogram_image(histog))
#cv2.imshow("Histogram OG equalised", make_histogram_image(histogeq))
cv2.imshow("Histogram Guide", make_histogram_image(hist))
cv2.imshow("gfunc", make_histogram_image(gfunc))
cv2.imshow("Histogram Final", make_histogram_image(histfinal))
#cv2.imshow("Custom Hist",make_histogram_image(histcustom))
#plt.hist(histog)
#plt.show()

cv2.waitKey(0)