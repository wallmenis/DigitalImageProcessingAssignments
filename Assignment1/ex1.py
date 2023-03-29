import numpy as np
import cv2



#for(i=0;i<image_source.res_x;i++)
#{
#    for(j=0;j<image_source.res_y;j++)
#    {
#        
#    }
#}

def makegrayscale(image_source):
    print(image_source.shape)
    grayscale=np.zeros((image_source.shape[0],image_source.shape[1]), np.uint8) # Θέλουμε να είναι 8bit λόγω του brightness depth.
    for i in range (0,image_source.shape[0]):
        for j in range (0,image_source.shape[1]):
            grayscale[i][j]=int(round(sum(image_source[i][j])/3))   #3 κανάλια χρώματος άρα δια 3
    return grayscale

def makeblur (image_source, mask_x, mask_y):
    output=np.zeros((image_source.shape[0],image_source.shape[1]), np.uint8)
    extended_source=np.zeros((image_source.shape[0]+2,image_source.shape[1]+2), np.uint8)
    for i in range (0,image_source.shape[0]):
        for j in range (0,image_source.shape[1]):
            extended_source[i+1][j+1]=image_source[i][j]
    for i in range (0,image_source.shape[0]):           # Θα αντιγάψουμε όλα τα pixels στις άκρες για να υπολογίσουμε το blur
        extended_source[i][1]=image_source[i][0]
        extended_source[i][image_source.shape[1]+1]=image_source[i][image_source.shape[1]-1]
    for j in range (0,image_source.shape[1]):
        extended_source[1][j]=image_source[0][j]
        extended_source[image_source.shape[0]+1][j]=image_source[image_source.shape[0]-1][j]
    #Εδώ έχουμε για τις άκρες
    extended_source[0][0]=image_source[0][0]
    extended_source[0][image_source.shape[1]+1]=image_source[0][image_source.shape[1]-1]
    extended_source[image_source.shape[0]+1][0]=image_source[image_source.shape[0]-1][0]
    extended_source[image_source.shape[0]+1][image_source.shape[1]+1]=image_source[image_source.shape[0]-1][image_source.shape[1]-1]
    

rgb_img=cv2.imread("source.png")

cv2.imshow("source image", rgb_img)

bw=makegrayscale(rgb_img)
print(bw.shape)
#makegrayscale(rgb_img)

cv2.imshow("grayscale", bw)
#cv2.imshow("blured 3x3 image", blr3x3)
#cv2.imshow("blured 7x7 image", blr7x7)
#cv2.imshow("blured 15x15 image", blr15x15)
#cv2.imshow("laplacian sharpened blured 15x15 image", laplace)

#cv2.imwrite("blured 3x3 image.png", blr3x3)
#cv2.imwrite("blured 7x7 image.png", blr7x7)
#cv2.imwrite("blured 15x15 image.png", blr15x15)
#cv2.imwrite("laplacian sharpened blured 15x15 image.png", laplace)
cv2.waitKey(0)