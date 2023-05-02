import numpy as np
import cv2
import scipy
#import pyjion; pyjion.enable()

def kernel_to_pixel(image_source,kernel,pos_x,pos_y):
    pixelsum=0
    tmpx=0
    tmpy=0
    for i in range (-kernel.shape[0]//2, kernel.shape[0]//2):
        for j in range (-kernel.shape[1]//2, kernel.shape[1]//2):
            tmpx=i+pos_x
            tmpy=j+pos_y
            #if((tmpx not in range (0,image_source.shape[0]))or (tmpy not in range (0,image_source.shape[1]))):    #Αντί να επεκτείνουμε τον "καμβά" που δουλεύουμε, βάζουμε απευθείας στην μάσκα τις τιμές για καλύτερη χωρική πολυπλοκότητα
                #print ("I GOT IN HERE! {} {}".format( tmpx,tmpy))
            if(tmpx < 0):
                tmpx=0
            if(tmpx > image_source.shape[0]-1):
                tmpx = image_source.shape[0]-1
            if(tmpy < 0):
                tmpy =0
            if(tmpy > image_source.shape[0]-1):
                tmpy = image_source.shape[1]-1
            #print(tmpx, tmpy)
            pixelsum=pixelsum+kernel[i][j]*image_source[tmpx][tmpy]
    return pixelsum

def med_to_pixel(image_source,mask_x,mask_y,pos_x,pos_y):
    pixelmed=0
    tmpx=0
    tmpy=0
    tmpimage=np.zeros((mask_x*mask_y))
    for i in range (-mask_x//2, mask_y//2):
        for j in range (-mask_x//2, mask_y//2):
            tmpx=i+pos_x
            tmpy=j+pos_y
                #Αντί να επεκτείνουμε τον "καμβά" που δουλεύουμε, βάζουμε απευθείας στην μάσκα τις τιμές για καλύτερη χωρική πολυπλοκότητα
                #print ("I GOT IN HERE! {} {}".format( tmpx,tmpy))
            if(tmpx < 0):
                tmpx= 0
            if(tmpx > image_source.shape[0]-1):
                tmpx = image_source.shape[0]-1
            if(tmpy < 0):
                tmpy = 0
            if(tmpy > image_source.shape[1]-1):
                tmpy = image_source.shape[1]-1
            #print(tmpx, tmpy)
            tmpimage[(i)*(j)+i]=image_source[tmpx][tmpy]
            if image_source[tmpx][tmpy] == 0:
                print("At {} {}: {}".format(tmpx, tmpy, image_source[tmpx][tmpy]))
            #print(image_source[tmpx][tmpy])
    #print(tmpimage)
    
    #tmpimage=np.sort(tmpimage)
    
    #pixelmed=tmpimage[mask_x*mask_y//2]
            #print(pixelmed)
    pixelmed=np.median(tmpimage)
    return pixelmed

def img_convolve (image_source,kernel):             # Κάνει συνέλιξη σε εικόνες με το δωσμένο kernel
    result_image=np.zeros((image_source.shape[0],image_source.shape[1]), np.uint8)
    inv_kernel=np.zeros((kernel.shape[0],kernel.shape[1]))      #Για τυπικότητα, αντιστρέφουμε το kernel
    for i in range (0, kernel.shape[0]):
        for j in range (0, kernel.shape[1]):
            inv_kernel[kernel.shape[0]-1-i][kernel.shape[1]-1-j]=kernel[i][j]
    for i in range (0, image_source.shape[0]):
        for j in range (0, image_source.shape[1]):
            result_image[i][j]=kernel_to_pixel(image_source,inv_kernel,i,j)
    return result_image


def makegrayscale(image_source):
    print(image_source.shape)
    grayscale=np.zeros((image_source.shape[0],image_source.shape[1]), np.uint8) # Θέλουμε να είναι 8bit λόγω του brightness depth.
    for i in range (0,image_source.shape[0]):
        for j in range (0,image_source.shape[1]):
            grayscale[i][j]=int(round(sum(image_source[i][j])/3))   #3 κανάλια χρώματος άρα δια 3
    return grayscale

def makeblur (image_source, mask_x, mask_y):
    kernel=np.ones((mask_x,mask_y))/(mask_x*mask_y)
    output=img_convolve(image_source,kernel)
    return output

def makeblurmedian(image_source,mask_x,mask_y):
    result_image=np.zeros((image_source.shape[0],image_source.shape[1]), np.uint8)
    for i in range (0, image_source.shape[0]):
        for j in range (0, image_source.shape[1]):
            result_image[i][j]=med_to_pixel(image_source,mask_x,mask_y,i,j)
    return result_image

def makelaplacian(image_source):
    kernel=np.array([[-1, -1, -1],[-1,8,-1],[-1,-1,-1]])/9     #Οι παράγωγοι κάνουν τέτοια μάσκα. Είναι δια 9 για λόγους έντασης
    result_image=img_convolve(image_source,kernel)
    return result_image    

rgb_img=cv2.imread("source.png")

cv2.imshow("source image", rgb_img)

bw=makegrayscale(rgb_img)
print(bw.shape)
#makegrayscale(rgb_img)
blr3x3=makeblur(bw,3,3)
print("Made 3x3")

blr9x9=makeblur(bw,9,9)
print("Made 9x9")

blr15x15=makeblur(bw,15,15)
print("Made 15x15")

blr3x3median=makeblurmedian(bw,3,3)
print("Made 3x3 median")

blr9x9median=makeblurmedian(bw,9,9)
print("Made 9x9 median")

blr15x15median=makeblurmedian(bw,15,15)
print("Made 9x9 median")


laplace=makelaplacian(blr3x3)
print("Made laplacian")

finallap=np.add(laplace,blr3x3)

cv2.imshow("grayscale", bw)
cv2.imshow("laplacian image", laplace)
cv2.imshow("blured 3x3 image", blr3x3)
cv2.imshow("blured 9x9 image", blr9x9)
cv2.imshow("blured 15x15 image", blr15x15)
cv2.imshow("blured 3x3 median image", blr3x3median)
cv2.imshow("blured 9x9 median image", blr9x9median)
cv2.imshow("blured 15x15 median image", blr15x15median)
cv2.imshow("laplacian sharpened blured 3x3 image", finallap)

cv2.imwrite("blured_3x3_image_ex1.png", blr3x3)
cv2.imwrite("blured_9x9_image_ex1.png", blr9x9)
cv2.imwrite("blured_15x15_image_ex1.png", blr15x15)
cv2.imwrite("blured_3x3_median_image_ex1.png", blr3x3median)
cv2.imwrite("blured_9x9_median_image_ex1.png", blr9x9median)
cv2.imwrite("blured_15x15_median_image_ex1.png", blr15x15median)
cv2.imwrite("laplacian_sharpened_blured_3x3_image_ex1.png", finallap)


cv2.waitKey(0)