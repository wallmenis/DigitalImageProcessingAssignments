import numpy as np
import cv2
#import pyjion; pyjion.enable()

def kernel_to_pixel(image_source,kernel,pos_x,pos_y):
    pixelsum=0
    tmpx=0
    tmpy=0
    for i in range (-kernel.shape[0]//2, kernel.shape[0]//2):
        for j in range (-kernel.shape[1]//2, kernel.shape[1]//2):
            tmpx=i+pos_x
            tmpy=j+pos_y
            if((tmpx not in range (0,image_source.shape[0]))or (tmpy not in range (0,image_source.shape[1]))):    #Αντί να επεκτείνουμε τον "καμβά" που δουλεύουμε, βάζουμε απευθείας στην μάσκα τις τιμές για καλύτερη χωρική πολυπλοκότητα
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

#def img_convolve_sub (image_source,kernel,coords):  # Θα κάνουμε υλοποίηση divide and conquer!
#    if coords[0]-coords[1] = 0 and coords[2] - coords[3] = 0:
#        return kernel_to_pixel(image_source,kernel,coords[0],coords[2])
#    output=np.
#    return output

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

    kernel=np.ones((mask_x,mask_y))/(mask_x*mask_y)     #Είναι το kernel για τον μέσο όρο
    output=img_convolve(image_source,kernel)
    return output

#def makelaplacian(bw, mask_x, mask_y):
    

rgb_img=cv2.imread("source.png")

cv2.imshow("source image", rgb_img)

bw=makegrayscale(rgb_img)
print(bw.shape)
#makegrayscale(rgb_img)
blr3x3=makeblur(bw,3,3)
print("Made 3x3")

#blr9x9=makeblur(bw,9,9)
#print("Made 9x9")

#blr15x15=makeblur(bw,15,15)
#print("Made 15x15")
#laplace3x3=makelaplacian(bw,3,3)
#print("Made laplacian 3x3")

cv2.imshow("grayscale", bw)
cv2.imshow("blured 3x3 image", blr3x3)
#cv2.imshow("blured 9x9 image", blr9x9)
#cv2.imshow("blured 15x15 image", blr15x15)
#cv2.imshow("laplacian sharpened blured 15x15 image", laplace)

#cv2.imwrite("blured 3x3 image.png", blr3x3)
#cv2.imwrite("blured 9x9 image.png", blr9x9)
#cv2.imwrite("blured 15x15 image.png", blr15x15)
#cv2.imwrite("laplacian sharpened blured 15x15 image.png", laplace)


#testing
blured_gray3x3=cv2.blur(bw,(3, 3)) #opencv for refference
#blured_gray9x9=cv2.blur(bw,(9, 9)) #opencv for refference
#blured_gray15x15=cv2.blur(bw,(15, 15)) #opencv for refference
#difference=np.subtract(blr3x3,blured_gray3x3)
#difference=np.subtract(blr9x9,blured_gray9x9)
#difference=np.subtract(blr15x15,blured_gray15x15)
cv2.imshow("blured 3x3 image cv", blured_gray3x3)
#cv2.imshow("blured 9x9 image cv", blured_gray9x9)
#cv2.imshow("blured 15x15 image cv", blured_gray15x15)
#cv2.imshow("blured image diff", difference)


cv2.waitKey(0)