import numpy as np
import cv2
#import pyjion; pyjion.enable()


def kernel_to_pixel(image_source,kernel,pos_x,pos_y,divider=1):
    pixelsum=0
    crpxs=pos_x-kernel.shape[0]//2
    crpxe=pos_x+kernel.shape[0]//2
    crpys=pos_y-kernel.shape[1]//2
    crpye=pos_y+kernel.shape[1]//2
    tmpimage=image_source[(crpxs):(crpxe+1), (crpys):(crpye+1)] # Τα +1 είναι για να φτιάξουμε το offset που κάνει η numpy
    for i in range(0,tmpimage.shape[0]):
        for j in range(0,tmpimage.shape[1]):
            #print(i,j,tmpimage.shape[0],tmpimage.shape[1],kernel.shape[0]//2,kernel.shape[1]//2)
            pixelsum=pixelsum+kernel[i][j]*tmpimage[i][j]/divider
    pixelsum=np.around(pixelsum)
    if pixelsum>255:
        pixelsum=255
    if pixelsum<0:
        pixelsum=0
    return pixelsum

def med_to_pixel(image_source,mask_x,mask_y,pos_x,pos_y):
    crpxs=pos_x-mask_x//2
    crpxe=pos_x+mask_x//2
    crpys=pos_y-mask_y//2
    crpye=pos_y+mask_y//2
    tmpimage=image_source[(crpxs):(crpxe+1), (crpys):(crpye+1)]
    pixelmed=np.median(tmpimage)
    return pixelmed

def img_convolve (image_source,kernel,divider=1):             # Κάνει συνέλιξη σε εικόνες με το δωσμένο kernel
    result_image=np.zeros((image_source.shape[0],image_source.shape[1]), np.uint8)
    inv_kernel=np.zeros((kernel.shape[0],kernel.shape[1]))      #Για τυπικότητα, αντιστρέφουμε το kernel
    mask_x=kernel.shape[0]
    mask_y=kernel.shape[1]
    tmpimage=np.pad(image_source, (mask_x,mask_y),mode='edge')  # Κάνουμε pad ανάλογα την εικόνα
    
    for i in range (0, kernel.shape[0]):
        for j in range (0, kernel.shape[1]):
            inv_kernel[kernel.shape[0]-1-i][kernel.shape[1]-1-j]=kernel[i][j]
    for i in range (mask_x, image_source.shape[0]+mask_x):
        for j in range (mask_y, image_source.shape[1]+mask_y):
            result_image[i-mask_x][j-mask_y]=kernel_to_pixel(tmpimage,inv_kernel,i,j,divider)
    return result_image


def makegrayscale(image_source):
    print(image_source.shape)
    grayscale=np.zeros((image_source.shape[0],image_source.shape[1]), np.uint8) # Θέλουμε να είναι 8bit λόγω του brightness depth.
    for i in range (0,image_source.shape[0]):
        for j in range (0,image_source.shape[1]):
            grayscale[i][j]=int(round(sum(image_source[i][j])/3))   #3 κανάλια χρώματος άρα δια 3
    return grayscale

def makeblur (image_source, mask_x, mask_y):
    kernel=np.ones((mask_x,mask_y))/(mask_x*mask_y)         # Η μάσκα μέσου όρου (άσοι όλοι)
    output=img_convolve(image_source,kernel)
    return output

def makeblurmedian(image_source,mask_x,mask_y):
    result_image=np.zeros((image_source.shape[0],image_source.shape[1]), np.uint8)
    tmpimage=np.pad(image_source, (mask_x,mask_y),mode='edge')
    #cv2.imshow("test", tmpimage)
    for i in range (mask_x, image_source.shape[0]+mask_x):
        for j in range (mask_y, image_source.shape[1]+mask_y):
            result_image[i-mask_x][j-mask_y]=med_to_pixel(tmpimage,mask_x,mask_y,i,j)     # Στην ουσία παίρνουμε την median τιμή από το μέρος της εικόνας
    return result_image

def makelaplacian(image_source):
    kernel=np.array([[-1, -1, -1],[-1,8,-1],[-1,-1,-1]])/9     #Οι παράγωγοι κάνουν τέτοια μάσκα. Είναι δια 9 για λόγους έντασης
    #kernel=np.array([[0, -2, 0],[-2,8,-2],[0,-2,0]])/9 
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
print("Made 15x15 median")

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

cv2.imwrite("laplacian_image_ex1.png", laplace)
cv2.imwrite("blured_3x3_image_ex1.png", blr3x3)
cv2.imwrite("blured_9x9_image_ex1.png", blr9x9)
cv2.imwrite("blured_15x15_image_ex1.png", blr15x15)
cv2.imwrite("blured_3x3_median_image_ex1.png", blr3x3median)
cv2.imwrite("blured_9x9_median_image_ex1.png", blr9x9median)
cv2.imwrite("blured_15x15_median_image_ex1.png", blr15x15median)
cv2.imwrite("laplacian_sharpened_blured_3x3_image_ex1.png", finallap)

cv2.waitKey(0)