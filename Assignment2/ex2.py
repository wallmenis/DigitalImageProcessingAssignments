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

ogblured=cv2.blur(original,(3,3))   # Έχοθμε καλύτερα αποτελέσματα αν κάνουμε blur τις εικόνες μας πρίν για να μπορέσουμε να διώξουμε "παράσιτα"

laplacian=cv2.Laplacian(ogblured,-1)    # Είναι η κλασική λαπλασιανή για 3x3 kernel

# Η υλοποίηση με sobel φίλτρα έγινε σε μία γραμμή όπως βλέπουμε παρακάτω. Έχουμε άρθροισμα 3 3x3 sobel φίλτρων τα οποία κατευθύνονται διαγώνια,
# κάθετα και οριζόντια αντίστοιχα. Πάλι έχουμε μια συνάρτηση αρθροίσματος που μας βοηθάει να μην έχουμε over/underflows. 
sobel=add_no_under_overflow(cv2.Sobel(ogblured,-1,dx=1,dy=1,ksize=3),add_no_under_overflow(cv2.Sobel(ogblured,-1,dx=1,dy=0,ksize=3),cv2.Sobel(ogblured,-1,dx=0,dy=1,ksize=3)))

canny=cv2.Canny(original,95,255) # Τα 90 και 255 είναι τα κάτω και άνω κατώφλια αντίστοιχα. Για την συγκεκριμένη εικόνα είχαμε καλά αποτελέσματα αποφεύγοντας
                                # πολύ κοντινές τιμές.

cv2.imshow("Original", original)
cv2.imshow("Original blured", ogblured)
cv2.imshow("3 (diagonal, horizontal and vertical) Sobel Filter", sobel)
cv2.imshow("Laplacian", laplacian)
cv2.imshow("Canny", canny)

cv2.imwrite("Sobel_ex2.png", sobel)
cv2.imwrite("Laplacian_ex2.png", laplacian)
cv2.imwrite("Canny_ex2.png", canny)

cv2.waitKey(0)