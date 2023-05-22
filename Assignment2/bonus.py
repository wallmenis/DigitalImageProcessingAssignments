# Ο κώδικας είναι μία μικρή τροποποίηση από το https://www.geeksforgeeks.org/holistically-nested-edge-detection-with-opencv-and-deep-learning/

import cv2
import numpy as np

#def add_no_under_overflow(im1,im2):                     # Πειράματα με ai image sharpening
#    final_image = np.zeros(im1.shape, dtype=np.uint16)
#    final_image += im1
#    final_image += im2
#    final_image[final_image>255] = 255
#    final_image[final_image<0] = 0
#    final_image=np.array(final_image,dtype=np.uint8)
#    return final_image


original=cv2.imread("house.tif")

(H, W) = original.shape[:2] # Αφαιρούμε την 3η διάσταση (γίνoνται 512x512 τα Η και W αντίστοιχα)
#print( H, W )
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "hed_pretrained_bsds.caffemodel") # Φορτώνουμε τα μοντέλα στο νευρωνικό δίκτυο
blob = cv2.dnn.blobFromImage(original, scalefactor=1.0 , swapRB=False, crop=False)  # Μετατρέπουμε τα δεδομένα για να εισαχθούν στο νευρωνικό δίκτυο
net.setInput(blob)  # Βάζουμε τα δεδομένα σε αυτό
result = net.forward()  # Αποθυκεύουμε το αποτέλεσμα
#print("Error")
#print (result.shape)
result = cv2.resize(result[0, 0], (W, H))   # Μετατροπή από 1x1x512x512 σε 512x512 (είναι ο τρόπος εξόδου του δικτυου και κάνουμε μετατροπή και όχι κανονικό resize)
#print (result.shape)
result = (255 * result).astype("uint8")     # Το αποτέλεσμα είναι τιμές 0 έως 1 και έτσι πρέπει να το κάνουμε normalize

#resultsharp=add_no_under_overflow(result,cv2.cvtColor(original,cv2.COLOR_BGR2GRAY))

cv2.imshow("Original", original)
cv2.imshow("End Result", result)

#cv2.imshow("Sharpened",resultsharp)

cv2.imwrite("End_Result_bonus.png", result)

cv2.waitKey(0)