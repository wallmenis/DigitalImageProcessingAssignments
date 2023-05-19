import cv2

original=cv2.imread("house.tif")

(H, W) = original.shape[:2]
print( H, W )
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "hed_pretrained_bsds.caffemodel")
blob = cv2.dnn.blobFromImage(original, scalefactor=1.0 , swapRB=False, crop=False)
net.setInput(blob)
result = net.forward()
print("Error")
result = cv2.resize(result[0, 0], (W, H))
result = (255 * result).astype("uint8")

cv2.imshow("Original", original)
cv2.imshow("End Result", result)

cv2.waitKey(0)