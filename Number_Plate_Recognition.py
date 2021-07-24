import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
import os
import csv

#Reading the image

img = cv2.imread('i2.jpg')
gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)


# random variation or brightness reduction and edege detection

noise_reduction = cv2.bilateralFilter(gray , 11, 17 , 17)

edge_detection = cv2.Canny(noise_reduction , 30 , 200)




#Find polygons or rectangles obiviously ... you know number plates ...and then apply mask

keypoints = cv2.findContours(edge_detection.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
rectangles = imutils.grab_contours(keypoints)
rectangles = sorted(rectangles, key=cv2.contourArea, reverse=True)

# Finding the approximate location of the number plate

number_plate_location = None

for rectangle in rectangles :
    approx = cv2.approxPolyDP(rectangle, 10, True)
    if len(approx) == 4:
        number_plate_location= approx
        break


#Highlighting the number plate (MASKING...)

mask = np.zeros(gray.shape,np.uint8)
image = cv2.drawContours(mask,[number_plate_location ],0,255,-1)
image = cv2.bitwise_and(img,img, mask = mask)


# Croping the image

(x,y) = np.where (mask == 255)
(x1,y1) = (np.min(x),np.min(y))
(x2,y2) = (np.max(x),np.max(y))
croped = gray[x1:x2+1 , y1:y2+1]


#Reading the text...

see = easyocr.Reader(['en'])
result = see.readtext(croped)
result

# Final result

text = result [0][-2]



print('--------------------------------------------')
print('--------------------------------------------')
print('--------------------------------------------')
print("The Number :",text)