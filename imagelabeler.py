import os
from PIL import Image
from numpy import asarray
import numpy as np
import string
import cv2 as cv
import sys

def brightness(img, value=30):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv.merge((h, s, v))
    img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    return img

for i in range (53,90):
    path = 'coordinate' + str(i) + '.txt'
    #text = open(path, 'r')
    i2 = str(i)
    if(i < 10):
        imgpath = 'diabetes research/ddb1_v02_01/images/diaretdb1_image00' + i2 + '.png'
    else:
        imgpath = 'diabetes research/ddb1_v02_01/images/diaretdb1_image0' + i2 + '.png'
    img = cv.imread(imgpath)
    if img is None:
        sys.exit('Could not load image.')
    with open(path) as file:
        lines = file.readlines()
    type = ''
    hard_count = 1
    soft_count = 1
    for line in lines:
        if(line[:-1] == 'hard_exudates:' or line[:-1] == 'soft_exudates:' or line[:-1] == 'all_coordinates:'):
            type = line[:-1]
        if(type == 'hard_exudates:' and line[:-1] != 'hard_exudates:' ):

            line = line.rstrip()
            coords = line.split(',')
            x = int(coords[0])
            y = int(coords[1])
            x2 = x-13
            y2 = y-13
            w = 26
            h = 26
            if(x2+w > img.shape[1]):
                w = img.shape[1] - x2
            if(y2+h > img.shape[0]):
                h = img.shape[0] - y2
            #print('width:' + str(img.shape[1]))
            #print('w' + str(w))
            #print('height:' + str(img.shape[0]))
            #print('h' + str(h))
            #print('x2' + str(x2))
            #print('y2' + str(y2))
            print(i2)
            crop = img[y2:y2+h,x2:x2+w]
            #img = cv.rectangle(img, (x-13,y+13),(x+13,y-13),(0,0,255),1)
            #img = cv.putText(img, type, (x-13, y-19), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
            crop = crop[:, :, 1]
            cv.imwrite("Hard_exudates_img_" + i2 + "_" + str(hard_count)+".png", crop)
            imgstr = "Hard_exudates_img_" + i2 + "_" + str(hard_count) + ".png"
            #cv.imshow(imgstr, crop)
            hard_count = hard_count + 1
        elif(type == 'soft_exudates:' and line[:-1] != 'soft_exudates:'):
            line = line.rstrip()
            coords = line.split(',')
            x = int(coords[0])
            y = int(coords[1])
            x2 = x - 13
            y2 = y - 13
            w = 26
            h = 26
            crop = img[y2:y2 + h, x2:x2 + w]
            #img = cv.rectangle(img, (x - 13, y + 13), (x + 13, y - 13), (0, 255, 0), 1)
            #img = cv.putText(img, type, (x - 13, y - 19), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
            crop = crop[:,:,1]
            cv.imwrite("Soft_exudates_img_" + i2 + "_" + str(soft_count) +".png", crop)
            imgstr = "Soft_exudates_img_" + i2 + "_" + str(soft_count) + ".png"
            #cv.imwrite(imgstr, img)
            soft_count = soft_count + 1
        elif(type == 'all_coordinates:' and line[:-1] != 'all_coordinates:'):
            line = line.rstrip()
            coords = line.split(',')
            x = int(coords[0])
            y = int(coords[1])
            #img = cv.rectangle(img, (x - 13, y + 13), (x + 13, y - 13), (255, 0, 0), 1)
            #img = cv.putText(img, 'other', (x - 13, y - 19), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)
    #low_green = np.array([1,168,168])
    #high_green = np.array([244,255,255])
    #img2 = brightness(img, 15)
    #img2 = cv.convertScaleAbs(img, 3,13)
    #img2 = cv.bilateralFilter(img,9,75,75)

    #img2 = cv.GaussianBlur(img2, (5,5),55)
    #img2 = cv.fastNlMeansDenoisingColored(img2, None, 10, 10, 7, 21)
    #_, img2 = cv.threshold(img2,167,245,cv.THRESH_BINARY_INV)
    #hsv = cv.cvtColor(img2, cv.COLOR_BGR2HSV)

    #mask = cv.inRange(hsv,low_green,high_green)
    #cv.imshow('mask', mask)
    #cv.imshow('mask', img2)
    #cv.imshow('Display window', img)
    #k = cv.waitKey(0) #how long to wait for user input, 0 means forever

    #if k == ord("s"): #if key pressed (k) = the "s" key, image is saved


