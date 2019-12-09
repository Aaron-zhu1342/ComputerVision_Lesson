import cv2
import numpy as np
import math

def Hough(img,img_a):
    img_canny = cv2.Canny(img,100,200)
    row = img_canny.shape[0]
    col = img_canny.shape[1]
    temp = []
    for i in range(row):
        for j in range(col):
            if img_canny[i,j] == 255:
                temp.append((i,j))

    contours = []
    for k in temp:
        x , y = k
        for angle in range(-90,90,10):
            r = x * math.cos((angle/180) * math.pi) + y * math.sin((angle / 180)*math.pi)
            contours.append((int(r),int(angle)))
    #print(contours[0][0])
    dst = []
    for g in contours: #voting
        count = 0
        row1 ,col1 = g
        for t in contours:
            if g == t:
                count = count + 1
        dst.append((row1,col1,count))
    #max_r , max_angle,max_vote
    contours_dst = []
    for temp_g in dst:
        if temp_g[2] > 80:
            #contours_dst.append((temp_g[0],temp_g[1],temp_g[2]))
            x0 = math.cos((temp_g[1] / 180) * math.pi) * temp_g[0]
            y0 = math.sin((temp_g[1] / 180) * math.pi) * temp_g[0]
            x1 = int(x0 + 1000 * (-math.cos((temp_g[1] / 180) * math.pi)))
            y1 = int(y0 + 1000 * (math.sin((temp_g[1] / 180) * math.pi)))
            x2 = int(x0 - 1000 * (-math.cos((temp_g[1] / 180) * math.pi)))
            y2 = int(y0 - 1000 * (math.sin((temp_g[1] / 180) * math.pi)))
            cv2.line(img_a, (x1, y1), (x2, y2), (255, 0, 0), 1)
    return img_a

def run():
    img_g = cv2.imread("7.jpg",0)
    img = cv2.imread("7.jpg")
    img_k = Hough(img_g,img)
    return img_k