import cv2
import numpy as np
import math
from scipy import ndimage

def img_gray(img):#图像的灰度化
    row,col= img.shape[0],img.shape[1]
    dst = np.zeros((row,col),np.uint8)
    for i in range(row):
        for j in range(col):
            (b,g,r) = img[i,j]
            k = (int(b)+int(g)+int(r))/3
            dst[i,j] = k
    return dst

def gausskernel(size):#计算高斯卷积核
    sigma = 1.5
    kernel = np.ones((size,size),np.float)
    for i in range (size):
        for j in range(size):
            #print((i,j))
            norm = math.pow(i,2)+math.pow(j,2)
            kernel[i,j] = math.exp(-norm/2*pow(sigma,2))  #求高斯卷积
    result = np.sum(kernel)
    kernel = kernel/result
    return kernel

def gaussian_filter(img):#高斯滤波
    row,col = img.shape[0],img.shape[1]
    dst = np.ones((row,col))
    print((row,col))
    count = 0
    kernel = gausskernel(3)
    # for i in range(row-1):
    #     for j in range(col-1):
    #         dst[i,j] = img[i,j]
    for i in range(row):
        for j in range(col):
            sum = 0
            for k in range(-1,2):
                for g in range(-1,2):
                    if i+k >= 0 and j+g >= 0 and i+k <= row-1 and j+g <= col-1:
                        sum = sum + img[i+k,j+g]*kernel[k+1,g+1]
            dst[i,j] = sum
            count = count + 1
    print(count)
    return dst

def gradient(img):             #求梯度与方向
    row,col = img.shape[0],img.shape[1]
    gradx = np.ones((row,col))
    grady = np.ones((row,col))
    dst_Gradient = np.ones((row, col))
    dst_seta = np.ones((row,col))
    sobelx = np.mat([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobely = np.mat([[1,2,1],[0,0,0],[-1,-2,-1]])
    for i in range(row):
        for j in range(col):
            sum_x = 0
            sum_y = 0
            for k in range(-1,2):
                for g in range(-1,2):
                    if i+k >=0 and j+g >=0  and i+k <= row-1 and j+g <= col-1:
                        sum_x = sum_x + img[i+k,j+g] * sobelx[k+1,g+1]
                        sum_y = sum_y + img[i+k,j+g] * sobely[k+1,g+1]
            gradx[i,j] = sum_x
            grady[i,j] = sum_y
            Gradient = math.sqrt(math.pow(sum_x,2)+math.pow(sum_y,2))
            seta = math.atan(sum_y/sum_x)
            dst_Gradient[i,j] = Gradient
            dst_seta[i,j] = seta
    return dst_Gradient ,dst_seta ,gradx,grady

def NMS(dst_Gradient,gradx,grady): #非最大值抑制
    row,col = dst_Gradient.shape[0],dst_Gradient.shape[1]
    NMS = np.copy(dst_Gradient)
    #设置边缘为不可能的分界点
    NMS[0,:] = 0
    NMS[row-1,:] = 0
    NMS[:,0] = 0
    NMS[:,col-1] = 0
    for i in range(1,row-1):
        for j in range(1,col-1):
            if dst_Gradient[i,j] == 0:
                NMS[i,j] = 0
            else:
                dx = gradx[i,j]
                dy = grady[i,j]
                gradtemp = dst_Gradient[i,j] #the gradient of current point
                if np.abs(dy) > np.abs(dx):#y方向幅度大
                    weight = np.abs(dx) / np.abs(dy)
                    grad1 = dst_Gradient[i-1,j]
                    grad2 = dst_Gradient[i+1,j]
                    if dx*dy > 0:
                        grad3 = dst_Gradient[i-1,j-1]
                        grad4 = dst_Gradient[i+1,j+1]
                    else:
                        grad3 = dst_Gradient[i-1,j+1]
                        grad4 = dst_Gradient[i+1,j-1]
                else:#x方向幅度大
                    weight = np.abs(dy)/np.abs(dx)
                    grad1 = dst_Gradient[i,j-1]
                    grad2 = dst_Gradient[i,j+1]
                    if dx*dy > 0:
                        grad3 = dst_Gradient[i+1,j-1]
                        grad4 = dst_Gradient[i-1,j+1]
                    else:
                        grad3 = dst_Gradient[i-1,j-1]
                        grad4 = dst_Gradient[i+1,j+1]
                pixel1 = weight * grad3 + (1-weight) * grad1
                pixel2 = weight * grad4 + (1-weight) * grad2
                if gradtemp >= pixel1 and gradtemp >= pixel2:
                    NMS[i,j] = gradtemp
                else:
                    NMS[i,j] = 0
    return  NMS

def Double_threshold(nms):
    row,col = nms.shape[0],nms.shape[1]
    dst = np.zeros((row,col))
    low_threshold = 0.1 * np.max(nms)
    high_threshold = 0.3 * np.max(nms)
    for i in range(row):
        for j in range(col):
            if nms[i,j] >= high_threshold:
                dst[i,j] = 1
            elif nms[i,j] <= low_threshold:
                dst[i,j] = 0;
            else:
                if i+1 < row and i-1 >= 0 and j+1 <col and j-1 >= 0:
                    if (nms[i-1,j-1:j+1] >= high_threshold).any() or (nms[i+1,j-1:j+1] >= high_threshold).any() or (nms[i,j-1] >= high_threshold) or (nms[i,j+1] >= high_threshold):
                        dst[i,j] = 1
    return dst

if __name__ == "__main__":
    img = cv2.imread("1.jpg")
    img_canny = cv2.Canny(img,100,200)
    img_g = img_gray(img)
    img_gaussian = gaussian_filter(img_g)
    dst_gradient ,dst_seta,dst_x,dst_y= gradient(img_gaussian)
    nms = NMS(dst_gradient,dst_x,dst_y)
    dst_img = Double_threshold(nms)
    cv2.imshow('Original img',img)
    cv2.imshow('My canny img',dst_img)
    cv2.imshow('CV2 canny img',img_canny)
    cv2.waitKey(0)