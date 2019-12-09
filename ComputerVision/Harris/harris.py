import cv2
import numpy as np
import math


k = 0.05
threshold = 0.01

def img_gray(img):#图像的灰度化
    row,col= img.shape[0],img.shape[1]
    dst = np.zeros((row,col),np.uint8)
    for i in range(row):
        for j in range(col):
            (b,g,r) = img[i,j]
            k = (int(b)+int(g)+int(r))/3
            dst[i,j] = k
    return dst

def gradient(img):             #用Sobel算子求Ix，Iy梯度
    row,col = img.shape[0],img.shape[1]
    gradx = np.ones((row,col))
    grady = np.ones((row,col))
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
    return gradx,grady


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
    return dst

def computeR(grad): #计算局部特征结果矩阵M的特征值和响应函数R
    D,T = list(map(np.linalg.det,grad)),list(map(np.trace,grad))
    R = np.array([i - k * j ** 2 for i, j in zip(D, T)])
    return R

def computecorner(R,row,col): # 找出角点
    R_max = np.max(R)
    corner = np.zeros((row,col),np.float)
    R = R.reshape((row,col))
    for i in range(row):
        for j in range(col):
            if R[i,j] > R_max * threshold:
                corner[i,j] = 255
    return corner

if __name__ == "__main__":
    img = cv2.imread("4.jpg")
    img_g = img_gray(img)
    row ,col = img.shape[0],img.shape[1]
    grad_all = np.zeros((row,col,3),np.float)
    grad_x,grad_y = gradient(img_g)
    grad_all[:,:,0] = grad_x ** 2
    grad_all[:,:,1] = grad_y ** 2
    grad_all[:,:,2] = grad_x * grad_y

    # print(grad_x)
    # print(grad_x_2)
    grad_all[:,:,0] = gaussian_filter(grad_all[:,:,0])
    grad_all[:,:,1] = gaussian_filter(grad_all[:,:,1])
    grad_all[:,:,2] = gaussian_filter(grad_all[:,:,2])
    grad_all = [np.array([[grad_all[i, j, 0], grad_all[i, j, 2]], [grad_all[i, j, 2], grad_all[i, j, 1]]]) for i in range(row) for j in range(col)]

    R = computeR(grad_all)
    corner = computecorner(R,row,col)

    #将角点标为红色
    img_new1 = np.copy(img)
    for i in range(corner.shape[0]):
        for j in range(corner.shape[1]):
            if corner[i,j] == 255:
                img_new1[i,j] = (0,0,255)

    print(corner)
    dst = cv2.cornerHarris(img_g,2,3,0.05)
    img_new2 = np.copy(img)
    img_new2[dst > 0.01 * dst.max()] = (0, 0, 255)

    cv2.imshow('Original photo',img)
    cv2.imshow('My Harris corner detect',img_new1)
    cv2.imshow('Cv2 Harris corner detect', img_new2)
    cv2.waitKey(0),