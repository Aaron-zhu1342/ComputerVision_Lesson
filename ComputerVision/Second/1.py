import cv2
from tkinter import *
import Harris.harris as harris
import Canny.canny as canny
import numpy as np
import tkinter as tk
from PIL import Image,ImageTk

class Pic:
    url = "4.jpg"

def Canny():
    img = cv2.imread(Pic.url)
    img_g = canny.img_gray(img)
    img_gaussian = canny.gaussian_filter(img_g)
    dst_gradient, dst_seta, dst_x, dst_y = canny.gradient(img_gaussian)
    nms = canny.NMS(dst_gradient, dst_x, dst_y)
    dst_img = canny.Double_threshold(nms)

    #cv2.imwrite('mycanny.jpg', dst_img*255)
    #cv2.imshow('',dst_img)
    #dst_img = cv2.resize(dst_img,(50,150))
    img_result_new = Image.fromarray(dst_img*255)
    img_2 = ImageTk.PhotoImage(img_result_new)
    L7.config(image=img_2,width = 350 ,height = 300)
    L7.image = img_2

    img_new = Image.fromarray(img)
    img_new_2 = ImageTk.PhotoImage(img_new)
    L5.config(image = img_new_2,width = 350 ,height = 300)
    L5.image = img_new_2

    img_open = cv2.Canny(img, 100, 200)
    img_open = Image.fromarray(img_open)
    img_open_2 = ImageTk.PhotoImage(img_open)
    L8.config(image=img_open_2,width = 350 ,height = 300)
    L8.image = img_open_2

def Harris():
    img = cv2.imread(Pic.url)
    img_g = harris.img_gray(img)
    row, col = img.shape[0], img.shape[1]
    grad_all = np.zeros((row, col, 3), np.float)
    grad_x, grad_y = harris.gradient(img_g)
    grad_all[:, :, 0] = grad_x ** 2
    grad_all[:, :, 1] = grad_y ** 2
    grad_all[:, :, 2] = grad_x * grad_y

    grad_all[:, :, 0] = harris.gaussian_filter(grad_all[:, :, 0])
    grad_all[:, :, 1] = harris.gaussian_filter(grad_all[:, :, 1])
    grad_all[:, :, 2] = harris.gaussian_filter(grad_all[:, :, 2])
    grad_all = [np.array([[grad_all[i, j, 0], grad_all[i, j, 2]], [grad_all[i, j, 2], grad_all[i, j, 1]]]) for i in
                range(row) for j in range(col)]

    R = harris.computeR(grad_all)
    corner = harris.computecorner(R, row, col)

    # 将角点标为红色
    img_new1 = np.copy(img)
    for i in range(corner.shape[0]):
        for j in range(corner.shape[1]):
            if corner[i, j] == 255:
                img_new1[i, j] = (255, 0, 0)

    img_result_new = Image.fromarray(img_new1)
    img_2 = ImageTk.PhotoImage(img_result_new)
    L7.config(image=img_2,width = 350 ,height = 300)
    L7.image = img_2

    img_new = Image.fromarray(img)
    img_new_2 = ImageTk.PhotoImage(img_new)
    L5.config(image=img_new_2, width=350, height=300)
    L5.image = img_new_2


    dst = cv2.cornerHarris(img_g, 2, 3, 0.05)
    img_new3 = np.copy(img)
    img_new3[dst > 0.01 * dst.max()] = (255, 0, 0)

    img_open = Image.fromarray(img_new3 )
    img_open_2 = ImageTk.PhotoImage(img_open)
    L8.config(image=img_open_2, width=350, height=300)
    L8.image = img_open_2

if __name__ == "__main__":
    window = tk.Tk()
    window.title('Computer Vision')
    window.geometry('1300x1000')
    var1 = tk.StringVar()

    L1 = tk.Label(window, text='Welcome to use the Computer Vision Program ', font=('Arial', 12), width=50, height=2)
    L1.grid(row=0,column=1)



    L5 = tk.Label(window,width = 50,height = 20)
    L5.grid(row=4, column=0)

    L7 = tk.Label(window,width = 50,height = 20)
    L7.grid(row = 4,column = 1)

    L8 = tk.Label(window,width = 50,height = 20)
    L8.grid(row = 4 , column = 2)
    L3 = tk.Label(window, text='原图:', font=('Arial', 12), width=50, height=1,anchor=tk.W)
    L3.grid(row=3,column = 0)
    L6 = tk.Label(window , text = '我的效果图:',font = ('Arial' , 12),width = 50,height = 1,anchor=tk.W)
    L6.grid(row = 3,column = 1)
    L4 = tk.Label(window, text = '系统的效果图:' , font = ('Arial',12),width = 50 ,height =1 ,anchor=tk.W)
    L4.grid(row = 3,column = 2)

    b1 = tk.Button(window, text='Canny', font=('Arial', 12), width=50, height=1, command=Canny)
    b1.grid(row=1, column  = 0)
    b2 = tk.Button(window, text='Harris', font=('Arial', 12), width=50, height=1, command=Harris)
    b2.grid(row=2, column = 0)
    window.mainloop()
