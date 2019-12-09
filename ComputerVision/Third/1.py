import cv2
from tkinter import *
import SIFT.sift2 as mysift
import numpy as np
import tkinter as tk
from PIL import Image,ImageTk

class Pic:
    url = "4.jpg"

def SIFT():
    sift = cv2.xfeatures2d.SIFT_create()
    img = cv2.imread(Pic.url)
    img_temp = img.copy()
    img_mysift = mysift.run(img)
    img_result_new = Image.fromarray(cv2.cvtColor(img_mysift, cv2.COLOR_BGR2RGB))
    img_crruent = ImageTk.PhotoImage(img_result_new)
    L7.config(image=img_crruent,width = 350 ,height = 300)
    L7.image = img_crruent

    img_result = Image.fromarray(cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB))
    img_crruent1 = ImageTk.PhotoImage(img_result)
    L5.config(image=img_crruent1, width=350, height=300)
    L5.image = img_crruent1

    kp1, des1 = sift.detectAndCompute(img,None)   #des1是描述子
    img3_sift = cv2.drawKeypoints(img_temp,kp1,img_temp,color=(0,0,255))
    img_result_sift = Image.fromarray(cv2.cvtColor(img3_sift, cv2.COLOR_BGR2RGB))
    img_crruent_sift = ImageTk.PhotoImage(img_result_sift)
    L8.config(image=img_crruent_sift, width=350, height=300)
    L8.image = img_crruent_sift
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

    b1 = tk.Button(window, text='SIFT', font=('Arial', 12), width=50, height=1, command=SIFT)
    b1.grid(row=1, column  = 0)
    b2 = tk.Button(window, text='', font=('Arial', 12), width=50, height=1, )
    b2.grid(row=2, column = 0)
    window.mainloop()
