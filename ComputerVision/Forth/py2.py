import cv2
from tkinter import *
import Forth.py4 as myHough
import numpy as np
import tkinter as tk
from PIL import Image,ImageTk

class Pic:
    url = "7.jpg"

def Hough():
    img_y = cv2.imread(Pic.url)
    img = myHough.run()

    img_new = Image.fromarray(img_y)
    img_new_2 = ImageTk.PhotoImage(img_new)
    L5.config(image=img_new_2, width=350, height=300)
    L5.image = img_new_2

    img_n = Image.fromarray(img)
    img_n_2 = ImageTk.PhotoImage(img_n)
    L7.config(image=img_n_2, width=350, height=300)
    L7.image = img_n_2

    img_k = cv2.imread(Pic.url)
    img_k_gray = cv2.cvtColor(img_k,cv2.COLOR_BGR2GRAY)
    img_k_canny = cv2.Canny(img_k_gray,50,200)
    lines1 = cv2.HoughLines(img_k_canny,1,np.pi/180,160)
    #print(lines[0])
    lines = lines1[:,0,:]
    for r , angle in lines[:]:
        a = np.cos(angle)
        b = np.sin(angle)
        x0 = a*r
        y0 = b*r
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img_k, (x1, y1), (x2, y2), (255, 0, 0), 2)

    img_open = Image.fromarray(img_k)
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

    b1 = tk.Button(window, text='Hough', font=('Arial', 12), width=50, height=1, command=Hough)
    b1.grid(row=1, column  = 0)
    b2 = tk.Button(window, text='', font=('Arial', 12), width=50, height=1, )
    b2.grid(row=2, column = 0)
    window.mainloop()
