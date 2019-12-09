from tkinter import *
import tkinter as tk
import cv2
import numpy as np
from PIL import Image,ImageTk
from matplotlib import pyplot as plt
class Pic:
    url = "1.jpg"

def FLY_go():
    plt.figure(figsize=(3, 3))
    img = cv2.imread(Pic.url,0)
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    fshift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(fshift[:,:,0],fshift[:,:,1]))
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.savefig('2D傅里叶.jpg',bbox_inches='tight')
    img = Image.open('2D傅里叶.jpg')
    img_fly = ImageTk.PhotoImage(img)#创建PhotoImage类的对象，让Label可以显示它
    L5.config(image=img_fly)
    L5.image = img_fly

def JZLB_go():
    img = cv2.imread(Pic.url)
    img_mean = cv2.blur(img,(5,5))
    img_1 = Image.fromarray(cv2.cvtColor(img_mean,cv2.COLOR_BGR2RGB))
    img_2 = ImageTk.PhotoImage(img_1)
    L5.config(image = img_2)
    L5.image = img_2

def ZZLB_go():
    img = cv2.imread(Pic.url)
    img_median = cv2.medianBlur(img, 5)
    img_1 = Image.fromarray(cv2.cvtColor(img_median,cv2.COLOR_BGR2RGB))
    img_2 = ImageTk.PhotoImage(img_1)
    L5.config(image = img_2)
    L5.image = img_2

def KYS_go():
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    original_img = cv2.imread(Pic.url)
    img_open = cv2.morphologyEx(original_img, cv2.MORPH_OPEN, element,iterations=1) #开运算
    img_open_new = Image.fromarray(cv2.cvtColor(img_open,cv2.COLOR_BGR2RGB))#转化为tif
    img_open1 = ImageTk.PhotoImage(img_open_new)
    L5.config(image = img_open1)
    L5.image = img_open1

def BYS_go():
    element = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    original_img  = cv2.imread(Pic.url)
    img_closed = cv2.morphologyEx(original_img,cv2.MORPH_CLOSE,element,iterations=1)#闭运算
    img_closed_new = Image.fromarray(cv2.cvtColor(img_closed,cv2.COLOR_BGR2RGB))
    img_closed1 = ImageTk.PhotoImage(img_closed_new)
    L5.config(image = img_closed1)
    L5.image = img_closed1

def JHH_go():
    img = cv2.imread(Pic.url)
    (b , g , r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    img_result = cv2.merge((bH,gH,rH))
    img_result_new = Image.fromarray(cv2.cvtColor(img_result,cv2.COLOR_BGR2RGB))
    img_crruent = ImageTk.PhotoImage(img_result_new)
    L5.config(image = img_crruent)
    L5.image = img_crruent

def SCTP_go():
    print('qq')

def main():
    L1 = tk.Label(window, text='Welcome to use the Computer Vision Programmer ', font=('Arial', 12), width=60, height=2)
    L1.grid()
    L2 = tk.Label(window, text='原图:', font=('Arial', 12), width=5, height=1)
    L2.grid(row=1, sticky=E)
    img_o = cv2.imread(Pic.url)
    img = Image.fromarray(cv2.cvtColor(img_o,cv2.COLOR_BGR2RGB))
    ph = ImageTk.PhotoImage(img) ###一定要有这句话，不然系统会将img当做垃圾回收掉，这是保存它的一个引用！！！！！！！！！（在这里卡了1个小时）
    L4 = Label(window, image=ph)
    L4.grid(row=1, column=4, sticky=E)

    global L5
    L5 = Label(window)
    L5.grid(row=9, column=4, sticky=E)

    L3 = tk.Label(window, text='效果图:', font=('Arial', 12), width=5, height=1)
    L3.grid(row=8, sticky=E)
    b1 = tk.Button(window, text='2D傅里叶变换', font=('Arial', 12), width=15, height=1, command=FLY_go)
    b1.grid(row=2, sticky=W)
    b2 = tk.Button(window, text='均值滤波', font=('Arial', 12), width=15, height=1, command=JZLB_go)
    b2.grid(row=3, sticky=W)
    b3 = tk.Button(window, text='中值滤波', font=('Arial', 12), width=15, height=1, command=ZZLB_go)
    b3.grid(row=4, sticky=W)
    b4 = tk.Button(window, text='形态学滤波(开)', font=('Arial', 12), width=15, height=1, command=KYS_go)
    b4.grid(row=5, sticky=W)
    b5 = tk.Button(window, text='形态学滤波(闭)', font=('Arial', 12), width=15, height=1, command=BYS_go)
    b5.grid(row=6, sticky=W)
    b6 = tk.Button(window, text='图像均衡化', font=('Arial', 12), width=15, height=1, command=JHH_go)
    b6.grid(row=7, sticky=W)
    b7 = tk.Button(window, text='读取图片', font=('Arial', 12), width=15, height=1, command=SCTP_go)
    b7.grid(row=8, sticky=W)
    window.mainloop()

window = tk.Tk()
window.title('Computer Vision')
window.geometry('800x700')
var1 = tk.StringVar()

main()





