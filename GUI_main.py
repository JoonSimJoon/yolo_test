from tkinter import *
from datetime import date
import detect_simple 

tday = date.today()
tday_w = tday.strftime('%Y-%b-%d-%A')
Exc_Location = './excel/result_excel'+"("+tday_w+")"+'.xlsx' #엑셀 링크
img_path = './data/central.jpg'

def init():
    root = Tk()
    root.title("GUI")
    root.geometry("1280x720+200+200")
    #btn_detect = Button(root,padx=30,pady=10, text="detect",command=detect_simple.img_detect(img_path))
    #btn_detect.pack()
    #btn_cut = Button(root,padx=30,pady=10, text="cut",command=detect_simple.img_cut(Exc_Location))
    #btn_cut.pack()
    btn_tst = Button(root,text="test",command=detect_simple.__name__)
    btn_tst.pack();
    root.mainloop()

if __name__ == '__main__':
    init()