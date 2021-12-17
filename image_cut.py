from PIL import Image
import openpyxl
import cv2
from datetime import date
import openpyxl 
import shutil
import os

def main(Exc_Location):
    img = cv2.imread('./data/central.jpg')
    shutil.rmtree('./cut_img')
    os.makedirs('./cut_img')

    wb = openpyxl.load_workbook(Exc_Location)
    ws = wb["bounding_boxes"]
    object_cnt = ws.cell(1,8).value


    for i in range(2,object_cnt+2):
        cropped_img = img[ws.cell(i,3).value:ws.cell(i,4).value,
            ws.cell(i,5).value:ws.cell(i,6).value].copy()
        cv2.imwrite('./cut_img/'+str(i-1)+'.jpg',cropped_img)
    
    wb.close()


if __name__ == '__main__':    
    tday = date.today()
    tday_w = tday.strftime('%Y-%b-%d-%A')
    Exc_Location = './excel/result_excel'+"("+tday_w+")"+'.xlsx'
    main(Exc_Location)