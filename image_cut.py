from PIL import Image
import openpyxl
import cv2

INPUT_SIZE = 416 #사진 크기

image1 = cv2.imread('./data/central.jpg')
#image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)

#img_input = cv2.resize(image1,(INPUT_SIZE,INPUT_SIZE))
img_input = image1
cropped_img = img_input[267:343,410:460].copy()


#cropped_img.show()

cv2.imwrite('./cut_img/1.jpg',cropped_img)
