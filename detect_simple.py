import tensorflow as tf
import core.utils as utils
from tensorflow.python.saved_model import tag_constants
import cv2
import numpy as np
import openpyxl 
from datetime import date
import os #모듈 import 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #enable 오류
MODEL_PATH = './checkpoints/yolov4-416' # 모델 기본 위치
IOU_THRESHOLD = 0.45  # 한계점 설정
SCORE_THRESHOLD = 0.40 # 최소 점수 만족
INPUT_SIZE = 416 #사진 크기

# load model
saved_model_loaded = tf.saved_model.load(MODEL_PATH, tags=[tag_constants.SERVING]) # 학습된 모델 부르기
infer = saved_model_loaded.signatures['serving_default'] #시그니쳐 키 설정

def main(img_path):
    
    img = cv2.imread(img_path) # img road
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # bgr 형식 rgb로 convert

    img_input = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE)) # 크기 재설정
    img_input = img_input / 255. # 크기 재설정
    img_input = img_input[np.newaxis, ...].astype(np.float32) # 축 설정
    img_input = tf.constant(img_input) # numpy를 tf 로 변환

    pred_bbox = infer(img_input) #  모델에 img 추가

    wb = openpyxl.Workbook()
    wb.active.title = "bounding_boxes" #excel file 설정
    wsheet = wb["bounding_boxes"]
    tday = date.today()
    tday_w = tday.strftime('%Y-%b-%d-%A')
    wsheet.cell(1,1).value = "classes"
    wsheet.cell(1,2).value = "score"
    wsheet.cell(1,3).value = "UL"
    wsheet.cell(1,4).value = "UR"
    wsheet.cell(1,5).value = "DL"
    wsheet.cell(1,6).value = "DR"


    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression( #nms 진행 
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=IOU_THRESHOLD,
        score_threshold=SCORE_THRESHOLD
    )
    #iou, score 넘는 애들만 체크 

    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    result,result_exc = utils.draw_bbox(img, pred_bbox) #바운딩 박스 추가
    for i in range(len(result_exc)):
        for j in range(len(result_exc[i])):
            wsheet.cell(i+2,j+1).value = result_exc[i][j]
    wsheet.cell(1,8).value = len(result_exc)
    

    result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR) # 재변환
    cv2.imwrite('res.png', result) # 사진 저장
    new_ex_file = './excel/result_excel'+"("+tday_w+")"+'.xlsx'
    wb.save(new_ex_file)

if __name__ == '__main__':
    img_path = './data/central.jpg'
    main(img_path)