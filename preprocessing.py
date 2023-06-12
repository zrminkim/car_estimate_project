import pandas as pd
import cv2
import numpy as np
import os
import json
import csv
import matplotlib.pyplot as plt

# import tensorflow as tf
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, BatchNormalization, Activation
car_data = "car_data.csv"
df = pd.read_csv(car_data) # image명, segmentation, bbox, part 내용 저장 dataframe
REAL_PATH = '/mnt/d/carcrushdata/Data/Training/realdata/damage_part'
OUTER_PATH = "/mnt/d/carcrushdata/Data/Training/realdata/outer"

def process_annotations(df,realPath,outerPath): 
    i=0
    for idx, row in df.iterrows(): # DataFrame 내 한 행씩 불러와서 작업
        file_name = row['File_Name']
        print(file_name)
        bbox = row['BBox'] # str 형식이다.
        bbox = json.loads(bbox) # json 형식 데이터를 python 객체로 읽기

        car_image = cv2.imread(os.path.join(realPath,file_name))
        image_rgb = cv2.cvtColor(car_image, cv2.COLOR_BGR2RGB)

        cimage = image_rgb[0:600,0:800]
        
        if cimage is not None: # 이미지가 저장 경로에 있을 때 이미지 저장 및 출력
            # for seg in segmentation:
            #     for polygon in seg:
            #         segment = np.array(polygon, dtype=np.int32)
            #         segment = segment.reshape((-1, 1, 2))  # Reshape the segment array to have shape (N, 1, 2)
            x, y, wx, hy = [int(i) for i in bbox] # bbox 좌표 뽑기 좌상단 x, 좌상단 y, bouding box의 W, bounding box의 H
            crop_image =  cimage[y:y+hy, x:x+wx]
            if crop_image.size > 0:
                crop_image_rgb = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
                save_path = os.path.join(outerPath,file_name)            
                cv2.imwrite(save_path,crop_image_rgb)
                i += 1
                print(i,"Image saved successfully.")
            else:
                print("Crop image is empty.")
            # cv2.imshow('Cropped Image', crop_image_rgb)
            # cv2.waitKey(0)  # Wait for a key press to close the window
            # cv2.destroyAllWindows()


process_annotations(df,REAL_PATH,OUTER_PATH)
print('hello')