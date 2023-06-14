import os
import json
import csv
import shutil
import pandas as pd
import cv2
from PIL import Image

i=1

# JSON 파일이 있는 디렉토리 경로
json_dir = r'C:\Users\acorn\Downloads\final\data\Training\label\TL_damage_part\damage_part'
val_json_dir = r'C:\Users\acorn\Downloads\final\data\Validation\label\VL_damage_part\damage_part'
train_dir = r"C:\Users\acorn\Downloads\final\data\Training\image\TS_damage_part\damage_part\\"
val_dir = r'C:\Users\acorn\Downloads\final\data\Validation\image\VS_damage_part\damage_part\\' 
img_save_dir = r"C:\Users\acorn\Desktop\f_pro\part_dmg_img_0613\\"
val_save_dir = r"C:\Users\acorn\Desktop\f_pro\val_part_dmg_img_0613\\"
# 학습 데이터를 저장할 리스트

def saveimg(imagename,image_folder):
    # 이미지 로딩
    image = cv2.imread(imagename)

    # BGR to RGB (OpenCV와 PIL간의 색상 채널 순서 재정렬)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_filename = file_name
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    image_file = os.path.join(image_folder, image_filename)
    Image.fromarray(image_rgb).save(image_file)

# JSON 파일을 순회하며 데이터 추출
for filename in os.listdir(json_dir):
    print('이미지 검색 수',i)

    if filename.endswith(".json"):
        # JSON 파일 경로
        json_path = os.path.join(json_dir, filename)
        print('filename :', filename)
        
        # JSON 파일 읽기
        with open(json_path, "r") as json_file:
            i += 1

            data = json.load(json_file)            
            
            # 파일 이름(url) 추출
            file_name = data['images']['file_name']
            
            # 세그멘테이션 좌표, part 추출
            annotations = data['annotations']
            damage = []
            # print(annotations)

            for annotation in annotations:
                # print(annotation)
                if annotation['damage'] is not None:
                    damage.append(annotation['damage'])

                elif annotation['part'] is not None:
                    part = annotation['part']
                    
            image_folder = img_save_dir + part
            imagename = train_dir + file_name
            damage = set(damage)
            if 'Scratched' in damage and len(damage) == 1:
                image_folder += '_s'
                saveimg(imagename, image_folder)
            else:
                image_folder += '_c'
                saveimg(imagename, image_folder)
            
            damage = []
  
    # 연습용 브레이크
"""
    if i>100:
        break
"""

# 학습 데이터 확인
print ('저장된 training_data 수: ',i)




