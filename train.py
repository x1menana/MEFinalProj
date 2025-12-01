import cv2
import os
from ultralytics import YOLO
# import random
import json
# print(cv2.__version__)

# declare necessary dictionaries
training_img_dir = 'data/images/train'
training_labels = 'data/labels/train'
categories_json = 'data/labels/classes_to_id.json'
yolo_model_path = 'yolo_models/yolov8n.pt'

def main():
    # obtain dictionary for classes
    categories_json = 'data/labels/classes_to_id.json'
    with open(categories_json, 'r') as file:
        class_dict = json.load(file)
    # print(f'classes: {class_dict}')

    # declare yolo model to finetune
    model = YOLO('yolov8n.pt')

    # fine tune TODO: increase epochs
    model.train(data='data.yaml', epochs=5, device='mps') # enable gpu
    pass

if __name__ == "__main__":
    main()