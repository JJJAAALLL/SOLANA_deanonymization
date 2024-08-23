import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO
import cv2
#%%
# Load a YOLOv8n PyTorch model
model = YOLO("detection1.pt",task='obb')

# Export the model to NCNN format
model.export(format="tflite")  # creates 'yolov8n_ncnn_model'
#%%

#%%
tflite_model_path='C:/Users/danin/Desktop/Jalal/python/Data_India_1/detection1_saved_model/detection1_float32.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
x_test=Image.open('C:/Users/danin/Desktop/Jalal/python/Data_India_1/bottom.png').resize((640, 640))
input_data = np.expand_dims(x_test, axis=0).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
#%%
boxx=[]
leng=output_data[0][0].size
for i in range(leng):
    check=output_data[:,:,i]
    if check[0][4]<0.1:
        continue
    boxx.append(check[0])
#%%
import numpy as np

def rotate_point(cx, cy, angle, px, py):
    s = np.sin(angle)
    c = np.cos(angle)

    # Translate point back to origin
    px -= cx
    py -= cy

    # Rotate point
    xnew = px * c - py * s
    ynew = px * s + py * c

    # Translate point back
    px = xnew + cx
    py = ynew + cy
    return px, py

def xywhr_to_xyxyxyxy(bboxes):
    converted_bboxes = []
    for bbox in bboxes:
        x,y, w, h, c, r = bbox

        # Calculate the half width and height
        half_w = w / 2.0
        half_h = h / 2.0

        # Calculate the corner points before rotation
        top_left = (x - half_w, y - half_h)
        top_right = (x + half_w, y - half_h)
        bottom_right = (x + half_w, y + half_h)
        bottom_left = (x - half_w, y + half_h)

        # Rotate the corner points around the center (x, y)
        top_left = rotate_point(x, y, r, *top_left)
        top_right = rotate_point(x, y, r, *top_right)
        bottom_right = rotate_point(x, y, r, *bottom_right)
        bottom_left = rotate_point(x, y, r, *bottom_left)

        # Append the converted bounding box to the list
        converted_bboxes.append([
            top_left[0], top_left[1],
            top_right[0], top_right[1],
            bottom_right[0], bottom_right[1],
            bottom_left[0], bottom_left[1]
        ])
    
    return converted_bboxes

# Example usage
bboxes = boxx

converted_bboxes = xywhr_to_xyxyxyxy(bboxes)
# for bbox in converted_bboxes:
    # print(bbox)
#%%
def draw_boxes(image, bboxes):
    xl,yl=image.shape[0],image.shape[1]
    for bbox in bboxes:
        points = np.array([
            [bbox[0]*yl, bbox[1]*xl],
            [bbox[2]*yl, bbox[3]*xl],
            [bbox[4]*yl, bbox[5]*xl],
            [bbox[6]*yl, bbox[7]*xl]
        ], np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)#%%
image=cv2.imread('C:/Users/danin/Desktop/Jalal/python/Data_India_1/bottom.png')
draw_boxes(image, converted_bboxes)

#%%
cv2.imwrite('check.png',image)
#%%
model=YOLO('detection1.pt')
results=model('C:/Users/danin/Desktop/Jalal/python/Data_India_1/1mUdYQCIrr.jpg')
#%%
r=results[0].obb.xyxyxyxy