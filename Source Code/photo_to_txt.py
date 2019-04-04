import cv2
import numpy as np
from skimage.filters import (threshold_sauvola)

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 320
LAYER_NAMES = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
MIN_CONFIDENCE = 0.5
NET_MODEL_NAME = "east_model.pb"
NO_SCALING_CONST = 1.0


def binarization(img_path, arg):
    img = cv2.imread(img_path, 0)
    if arg == 'ot':
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    elif arg == "atm":
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 127, 1)
    elif arg == 'atg':
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 127, 1)
    else:
        img = custom_threshold(img)
    return img


def custom_threshold(img):
    thr_img = threshold_sauvola(img, window_size=25)
    bin_img = thr_img > img
    return thr_img


def text_detection(img_path):
    img = cv2.imread(img_path)
    orig = img.copy()

    (height, width) = img.shape[:2]
    ratio_w = width / float(IMAGE_WIDTH)
    ratio_h = height / float(IMAGE_HEIGHT)
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    height, width = IMAGE_HEIGHT, IMAGE_WIDTH

    net = cv2.dnn.readNet(NET_MODEL_NAME)
    img_blob = cv2.dnn.blobFromImage(img, NO_SCALING_CONST, (width, height), (123.68, 116.78, 103.94), swapRB=True,
                                     crop=False)
    net.setInput(img_blob)
    (scores, geometry) = net.forward(LAYER_NAMES)

    (num_row, num_col) = scores.shape[2:4]
    rects, confidence, data = [], [], []
    for y in range(0, num_row):
        scores_data = scores[0, 0, y]
        data_0x = geometry[0, 0, y]
        data_1x = geometry[0, 1, y]
        data_2x = geometry[0, 2, y]
        data_3x = geometry[0, 3, y]
        angle_data = geometry[0, 4, y]

        for x in range(0, num_col):
            if scores_data[x] < MIN_CONFIDENCE:
                continue

            (offset_x, offset_y) = (x * 4.0, y * 4.0)
            angle = angle_data[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = data_0x[x] + data_2x[x]
            w = data_1x[x] + data_3x[x]

            end_x = int(offset_x + (cos * data_1x[x]) + (sin * data_2x[x]))
            end_y = int(offset_y - (sin * data_1x[x]) + (cos * data_2x[x]))
            start_x = int(end_x - w)
            start_y = int(end_y - h)

            rects.append((start_x, start_y, end_x, end_y))
            confidence.append(scores_data[x])
            data.clear()

    return rects, confidence
