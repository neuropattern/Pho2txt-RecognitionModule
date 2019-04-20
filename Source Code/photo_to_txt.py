import json

import cv2
import numpy as np
import pytesseract
from nms import (felzenszwalb, fast, malisiewicz, nms)
from skimage.filters import (threshold_sauvola)

from filtration import retinex
from utils import utils

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 320
LAYER_NAMES = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
MIN_CONFIDENCE = 0.5
NET_MODEL_NAME = "east_model.pb"
NO_SCALING_CONST = 1.0
TESSERACT_CONFIG = "-l eng --oem 1 --psm 8"


def filtration(img_name):
    src = cv2.imread(img_name, cv2.IMREAD_COLOR)

    if src is None:
        print('Error opening image')
        return -1

    src = cv2.GaussianBlur(src, (3, 3), 0)

    with open('config.json', 'r') as f:
        config = json.load(f)

    img_msrcp = retinex.MSRCP(
        src,
        config['sigma_list'],
        config['low_clip'],
        config['high_clip']
    )

    cv2.imwrite('filtered.png', img_msrcp)
    return img_msrcp


def binarization(img, arg):
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


def get_image_ratio(img):
    (height, width) = img.shape[:2]
    return width / float(IMAGE_WIDTH), height / float(IMAGE_HEIGHT)


def text_recognition(img, boxes):
    (height, width) = img.shape[:2]
    (ratio_w, ratio_h) = get_image_ratio(img)

    boxes = sorted(boxes, key=lambda r: r[1])

    text = ""
    for (start_x, start_y, end_x, end_y) in boxes:
        start_x = int(start_x * ratio_w)
        start_y = int(start_y * ratio_h)
        end_x = int(end_x * ratio_w)
        end_y = int(end_y * ratio_h)

        d_x = int((end_x - start_x) * 0.1)
        d_y = int((end_y - start_y) * 0.1)

        start_x = max(0, start_x - d_x)
        start_y = max(0, start_y - d_y)
        end_x = min(width, end_x + (d_x * 1))
        end_y = min(height, end_y + (d_y * 1))

        roi = img[start_y:end_y, start_x:end_x]
        string = pytesseract.image_to_string(roi, config=TESSERACT_CONFIG) + ' '
        text += string

    return text


def text_detection(img):
    orig = img.copy()
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    height, width = IMAGE_HEIGHT, IMAGE_WIDTH

    net = cv2.dnn.readNet(NET_MODEL_NAME)
    img_blob = cv2.dnn.blobFromImage(img, NO_SCALING_CONST, (width, height), (123.68, 116.78, 103.94), swapRB=True,
                                     crop=False)
    net.setInput(img_blob)
    (scores, geometry) = net.forward(LAYER_NAMES)

    (rects, confidences, baggage) = handle_detection_results(scores, geometry)

    offsets, thetas = [], []
    for i in baggage:
        offsets.append(i['offset'])
        thetas.append(i['angle'])

    functions = [felzenszwalb.nms, fast.nms, malisiewicz.nms]

    for i, function in enumerate(functions):
        indicies = nms.boxes(rects, confidences, nms_function=function, confidence_threshold=MIN_CONFIDENCE,
                             nsm_threshold=0.4)
        indicies = np.array(indicies).reshape(-1)
        draw_rects = np.array(rects)[indicies]

    (ratio_width, ratio_height) = get_image_ratio(orig)
    polygons = utils.rects2polys(rects, thetas, offsets, ratio_width, ratio_height)

    for i, function in enumerate(functions):
        indicies = nms.polygons(polygons, confidences, nms_function=function, confidence_threshold=MIN_CONFIDENCE,
                                nsm_threshold=0.4)
        indicies = np.array(indicies).reshape(-1)
        draw_polys = np.array(polygons)[indicies]

    return rects, confidences


def draw_polygons(img, polygons, ratioWidth, ratioHeight, color=(0, 0, 255), width=1):
    for polygon in polygons:
        pts = np.array(polygon, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, color, width)


def handle_detection_results(scores, geometry):
    (num_row, num_col) = scores.shape[2:4]
    rects, confidence, baggage = [], [], []

    for y in range(0, num_row):
        scores_data = scores[0, 0, y]
        data_top = geometry[0, 0, y]
        data_right = geometry[0, 1, y]
        data_bottom = geometry[0, 2, y]
        data_left = geometry[0, 3, y]
        angle_data = geometry[0, 4, y]

        for x in range(0, num_col):
            if scores_data[x] < MIN_CONFIDENCE:
                continue

            (offset_x, offset_y) = (x * 4.0, y * 4.0)

            upper_right = (offset_x + data_right[x], offset_y - data_top[x])
            lower_right = (offset_x + data_right[x], offset_y + data_bottom[x])
            upper_left = (offset_x - data_left[x], offset_y - data_top[x])
            lower_left = (offset_x - data_left[x], offset_y + data_bottom[x])

            rects.append([int(upper_left[0]), int(upper_left[1]), int(lower_right[0] - upper_left[0]),
                         int(lower_right[1] - upper_left[1])])
            confidence.append(float(scores_data[x]))
            baggage.append({"offset": (offset_x, offset_y), "angle": angle_data[x], "upper_right": upper_right,
                            "lower_right": lower_right, "upper_left": upper_left, "lower_left": lower_left,
                            "data_top": data_top[x], "data_right": data_right[x], "data_bottom": data_bottom[x],
                            "data_left": data_left[x]})

    return rects, confidence, baggage


def show_boundaries(img, boxes):
    (ratio_w, ratio_h) = get_image_ratio(img)
    for (start_x, start_y, end_x, end_y) in boxes:
        start_x = int(start_x * ratio_w)
        start_y = int(start_y * ratio_h)
        end_x = int(end_x * ratio_w)
        end_y = int(end_y * ratio_h)
        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)


def draw_box(img, boxes):
    (ratio_w, ratio_h) = get_image_ratio(img)
    for (x, y, w, h) in boxes:
        start_x = int(x * ratio_w)
        start_y = int(y * ratio_h)
        end_x = int((x + w) * ratio_w)
        end_y = int((y + h) * ratio_h)
        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)


def to_txt(img_name, arg):
    img = cv2.imread(img_name)
    filtration(img)
    binarization(img, arg)
    (rects, confidence) = text_detection(img)
    text = text_recognition(img, rects)
    return text
