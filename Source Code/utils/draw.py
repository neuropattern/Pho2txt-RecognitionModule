import cv2
import numpy as np


def draw_polygons(img, polygons, color=(0, 0, 255), width=1):
    for polygon in polygons:
        polygon = np.array(polygon, np.int32)
        polygon = polygon.reshape((-1, 1, 2))
        cv2.polylines(img, [polygon], True, color, width)
    return img


def draw_boundaries(img, boxes, img_ratio_w, img_ratio_h):
    for (start_x, start_y, end_x, end_y) in boxes:
        start_x = int(start_x * img_ratio_w)
        start_y = int(start_y * img_ratio_h)
        end_x = int(end_x * img_ratio_w)
        end_y = int(end_y * img_ratio_h)
        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    return img


def draw_box(img, boxes, img_ratio_w, img_ratio_h):
    for (x, y, w, h) in boxes:
        start_x = int(x * img_ratio_w)
        start_y = int(y * img_ratio_h)
        end_x = int((x + w) * img_ratio_w)
        end_y = int((y + h) * img_ratio_h)
        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 1)
