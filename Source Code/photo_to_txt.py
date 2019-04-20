import json

import cv2
import numpy as np
import pytesseract
from nms import (felzenszwalb, nms)
from skimage.filters import (threshold_sauvola)

from filtration import retinex
from utils import constant
from utils import utils


class PhotoToTxt:

    def __init__(self, img, bin_arg='atm'):
        self.img = cv2.imread(img)
        self.filtered_img = self.img.copy()
        self.bin_img = self.img.copy()
        self.bin_arg = bin_arg
        self.img_width, self.img_height = self.img.shape[:2]

    def filtration(self):
        mod_img = self.img.copy()

        mod_img = cv2.GaussianBlur(mod_img, (3, 3), 0)

        with open('filtration/config.json', 'r') as f:
            config = json.load(f)

        self.filtered_img = retinex.MSRCP(mod_img, config['sigma_list'], config['low_clip'], config['high_clip'])
        cv2.imwrite('FilteredImage.png', mod_img)

    def binarization(self):
        if self.bin_arg == 'ot':
            self.bin_img = cv2.threshold(self.filtered_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        elif self.bin_arg == "atm":
            self.bin_img = cv2.adaptiveThreshold(self.filtered_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                                 127, 1)
        elif self.bin_arg == 'atg':
            self.bin_img = cv2.adaptiveThreshold(self.filtered_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, 127, 1)
        else:
            self.bin_img = self.custom_threshold(self.filtered_img)
        cv2.imwrite('BinarizationImage.png', self.bin_img)

    @staticmethod
    def custom_threshold(filtered_img):
        thr_img = threshold_sauvola(filtered_img, window_size=25)
        bin_img = thr_img > filtered_img
        return bin_img

    def get_image_ratio(self):
        return self.img_width / float(constant.IMAGE_WIDTH), self.img_height / float(constant.IMAGE_HEIGHT)

    def text_recognition(self, boxes):
        (ratio_w, ratio_h) = self.get_image_ratio()

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
            end_x = min(self.img_width, end_x + (d_x * 1))
            end_y = min(self.img_height, end_y + (d_y * 1))

            roi = self.img[start_y:end_y, start_x:end_x]
            string = pytesseract.image_to_string(roi, config=constant.TESSERACT_CONFIG) + ' '
            text += string

        return text

    def text_detection(self):
        img_copy = cv2.resize(self.img, (constant.IMAGE_WIDTH, constant.IMAGE_HEIGHT))
        self.img_height, self.img_width = constant.IMAGE_HEIGHT, constant.IMAGE_WIDTH
        (ratio_width, ratio_height) = self.get_image_ratio()

        net = cv2.dnn.readNet(constant.NET_MODEL_NAME)
        img_blob = cv2.dnn.blobFromImage(img_copy, constant.NO_SCALING_CONST, (self.img_width, self.img_height),
                                         (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(img_blob)
        (scores, geometry) = net.forward(constant.LAYER_NAMES)

        (rects, confidences, baggage) = self.handle_detection_results(scores, geometry)

        offsets, thetas = [], []
        for i in baggage:
            offsets.append(i['offset'])
            thetas.append(i['angle'])

        indicies = nms.boxes(rects, confidences, nms_function=felzenszwalb.nms,
                             confidence_threshold=constant.MIN_CONFIDENCE, nsm_threshold=0.4)
        indicies = np.array(indicies).reshape(-1)
        draw_rects = np.array(rects)[indicies]

        polygons = utils.rects2polys(rects, thetas, offsets, ratio_width, ratio_height)

        indicies = nms.polygons(polygons, confidences, nms_function=felzenszwalb.nms,
                                confidence_threshold=constant.MIN_CONFIDENCE, nsm_threshold=0.4)
        indicies = np.array(indicies).reshape(-1)
        draw_polys = np.array(polygons)[indicies]

        return rects, confidences

    @staticmethod
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
                if scores_data[x] < constant.MIN_CONFIDENCE:
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
