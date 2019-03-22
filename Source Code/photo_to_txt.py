import cv2
from skimage.filters import (threshold_sauvola)


def binarization(img_name, arg):
    img = cv2.imread(img_name, 0)
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
