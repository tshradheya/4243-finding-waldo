import cv2
import numpy as np

def im2single(im):
    im = im.astype(np.float32) / 255
    return im

def single2im(im):
    im *= 255
    im = im.astype(np.uint8)
    return im

def load_image(path):
    return im2single(cv2.imread(path))[:, :, ::-1]

def load_image_gray(path):
    img = load_image(path)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

