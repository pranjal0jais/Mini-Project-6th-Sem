from PIL import Image
import cv2
import numpy as np


def apply_clahe(img):
    img_arr = np.array(img.convert('L'))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(img_arr)
    return Image.fromarray(enhanced_img).convert("RGB")


def extract_features(model, path):
    img = Image.open(path).convert('RGB')
    img = apply_clahe(img)
    return model.get_vec(img, tensor=False)
