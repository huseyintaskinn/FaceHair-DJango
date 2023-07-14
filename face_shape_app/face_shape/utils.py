import requests
from PIL import Image
import numpy as np
from io import BytesIO
from .models import FaceShape
import os
from django.conf import settings
from keras.models import load_model

model = load_model("media/face_shape_model.h5")

# Resim analizi fonksiyonu
def analyze_face_shape(image_file):
    # Resmi açma ve boyutlandırma
    image = Image.open(image_file)
    image = image.resize((64, 64))
    
    # Resmi diziye dönüştürme
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0
    
    # Yüz şeklini tahmin etme
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    
    # Sınıf etiketini dönüştürme
    face_shapes = ["Kalp", "Kare", "Oval", "Uzun", "Yuvarlak"]
    predicted_shape = face_shapes[predicted_class]

    return predicted_shape

from PIL import Image
import cv2
import dlib

import numpy
from datetime import datetime

PREDICTOR_PATH = "media/shape_predictor_68_face_landmarks.dat"

DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor(PREDICTOR_PATH)

def get_landmarks(im):
    rects = DETECTOR(im, 1)
    
    if len(rects) > 1:
        raise Exception("Too many faces")
    if len(rects) == 0:
        raise Exception("Not enough faces")

    return numpy.matrix([[p.x, p.y] for p in PREDICTOR(im, rects[0]).parts()])

def transformation_from_points(points1, points2):
    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T @ points2)
    R = (U @ Vt).T

    return numpy.hstack([(s2 / s1) * R,
                        (c2.T - (s2 / s1) * R @ c1.T)])

def create_mask(points, shape, face_scale):
    groups = [
    # indices of brow and eye landmarks
    [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
    # indices of mouth and nose landmarks
    [27, 28, 29, 30, 31, 32, 33, 34, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
    ]
    mask_im = numpy.zeros(shape, dtype=numpy.float64)
    for group in groups:
        cv2.fillConvexPoly(mask_im,
                        cv2.convexHull(points[group]),
                        color=(1, 1, 1))
    feather_amount = int(0.2 * face_scale * 0.5) * 2 + 1
    kernel_size = (feather_amount, feather_amount)
    mask_im = (cv2.GaussianBlur (mask_im, kernel_size, 0) > 0) * 1.0 # dilate
    mask_im = cv2.GaussianBlur(mask_im, kernel_size, 0) #blur 
    
    return mask_im

def correct_colours (warped_face_im, body_im, face_scale):
    blur_amount = int(3 * 0.5 * face_scale) * 2 + 1
    kernel_size = (blur_amount, blur_amount)

    face_im_blur = cv2.GaussianBlur (warped_face_im, kernel_size, 0)
    body_im_blur = cv2.GaussianBlur (body_im, kernel_size, 0)

    return numpy.clip(0. + body_im_blur + warped_face_im - face_im_blur, 0, 255)

def yuzBirlestir(url, url2):
    try:
        print(url)
        print(url2)
        face_im = cv2.imread(url)
        body_im = cv2.imread(url2)
        
        face_points = get_landmarks(face_im)
        body_points = get_landmarks(body_im)

        M = transformation_from_points(face_points, body_points)
        warped_face_im = cv2.warpAffine(face_im, M, (body_im.shape[1],body_im.shape[0]))

        face_scale = numpy.std(body_points)
        corrected_face_im = correct_colours(warped_face_im, body_im, face_scale)

        mask_im = create_mask(body_points, body_im.shape, face_scale)
        combined_im = (corrected_face_im * mask_im + body_im * (1 - mask_im))

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        # Dosya adını oluşturun
        filename = f"combined_{timestamp}.jpg"
        # Dosya yolunu oluşturun
        filepath = os.path.join(settings.MEDIA_ROOT, 'oneriler', filename)
        cv2.imwrite(filepath, combined_im)
        data = f"media/oneriler/{filename}"
    except:
        data = "Hata"

    return data
