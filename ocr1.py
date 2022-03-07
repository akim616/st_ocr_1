from socket import herror
from unittest import result
import streamlit as st
from io import BytesIO, StringIO
from PIL import Image
import numpy as np
import cv2
import pytesseract 

# For st
def img_upload():
    uploaded_file = st.file_uploader("Choose a image file", type=["jpg", 'png'])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        return cv2.imdecode(file_bytes, 1)

# Image Preprocess
def rescale_custom(img, n_w):
    h, w=img.shape[:2]
    new_width=n_w
    dim=(new_width, int(h*new_width/float(w)))
    new_img=cv2.resize(img,dim)
    ratio=img.shape[1]/float(new_img.shape[1]) 
    return new_img, ratio

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def hist_equalizer(img):
    clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)

def otsu_thresh(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def adaptive_thresh(image, n, c):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, n, c) 


header=st.container()
data=st.container()
result=st.container()


with header:
    st.title('Business Card OCR')
    st.text('Optical Character Recognition with OpenCV and Pytesseract.')


with data:
    st.header('Upload file')
    image=img_upload()
    if image is not None:
        st.image(image, channels="BGR")


with result:
    st.header('Results:')
    if image is not None:
        image_resized, ratio=rescale_custom(image, 600)
        image_gray=grayscale(image_resized)
        image_equalized=hist_equalizer(image_gray)
        image_denoised=noise_removal(image_equalized)
        image_otsu_th=otsu_thresh(image_denoised)
        myconfig=r"--psm 12 --oem 3"
        text=pytesseract.image_to_string(image_otsu_th, config=myconfig)
        new_text='\n'.join(text.split('\n\n'))
        st.write(new_text)

        



