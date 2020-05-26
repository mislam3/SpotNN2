# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:31:32 2020

@author: siamm
"""

# Parking spot occupancy/vacancy recognition

# libraries
from PIL import image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
import cv2
from keras.models import load_model
import numpy as np



from keras.preprocessing import image
model = load_model('facefeatures_new_model_final.h5')

# Loading the cascades; gets distinct features from the data provided by intel via link: use proper one for vehicle features
vehicle_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def vehicle_extractor(img):
    # function detects vehicles and returns the cropped vehicle
    # If no vehicle detected, it returns the input imagee
    
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    vehicles = vehicle_cascade.detectMultiScale(img, 1.3, 5)
    
    if vehicles is ():
        return None
    
    
    # crop all faces found
    for (x,y,w,h) in vehicles:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        cropped_vehicle = img[y:y+h, x:x+w]
        
    return cropped_vehicle

# Performing some face recognition using webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    # canvas = detect(gray, frame)
    # image, vehicle =vehicle_detector(frame)
    
    vehicle=vehicle_extractor(frame)
    if type(vehicle) is np.ndarray:
        vehicle = cv2.resize(vehicle, (224, 224))
        im = image.fromarray(vehicle, 'RGB')
        # im = i\Image.fromarray(vehicle, 'RGB')
        #Resizing into 128*128 because we trained the model with this image size
        img_array = np.array(im)
        # our keras model used a 4d tensor, (images * height * width * chann..)
        # so changing dimensions 1*128*128*3
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        print(pred)
        
        name="None Matching"
        
        if(pred[0][3]>0.5):
            name='Vehicle'
        cv2.putText(frame,name, (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        
    else:
        cv2.putText(frame,"No Vehicle Found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()

    