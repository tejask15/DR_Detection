# image height and image width ----> GLOBAL

import cv2
from PIL import Image
import numpy as np
import os
import math
import sys
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,LearningRateScheduler
from tqdm import tqdm_notebook as tqdm
from collections import Counter
import random
# Some utilites
import numpy as np
from util import base64_to_pil


img_ht = 380
img_wd = 380

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
       
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img
   
   
def circle_crop(img, sigmaX):  
    """
    Create circular crop around image centre    
    """    
    img = crop_image_from_gray(img)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   
    height, width, depth = img.shape    
   
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
   
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img=cv2.addWeighted(img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)
    return img

def preprocess_image_gb(img_path):
    img = cv2.imread(img_path)
    cv2.imwrite('static/uploads/normal.jpg',img)
    img_t = circle_crop(img,sigmaX = 30)
    img_blue_channel=cv2.resize(cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB),(256,256))
    #f, axarr = plt.subplots(1,2,figsize = (11,11))
    #axarr[0].imshow(img_t)
    #axarr[1].imshow(img_r)
    #plt.title('After applying Circular Crop and Gaussian Blur')
    #plt.show()
    cv2.imwrite('static/pics/saved.jpg',img_blue_channel)
    #return img_blue_channel
    return True

def ans_predict_gb(img,model,desired_size):
    img = preprocess_image_gb(img)
    img = np.expand_dims(img,axis = 0)
    ans = model.predict(img) > 0.5
    ans = (ans.astype(int).sum(axis=1) - 1)[0]
    return ans

def preprocess_image(path, desired_size=256):
    im = Image.open(path)
    ixp = preprocess_image_gb(path)
    img = im.resize((desired_size,desired_size), resample=Image.LANCZOS)
    return img

def build_model(cnn_net):
    model = Sequential()
    model.add(cnn_net)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.00005),
        metrics=['accuracy']
    )
    
    return model

import efficientnet.tfkeras as efn



def load_b3():
    cnn_net = efn.EfficientNetB3(weights='imagenet',include_top=False,input_shape=(256, 256, 3))
    model = build_model(cnn_net)
    model.load_weights('models/B3_weights.h5')
    return model

def load_b3_proc():
    cnn_net = efn.EfficientNetB3(weights='imagenet',include_top=False,input_shape=(256, 256, 3))
    model = build_model(cnn_net)
    model.load_weights('models/B3_img_proc_weights.h5')
    return model

def load_b5():
    cnn_net = efn.EfficientNetB5(weights='imagenet',include_top=False,input_shape=(256, 256, 3))
    model = build_model(cnn_net)
    #model.load_weights('models/effb5_old_new.h5')
    model.load_weights('models/B5_weights.h5')
    return model

def load_b5_proc():
    cnn_net = efn.EfficientNetB5(weights='imagenet',include_top=False,input_shape=(256, 256, 3))
    model = build_model(cnn_net)
    model.load_weights('models/B5_img_proc_weights.h5')
    return model



def ans_predict(img,model,desired_size):
    img = preprocess_image(img,desired_size)
    img = np.expand_dims(img,axis = 0)
    ans = model.predict(img) > 0.5
    ans = (ans.astype(int).sum(axis=1) - 1)[0]
    return ans

def ans_predict_prc(img,model,desired_size):
    img = new_preprocess_image(img,desired_size)
    img = np.expand_dims(img,axis = 0)
    ans = model.predict(img) > 0.5
    ans = (ans.astype(int).sum(axis=1) - 1)[0]
    return ans

def mode_ans(lst):
    l_ans = []
    d_mem_count = Counter(lst)
    cnt_max = max(d_mem_count.values())
    for k in d_mem_count.keys():
        if(d_mem_count[k] == cnt_max):
            l_ans.append(k)
    if(len(l_ans) == 1):
        return "Diabetic Retinopathy class "+str(l_ans[0])
    else:
        l_ans.sort()
        return "Diabetic Retinopathy class is between class : "+str(l_ans[0])+" to class : "+str(l_ans[-1])
