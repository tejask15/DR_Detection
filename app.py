import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import cv2
from PIL import Image
import numpy as np
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
import math 
import os
import sys 
# Some utilites
import numpy as np
from util import *
from image_predict import *
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from fpdf import FPDF
from datetime import date

app = Flask(__name__)

picFolder = os.path.join('static', 'pics')
print(picFolder)
app.config['UPLOAD_FOLDER'] = picFolder
pic1 = os.path.join(app.config['UPLOAD_FOLDER'], 'saved.jpg')
print(pic1)
model_b3 = load_b3()
model_b5 = load_b5()
model_b3_proc = load_b3_proc()
model_b5_proc = load_b5_proc()

print('Model loaded.')
def generateReport(name,age,gender,aadhar,phone,eye,condition,diagnosis_type):
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=20)
    pdf.image('static/report/header.png', 0,0,215,50)

    today = date.today()

    pdf.set_font("Arial", size=14)
    pdf.text(10,80,'NAME')
    pdf.text(55,80,':')
    
    pdf.text(60,80,name)
    pdf.text(10,90,'AGE')
    pdf.text(55,90,':')
    pdf.text(60,90,age)
    pdf.text(10,100,'GENDER')
    pdf.text(55,100,':')
    pdf.text(60,100,gender)
    pdf.text(10,110,'AADHAR NUMBER')
    pdf.text(55,110,':')
    pdf.text(60,110,aadhar)
    pdf.text(10,120,'PHONE NUMBER' )
    pdf.text(55,120,':')
    pdf.text(60,120,phone)
    pdf.text(10,130,"EYE ")
    pdf.text(55,130,':')
    pdf.text(60,130,eye)
    pdf.text(10,140,'PATIENT HISTORY')
    pdf.text(55,140,':')
    pdf.text(60,140,condition)
    pdf.text(150,60,'DATE:')
    pdf.text(165,60,str(today))



    pdf.image('static/uploads/normal.jpg', 40, 150, 50,50)
    pdf.image('static/pics/saved.jpg', 110, 150, 50,50)

    pdf.text(55,210,'Fundus Image')
    pdf.text(115,210,'Processed Image')
    pdf.set_font("Arial", size=18)
    pdf.text(10,220,'DIAGNOSIS: ')
    pdf.text(48,220,diagnosis_type)
    pdf.dashed_line(5,225,205,225)
    pdf.set_font("Arial", size=14)
    pdf.text(10,230,'Class [0]: No Diabetic Retinopathy')
    pdf.text(10,235,'Class [1]: Mild nonproliferative diabetic retinopathy')
    pdf.text(10,240,'Class [2]: Moderate nonproliferative diabetic retinopathy ')
    pdf.text(10,245,'Class [3]: Severe nonproliferative diabetic retinopathy')
    pdf.text(10,250,'Class [4]: Proliferative diabetic retinopathy')
    pdf.dashed_line(5,255,205,255)
    pdf.text(10,260,"REMARKS : ")

    pdf.image('static/report/footer.png', 0,291,215,6)

    pdf.output("static/report/Report.pdf")
@app.route('/')
def home():
    return render_template('index.html')

#@app.route('/predict',methods=['POST'])
def prediction(filepath):
    '''
    For rendering results on HTML GUI
    '''
    Info = [(x) for x in request.form.values()]
    print(Info)
    name=Info[0]
    age=Info[1]
    gender=Info[2]
    aadhar=Info[3]
    phone=Info[4]
    eye=Info[5]
    condition=Info[6]

    #img = cv2.imread(filepath) 
        
    #img_arr = np.array(img)
    #img_arr = img_arr[:,:,:3]

    ans_b3 = ans_predict(filepath,model_b3,256)
    print(ans_b3,"ans_b3")
    ans_b5 = ans_predict(filepath,model_b5,256)
    print(ans_b5,"ans_b5")
    ans_b3_proc = ans_predict(filepath,model_b3_proc,256)
    print(ans_b3_proc,"ans_b3_proc")
    ans_b5_proc = ans_predict(filepath,model_b5_proc,256)
    print(ans_b5_proc,"ans_b5_proc")

    l = [ans_b3,ans_b5,ans_b5,ans_b3_proc,ans_b5_proc]
    
    ans_mode = mode_ans(l)

    diagnosis_type=str(ans_mode)
    generateReport(name,age,gender,aadhar,phone,eye,condition,diagnosis_type)
    return ans_mode,name,age,aadhar,eye

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print(file_path)
        # Make prediction
        preds,name,age,aadhar,eye = prediction(file_path)
        print(pic1)
        picn='static/uploads/normal.jpg'
        return render_template('index.html', prediction_text=preds,user_image=pic1,user_image_normal=picn,preds=preds,name=name,age=age,aadhar=aadhar,eye=eye)
    
if __name__ == "__main__":
    app.run(debug=True)

try:
    os.remove('static/pics/saved.jpg')
    os.remove('static/uploads/normal.jpg')
    print('Done')
except OSError:
    pass