import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import math


app = Flask(__name__)


model =load_model('BrainTumor10Epochs.h5')
print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo,A,s):
	if classNo==0:
		return "No Brain Tumor", ""
	else:
	    return "Yes, Brain Tumor"+'\n' + "Area covered :" + str(A)+'\n'+ "Intensity of Brain Tumor :" + str(s) + '\n' ,  "It has sharp edge" if s<=43.5 else " It has cloudy edge"
                 

def getResult(img):
    image=cv2.imread(img)
    imag1=cv2.imread(img)
    image = Image.fromarray(image)
    image = image.resize((64, 64))
    image=np.array(image)
    print(image)
    s = np.std(imag1) * 0.5
    imag1 = np.array(imag1)  
    l = np.count_nonzero(imag1 == 1)
    A = (0.264)* math.sqrt(l)
    input_img = np.expand_dims(image, axis=0)
    
    result=model.predict(input_img)
    result_final=np.argmax(result,axis=1)
    #result_final=np.where(result == np.amax(result))[1][0]
    #result_final = result[0][result_final]
    print(result_final)
    print(l)
    return result_final,A,s 


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value,ar,s=getResult(file_path)
        result,result1=get_className(value,ar,s) 
    
        return result+result1
    return None


if __name__ == '__main__':
    app.run(debug=True)