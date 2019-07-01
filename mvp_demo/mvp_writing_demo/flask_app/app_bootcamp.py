from flask import Flask, render_template, request
import chardet
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import keras.models
import re
import base64
import sys
import os
import tensorflow as tf
import json
import requests
import numpy as np
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image


""" initalize our flask app """
app = Flask(__name__)

""" load index_to_char dictionary """
with open('./flask_app/data.json', 'r') as f:
    index_to_char = json.load(f)


def save_as_image(data, img_name):
    """ decoding an image from base64 into raw representation """
    data = data.decode('ascii')
    img_str = re.search(r'base64,(.*)', data).group(1)
    with open(img_name, 'wb') as output:
        output.write(base64.b64decode(img_str))


def load_and_preprocess(img_name, target_size):
    img = image.load_img(img_name, target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    return img


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    """ whenever the predict method is called, we're going
          to input the user drawn character as an image into the model
          perform inference, and return the classification
          get the raw data format of the image
    """
    data = request.get_data()

    img_name = 'output.png'
    save_as_image(data, img_name)

    img = image.load_img(img_name, target_size=(28, 28))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    payload = json.dumps({'instances': img.tolist()})
    r = requests.post(
        'http://localhost:8501/v1/models/bootcamp_demo:predict',
        data=payload)

    pred = json.loads(r.text)
    pred_index = np.argmax(pred['predictions'], axis=1)
    index = str(pred_index[0] + 1)
    char = index_to_char[index]

    return char


if __name__ == "__main__":
    # decide what port to run the app in
    port = int(os.environ.get('PORT', 5000))
    # run the app locally on the givn port
    app.run(host='0.0.0.0', port=port)
