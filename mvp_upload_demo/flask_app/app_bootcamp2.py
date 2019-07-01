from flask import Flask, render_template, request
import os
import re
import json
import requests
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

""" initialize our flask app """
app = Flask(__name__)

""" load index_to_char dictionary """
with open('./flask_app/data.json', 'r') as f:
    index_to_char = json.load(f)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    """ Get the file from post request """
    f = request.files['file']

    """ Save the file to ./uploads """
    base_path = './flask_app/uploads'
    file_path = os.path.join(base_path, secure_filename(f.filename))
    f.save(file_path)

    """ load image """
    img = image.load_img(file_path, target_size=(28, 28))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    print(img.shape)
    payload = json.dumps({'instances': img.tolist()})
    r = requests.post(
        'http://localhost:8501/v1/models/bootcamp_demo2:predict',
        data=payload)

    pred = json.loads(r.text)
    pred_index = np.argmax(pred['predictions'], axis=1)
    index = str(pred_index[0] + 1)
    char = index_to_char[index]

    return char


if __name__ == '__main__':
    # decide what port to run the app in
    port = int(os.environ.get('PORT', 5000))
    # run the app locally on the givn port
    app.run(host='0.0.0.0', port=port)
