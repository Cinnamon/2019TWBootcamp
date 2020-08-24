from flask import Flask, request, make_response
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from  ssd import SSD
from encoder import Dataencoder
import os
import requests
import base64 
from PIL import Image
import io
PyTorch_REST_API_URL = 'http://127.0.0.1:5000/predict'
image_path = './12.jpg'
#def predict_result(image_path):
image = open(image_path, 'rb').read()
#print(image)
payload = {'image':image}
# submit the request
r = requests.post(PyTorch_REST_API_URL, files=payload)
print(r)
r = r.json()

if r['success']:
    data = r['prediction']
    data = base64.b64decode(data)
    image = Image.open(io.BytesIO(data))
    image.show()
else:
    print('Request Failed!')




#if __name__ == '__main__':
	#load_model()
#	app.run(debug=True)






