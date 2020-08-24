import keras 
from keras.models import load_model
import cv2
import os
import numpy as np
import sys
from keras.applications.resnet50 import ResNet50
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Flatten, Dense
from keras.models import Sequential, Model

from keras.preprocessing import image
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.applications.xception import preprocess_input
import time

#這裡的load會超久
model = load_model('./ResNet_1.h5')

from keras.preprocessing import image as image_utils
from shutil import copyfile
path = "./test" #欲辨識之資料夾名稱 
new_path = "./dest" #欲儲存之目的地資料夾名稱
def predict_yes(path):
    files= os.listdir(path)#得到資料夾下的所有檔名稱
    s = [] 
    for file in files:
        if file.endswith('.jpg'):
            print('catch file:',file)
            test_image = image_utils.load_img(path+'/'+file, target_size=(250, 250))
            test_image = image_utils.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0) 
            test_image /= 255
            result = model.predict(test_image)
            probability_to_label = np.argmax(result)
            if probability_to_label == 0:
                copyfile(path+'/'+file, new_path+'/'+file)
                print('successfully catch and copy the invoice numbers path to :',new_path+'/'+file)
    pass

tStart = time.time()
predict_yes('./test')
tEnd = time.time()
print("Total predict time: ", tEnd - tStart)