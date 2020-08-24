# -*- coding: utf-8 -*-

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import tensorflow as tf

from retina.utils import visualize_boxes
from filter_new import *

class infer_app():
    '''
    def load_inference_model():
        model_path=os.path.join('snapshots', 'resnet.h5')
        model = models.load_model(model_path, backbone_name='resnet50')
        model = models.convert_model(model)
        model.summary()
        return model
    '''
    '''
    def post_process(self,boxes, original_img, preprocessed_img):
        # post-processing
        h, w, _ = self.preprocessed_img.shape
        h2, w2, _ = self.original_img.shape
        boxes[:, :, 0] = self.boxes[:, :, 0] / w * w2
        boxes[:, :, 2] = self.boxes[:, :, 2] / w * w2
        boxes[:, :, 1] = self.boxes[:, :, 1] / h * h2
        boxes[:, :, 3] = self.boxes[:, :, 3] / h * h2
        return self.boxes
    '''
    @staticmethod
    def pred_string(frame):
        global model
        image = frame
        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        
        # preprocess image for network
        image = preprocess_image(image)
        image, _ = resize_image(image, 416, 448)
        
        # process image
        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        print("processing time: ", time.time() - start)

        #######################################
        h, w, _ = image.shape
        h2, w2, _ = draw.shape
        boxes[:, :, 0] = boxes[:, :, 0] / w * w2
        boxes[:, :, 2] = boxes[:, :, 2] / w * w2
        boxes[:, :, 1] = boxes[:, :, 1] / h * h2
        boxes[:, :, 3] = boxes[:, :, 3] / h * h2        
        #######################################
        
        labels = labels[0]
        scores = scores[0]
        boxes = boxes[0]
        out_image, pred_str, bb_cord = visualize_boxes(draw, boxes, labels, scores, class_labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        pred_str_np = []
        #print("pred_str:",pred_str)
        #print("bb_cord:",bb_cord)

        for i,val in enumerate(pred_str):
            #print (val[0][0])
            pred_str_np.append(int(val[0][0]))
        pred_str_np = np.array(pred_str_np)[:,np.newaxis]
        #print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        if (pred_str_np != []):
            print("pred_str_np: ", pred_str_np)
        cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR))
        labels = [int(i[0][0]) for i in pred_str]
        if len(bb_cord) > 0:
            return FindNumber(bb_cord, labels)
        else:
            return "No Number Found QAQ"
        '''
        if (len(pred_str_np) != 8) :
            pred_str_n_bb = str(1111111)            
            return pred_str_n_bb
        else:

            pred_str_n_bb = np.concatenate((np.array(bb_cord),pred_str_np),axis=1)
            pred_str_n_bb_sort = pred_str_n_bb[pred_str_n_bb[:,1].argsort()]
            #pred_str_n_bb_sort = np.array2string(pred_str_n_bb_sort)
            print("pred_str_n_bb_sort[:,2]: ",pred_str_n_bb_sort[:,2])
            print("len pred_str_n_bb_sort[:,2]: ",len(pred_str_n_bb_sort[:,2]))
            #print("Type print(pred_str_n_bb_sort[:,2]): ",type(pred_str_n_bb_sort[:,2]))
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR))
            cc = pred_str_n_bb_sort[:,2]
            bb = np.array2string(cc, formatter={'float_kind':lambda x: "%.d" % x}).replace(" ","")
            bb = bb.replace("[","")
            st = bb.replace("]","")
            return st
        '''

    def __init__(self):    
        global model, graph
        MODEL_PATH = 'snapshots/resnet50_pascal_01.h5'
        #IMAGE_PATH = 'samples/JPEGImages/4.jpg'
        #model_path=os.path.join('snapshots', 'resnet.h5')
        model = models.load_model(MODEL_PATH, backbone_name='resnet50')
        model._make_predict_function()
        self.graph = tf.get_default_graph()
        model = models.convert_model(model)
        #model.summary()

        # load image
        #image = read_image_bgr(IMAGE_PATH)
                
    # 5. plot
    #plt.imshow(out_image)
    #plt.show()