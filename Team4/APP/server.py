from flask import Flask, request, make_response, render_template , flash, redirect, url_for
import torch
from werkzeug.utils import secure_filename
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from  ssd import SSD
from encoder import Dataencoder
import os
import json
import flask
import io
from io import BytesIO
from PIL import Image, ImageDraw
import base64
from time import sleep
from torch import nn
import torchvision.models as models
import signal


UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

class_model = None
object_model = None
use_gpu = False

# Preprocess the input picture
transform = transforms.Compose([
            transforms.Resize((224,224)),
            #transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


class feature_extractor(nn.Module):

    def __init__(self, num_classes = 3):

        super(feature_extractor, self).__init__()
        self.num_classes = num_classes
        
        #https://pytorch.org/docs/stable/torchvision/models.html
        resnet50 = models.resnet50(pretrained=True)
        modules = list(resnet50.children())[:-1]
        self.net = nn.Sequential(*modules)
        
        #for resnet50, it will be 2048, but may different in other models, you need to check
        self.ft_size = 2048
        
        self.classifier = nn.Linear(self.ft_size, num_classes)
        
    def forward(self, x):
        features = self.net(x)
        features = features.view(-1, self.ft_size)
        x = self.classifier(features)
        return x


def handler(sig, frame):
    print('delete file on uploads')
    global file_num
    for i in range(file_num):
        filename = '{:04}'.format(i) + '.jpg'
        os.remove('static/uploads/'+ filename) 
    exit(0)
signal.signal(signal.SIGINT, handler)


# Check the format of uploaded file
def allowed_file(filename):
    return '.' in filename and \
           filename.split('.', 1)[1] in ALLOWED_EXTENSIONS

def load_model():
    global file_num
    file_num = 0
    global class_model
    class_model = feature_extractor()
    model_path = './class_model.pki'
    tmp = torch.load(model_path, map_location={'cuda:0':'cpu'})
    class_model.load_state_dict(tmp)
    class_model.eval()
    del tmp


    global object_model
    object_model = SSD(depth=50, width=1)
    model_path = './ssd_patch.pki'
    tmp = torch.load(model_path, map_location={'cuda:0': 'cpu'})
    object_model.load_state_dict(tmp)
    object_model.eval()
    

def prepare_image(img, target_size):
    # Preprocess the image, return the image with target size.
    #transform = transforms.ToTensor()        # value in image will be 0~1
    img = img.resize((target_size, target_size))

    # add batch axis 
    #img = img[None]

    #return torch.autograd.Variable(img, volatile=True)
    return img


def mosaic(frame, x, y, w, h, neighbor=9):
    #fh, fw = frame.shape[0], frame.shape[1]
    draw = ImageDraw.Draw(frame)
    frame = np.array(frame)
    fh, fw = frame.shape[0], frame.shape[1]
    if (y + h > fh) or (x + w > fw):
        return
      
    for i in range(0, h-neighbor, neighbor):
        for j in range(0, w-neighbor, neighbor):
            rect = [j+x, i+y, neighbor, neighbor]
            color = frame[i+y][j+x].tolist()
            color = tuple(color)
            left_up = (rect[0], rect[1])
            right_down = (rect[0]+neighbor-1, rect[1]+neighbor-1)
            #cv2.rectangle(frame, left_up, right_down, color, -1)
            draw.rectangle([left_up, right_down], fill=color)

def object_detection(image, filename):
    global file_num
    final_path = None
    #final_path = './static/uploads/a.jpg'
    w, h = image.size
    img = prepare_image(image, target_size=300)
    transform = transforms.ToTensor()        # value in image will be 0~1
    x = transform(img)

    with torch.no_grad():
        loc_pred, conf_pred = object_model(x.unsqueeze(0))    # img [1, 3, 300, 300]

    data_encoder = Dataencoder()
    conf_pred_softmax = F.softmax(conf_pred.squeeze(0), dim=1) # [8732, 2]
    draw = ImageDraw.Draw(image)
    boxes, labels, scores = data_encoder.decode(loc_pred.data.squeeze(0), conf_pred_softmax.data)
    for box, label, score in zip(boxes, labels, scores):
        box[:,0] *= w
        box[:,1] *= h
        box[:,2] *= w
        box[:,3] *= h
        for b, s in zip(box, score):
            if s > 0.6:
                b = list(b)     # [x_min, y_min, x_max, y_max]
                for i in range(4):
                    b[i] = int(b[i].item())
                #mosaic(image, int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1]))
                draw.rectangle([(b[0], b[1]),(b[2], b[3])], fill =(88, 106, 127))
    place = filename.find('.')
    file_type = filename[place:] 
    #image.save(filename, file_type)
    print(file_num)
    filename = '{:04}'.format(file_num) + '.jpg'
    file_num += 1
    final_path = './static/uploads/'
    image.save(final_path+filename)
    image.close()
    return final_path + filename


@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-store'
    return response


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    file_path = None
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            #print(filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            #print(file_path)
            #print(os.path.join('/static/uploads/',filename))
            #print(saved_file_path)
            #return file.filename
    return render_template('index.html', upload_image=file_path)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # initialize the dictionary that will be returned from the view
    #data = {'success': False}
    response = {'success':False}
    final_path = None

    if request.method == 'POST':
        filename = request.values['img_to_predict']
        print(filename)
        image = Image.open(filename)

        if image != None:
            response['success'] = True
            t_img = transform(image)
            out = class_model(t_img.view(1,-1,224,224))
            predict = torch.max(out, 1)[1]
            if predict != 2:
                final_path = object_detection(image, filename)
            else:
                final_path = filename
            print('Success predict image!')
            return render_template('index.html', upload_image=final_path)
        else:
            return '<pre>' + 'No such file' + '</pre>'
  


load_model()
app.run(host='127.0.0.1', port=5000)

