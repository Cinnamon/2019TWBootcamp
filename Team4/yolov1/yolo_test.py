import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable
from yolo_models import Yolov1_vgg16bn, vgg19_bn
from yolo_loss import yoloLoss
from yolo_dataset import yolodataset, testdataset
from yolo_predict import decoder
import numpy as np


def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)

def save_checkpoint(checkpoint_path, model):
    state = {'state_dict': model.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)


def main():
    use_gpu = torch.cuda.is_available()

    test_root = 'WeaponS/WeaponS/'   #where you place images



    batch_size = 16

    model = vgg19_bn()
    print('load pre-trained model')

    load_checkpoint('models/model_yolo_20.pth', model)    #change to your own model path


    save_root = 'test_label'
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    if use_gpu:
        model.cuda()

    model.eval()

    valid_dataset = testdataset(root=test_root, transform = transforms.ToTensor(), label='yolo_test.txt')
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2)
    print('start test')
    

    with torch.no_grad():
        for idx, (images,name,h,w) in enumerate(valid_loader):
            images = Variable(images)
            if use_gpu:
                images = images.cuda()
            
            pred = model(images)

            keep = decoder(pred.cpu())
            
            pred = pred.squeeze().cpu()
            
            f = open(os.path.join(save_root, str(name[0])), 'w')
            w = int(w)
            h = int(h)
            for i in range(len(keep)):
                num_cell = keep[i][0]
                xmin, xmax = str(keep[i][1][0]*w/224), str(keep[i][1][2]*w/224)
                ymin, ymax = str(keep[i][1][1]*h/224), str(keep[i][1][3]*h/224)
                cofid = keep[i][1][4]
                cell_i = num_cell//7
                cell_j = num_cell%7
                value, index = torch.max(pred[cell_i,cell_j,10:],0)

                #shrink box ex:size*0.9
                '''
                weight = xmax - xmin
                height = ymax - ymin
                xmin += 0.05*weight
                ymin += 0.05*height
                xmax -= 0.05*weight
                ymax -= 0.05*height
                '''
                f.write('{} {} {} {} {} {}\n'.format(xmin,ymin,xmax,ymax,'gun',(value.numpy()*cofid)))
                
        f.close()     
        print('finish')

if __name__=='__main__':
    main()