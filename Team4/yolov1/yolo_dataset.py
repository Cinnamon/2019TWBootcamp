import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import os
import sys
import random
import glob
import matplotlib.pyplot as plt

class yolodataset(Dataset):
    def __init__(self, root, label, train, transform):
        print('init data')
        self.root = root
        self.train = train
        self.transform = transform
        self.fnames = []
        self.image_size = 224
        self.label = {}

        f = open(label, 'r')
        for line in f:
            t = line.replace('\n','').split(' ')
            self.fnames.append(os.path.join(root,t[0]))
            
            num_box = int((len(t)-3)/4)
            if num_box == 0:
                print('?')
                self.label[t[0]] = 0
            else:
                self.label[t[0]] = []
                for i in range(num_box):
                    self.label[t[0]].append([t[3+4*i],t[4+4*i],t[5+4*i],t[6+4*i]])
        print(self.fnames[0])
    
    def __getitem__(self, idx):
        img = cv2.imread(self.fnames[idx])
        name = os.path.basename(self.fnames[idx])
        
        boxes = []
        if self.label[name] == 0:
            boxes.append([0,0,0,0])
            boxes.append([0,0,0,0])
        else:
            for i in range(len(self.label[name])):
                    tmp = self.label[name][i]
                    xmin, ymin = float(tmp[0]), float(tmp[1])
                    xmax, ymax = float(tmp[2]), float(tmp[3])
                    boxes.append([xmin,ymin,xmax,ymax])
        boxes = torch.Tensor(boxes)

        #data augmentation
        if self.train:
            img, boxes = self.random_flip(img, boxes)
            img, boxes = self.randomScale(img, boxes)
            img,boxes,all_label = self.randomShift(img,boxes,0)
        
        img = self.BGR2RGB(img)

        target = self.encoder(img, boxes) #7*7*11
        img = cv2.resize(img,(self.image_size, self.image_size))
        img = self.transform(img)
        
        
        return img, target
    def __len__(self):
        return len(self.fnames)
    
    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    def subMean(self,bgr,mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr
    
    def encoder(self, img, label):
        #original image size = 512*512
        grid_num = 7
        target = torch.zeros((grid_num,grid_num,11))
        h,w,_ = img.shape
        x_cell_size = w/grid_num
        y_cell_size = h/grid_num
        
        if type(label[0]) == int:
            return target
        for i in range(len(label)):
            
            tmp = label[i]
            xmin, ymin = float(tmp[0]), float(tmp[1])
            xmax, ymax = float(tmp[2]), float(tmp[3])
            xc, yc = (xmax+xmin)/2, (ymax+ymin)/2
            
            center_x, center_y = int(xc//x_cell_size), int(yc//y_cell_size)
            #print(center_x, center_y)
            # box1
            target[center_x, center_y, 0] = (xc%x_cell_size)/x_cell_size
            target[center_x, center_y, 1] = (yc%y_cell_size)/y_cell_size
            target[center_x, center_y, 2] = (xmax-xmin)/w
            target[center_x, center_y, 3] = (ymax-ymin)/h
            target[center_x, center_y, 4] = 1
            # box2
            target[center_x, center_y, 5] = (xc%x_cell_size)/x_cell_size
            target[center_x, center_y, 6] = (yc%y_cell_size)/y_cell_size
            target[center_x, center_y, 7] = (xmax-xmin)/w
            target[center_x, center_y, 8] = (ymax-ymin)/h
            target[center_x, center_y, 9] = 1

            #class
            target[center_x, center_y, 10] = 1 
                
        return target

    def randomShift(self,bgr,boxes,labels):
        #平移变换
        center = (boxes[:,2:]+boxes[:,:2])/2
        if random.random() <0.5:
            height,width,c = bgr.shape
            after_shfit_image = np.zeros((height,width,c),dtype=bgr.dtype)
            after_shfit_image[:,:,:] = (104,117,123) #bgr
            shift_x = random.uniform(-width*0.1,width*0.1)
            shift_y = random.uniform(-height*0.1,height*0.1)
            #print(bgr.shape,shift_x,shift_y)
            #原图像的平移
            if shift_x>=0 and shift_y>=0:
                after_shfit_image[int(shift_y):,int(shift_x):,:] = bgr[:height-int(shift_y),:width-int(shift_x),:]
            elif shift_x>=0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),int(shift_x):,:] = bgr[-int(shift_y):,:width-int(shift_x),:]
            elif shift_x <0 and shift_y >=0:
                after_shfit_image[int(shift_y):,:width+int(shift_x),:] = bgr[:height-int(shift_y),-int(shift_x):,:]
            elif shift_x<0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),:width+int(shift_x),:] = bgr[-int(shift_y):,-int(shift_x):,:]

            shift_xy = torch.FloatTensor([[int(shift_x),int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:,0] >0) & (center[:,0] < width)
            mask2 = (center[:,1] >0) & (center[:,1] < height)
            mask = (mask1 & mask2).view(-1,1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if len(boxes_in) == 0:
                return bgr,boxes,labels
            box_shift = torch.FloatTensor([[int(shift_x),int(shift_y),int(shift_x),int(shift_y)]]).expand_as(boxes_in)
            boxes_in = boxes_in+box_shift
            #print(mask.view(-1))
            for i in range(boxes_in.size()[0]):
                if boxes_in[i][2] > width or boxes_in[i][3] > height:
                    return bgr, boxes, labels
            #labels_in = labels[mask.view(-1)]
            return after_shfit_image,boxes_in,labels
        return bgr,boxes,labels

    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h,w,_ = im.shape
            if boxes[0][0] != 0:
                xmin = w - boxes[:,2]
                xmax = w - boxes[:,0]
                boxes[:,0] = xmin
                boxes[:,2] = xmax
            return im_lr, boxes
        return im, boxes

    def randomScale(self,bgr,boxes):

        if random.random() < 0.5:
            scale = random.uniform(0.9,1.1)
            height,width,c = bgr.shape
            bgr = cv2.resize(bgr,(int(width*scale),int(height*scale)))
            scale_tensor = torch.FloatTensor([[scale,scale,scale,scale]]).expand_as(boxes)
            boxes_scale = boxes * scale_tensor
            for i in range(boxes.size()[0]):
                if boxes_scale[i][2] > width or boxes_scale[i][3] > height:
                    return bgr, boxes
            return bgr,boxes_scale
        return bgr,boxes

class testdataset(Dataset):
    def __init__(self, root, transform, label):
        print('init data')
        self.root = root
        self.transform = transform
        self.fnames = []
        f = open(label, 'r')
        for line in f:
            t = line.replace('\n','').split(' ')
            self.fnames.append(os.path.join(root,t[0]))

        #self.fnames = sorted(glob.glob(root+'*.jpg'))
        self.image_size = 224
        
        self.mean = (123,117,104) #RGB

        #print(self.len)
    def __getitem__(self, idx): 
        img = cv2.imread(self.fnames[idx])
        name = os.path.basename(self.fnames[idx]).replace('jpg','txt')
        h,w,_ = img.shape
        img = self.BGR2RGB(img)
        img = cv2.resize(img,(self.image_size, self.image_size))
        img = self.transform(img)

        return img, name, h, w

    def __len__(self):
        return len(self.fnames)
        
    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
def main():
    dataset = yolodataset(root = 'WeaponS/WeaponS/', train = True, transform = transforms.ToTensor(), label='yolo_train.txt')
    #dataset = testdataset(root = 'WeaponS/WeaponS/', transform = transforms.ToTensor(), label='yolo_test.txt')
    #test_root = 'test/'
    #test_dataset = testdataset(root=test_root, transform = transforms.ToTensor())
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for idc, (image, target) in enumerate(test_loader):
        
        if idc==0:
            img = image.squeeze().numpy().transpose(1,2,0)
            
            plt.imshow(img)
            plt.show()
            break
        
          
    print('finish') 
    
if __name__ == '__main__':
    main()
        
