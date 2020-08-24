import os
import sys
import os.path

import random
import numpy as np

import torch
import torch.utils.data as data
import cv2
import torchvision.transforms as transforms

from encoder import Dataencoder 
class ListDataset(data.Dataset):
    img_size = 300
    def __init__(self, root, list_file, train, transform):
        self.root = root
        self.list_file = list_file
        self.train = train
        self.transform = transform
        self.data_encoder = Dataencoder()
        self.fnames = []
        self.boxes = []
        # box in boxes would be [x_min,y_min, x_max, y_max] between 0,1
        self.labels = [] 
        self.size = 300
        # input one line with [filename, w, h, box1_xmin, box1_y_min, box1_x_max, box1_y_max, .......]
        with open(list_file) as f:
            input_lines = f.readlines()
            self.num_samples = len(input_lines)
        for line in input_lines:
            items = line.replace('\n','').split(' ')
            num_box = (len(items)-3)//4
            self.fnames.append(items[0])
          
            box = []
            label = []
            for i in range(num_box):
                x_m, y_m, x_M, y_M = float(items[i*4+3]), float(items[i*4+4]), float(items[i*4+5]), float(items[i*4+6])
                box.append([x_m, y_m, x_M, y_M])
                label.append(1)
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
             
                
    def __getitem__(self, idx):
        img = cv2.imread(self.root + self.fnames[idx])
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx]
            
        if self.train:
            img, boxes = self.random_flip(img, boxes)
            img, boxes, labels = self.random_crop(img, boxes, labels)  
        w, h = img.shape[1], img.shape[0]
        boxes = torch.cat([(boxes[:,:2] + boxes[:,2:])/2,(boxes[:,2:]-boxes[:,:2])], 1)
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)
        #print('boxes',boxes) 
        img =  self.BGR2RGB(img)
        img =  cv2.resize(img, (self.size, self.size))
        img = self.transform(img)
        loc_target, con_target = self.data_encoder.encode(boxes , labels)
        
        return img, loc_target, con_target
        
        
        
        
        
            
    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    def random_flip(self, img, boxes):
        '''Randomly flip the image and adjust the bbox locations.
        For bbox (xmin, ymin, xmax, ymax), the flipped bbox is:
        (w-xmax, ymin, w-xmin, ymax).
        Args:
          img: (ndarray.Image) image. f
          boxes: (tensor) bbox locations, sized [#obj, 4].
        Returns:
          img: (ndarray.Image) randomly flipped image.
          boxes: (tensor) randomly flipped bbox locations, sized [#obj, 4].
        '''
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            w = img.shape[1]
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:,0] = xmin
            boxes[:,2] = xmax
        return img, boxes

    def random_crop(self, img, boxes, labels):
        '''Randomly crop the image and adjust the bbox locations.
        For more details, see 'Chapter2.2: Data augmentation' of the paper.
        Args:
          img: (ndarray.Image) image.
          boxes: (tensor) bbox locations, sized [#obj, 4].
          labels: (tensor) bbox labels, sized [#obj,].
        Returns:
          img: (ndarray.Image) cropped image.
          selected_boxes: (tensor) selected bbox locations.
          labels: (tensor) selected bbox labels.
        '''
        imw, imh = img.shape[1], img.shape[0]
        while True:
            min_iou = random.choice([None, 0.1, 0.3, 0.5, 0.7, 0.9])# random choice the one 
            if min_iou is None:
                return img, boxes,labels

            for _ in range(100):
                w = random.randrange(int(0.1*imw), imw)
                h = random.randrange(int(0.1*imh), imh)

                if h > 2*w or w > 2*h or h < 1 or w < 1:
                    continue

                x = random.randrange(imw - w)
                y = random.randrange(imh - h)
                roi = torch.Tensor([[x, y, x+w, y+h]])
                
                center = (boxes[:,:2] + boxes[:,2:]) / 2  # [N,2]
                roi2 = roi.expand(len(center), 4)  # [N,4]
    
                mask = (center > roi2[:,:2]) & (center < roi2[:,2:])  # [N,2]
                mask = mask[:,0] & mask[:,1]  #[N,]

                if not mask.any():
                    continue
              
                selected_boxes = boxes.index_select(0, mask.nonzero().squeeze(1))
                
                iou = self.data_encoder.iou2(selected_boxes, roi)
                if iou.min() < min_iou:
                    continue
                img = img[y:y+h, x:x+w, :]
                
                selected_boxes[:,0].add_(-x).clamp_(min=0, max=w)
                selected_boxes[:,1].add_(-y).clamp_(min=0, max=h)
                selected_boxes[:,2].add_(-x).clamp_(min=0, max=w)
                selected_boxes[:,3].add_(-y).clamp_(min=0, max=h)

                return img, selected_boxes, labels[mask]

    def __len__(self):
        return self.num_samples




