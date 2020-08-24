import sys
import os

import cv2
import numpy as np

# usage : python3 visualize.py <image.jpg> <label.txt>
CLASSES = ('gun')

Color = [[0, 0, 0]]

def parse_det(detfile):
    result = []
    with open(detfile, 'r') as f:
        for line in f:
            token = line.strip().split()
            if len(token) != 6:
                continue
            x1 = int(float(token[0]))
            y1 = int(float(token[1]))
            x2 = int(float(token[2]))
            y2 = int(float(token[3]))
            cls = token[4]
            prob = float(token[5])
            result.append([(x1,y1),(x2,y2),cls,prob])
    return result 

if __name__ == '__main__':
    imgfile = sys.argv[1]
    detfile = sys.argv[2]

    image = cv2.imread(imgfile)
    result = parse_det(detfile)
    for left_up,right_bottom,class_name,prob in result:
        
        color = Color[CLASSES.index(class_name)]
        cv2.rectangle(image,left_up,right_bottom,color,2)
        label = class_name+str(round(prob,2))
        
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        p1 = (left_up[0], left_up[1]- text_size[1])
        cv2.rectangle(image, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
        cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)

    cv2.imwrite('result.jpg',image)
