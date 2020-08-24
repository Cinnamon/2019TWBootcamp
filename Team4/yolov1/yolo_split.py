import csv
import numpy as np
import os
#split train and valid data for yolo model

'''
file format : image_name width height xmin ymin xmax ymax (first box) xmin ymin xmax ymax(second box).....
normally object detection have many object class, but our task only have one class, gun
'''
def yolo_format():
    with open('labels.csv','r') as f:
        file = open('a.txt','w')
        csv_reader = csv.reader(f, delimiter=',')
        line_count = 0
        tmp = ''
        for row in csv_reader:
            if line_count > 0:
                if row[0] == tmp:
                    tmp = row[0]
                    if tmp == '2815.jpg':
                        file.write('\n{} {} {} {} {} {} {}'.format(row[0], row[1], row[2], row[4], row[5], row[6], row[7]))
                    else:
                        file.write(' {} {} {} {}'.format(row[4], row[5], row[6], row[7]))
                else:
                    tmp = row[0]
                    if tmp == '2815.jpg':
                        continue
                    file.write('\n{} {} {} {} {} {} {}'.format(row[0], row[1], row[2], row[4], row[5], row[6], row[7]))
            else:
                line_count+=1

def split():
    f = open('a.txt', 'r')
    lines = f.readlines()
    print(len(lines))

    t = open('yolo_train.txt','w')
    v = open('yolo_test.txt','w')
    for i in range(1, len(lines)-1):
        if i%5==0:
            v.write(lines[i])
        else:
            t.write(lines[i])

if __name__ =='__main__':
    yolo_format()
    split()
    os.remove('a.txt')

