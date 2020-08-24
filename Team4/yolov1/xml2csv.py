import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def xml_to_csv(path):
    xml_list = []
    i = 1
    for xml_file in sorted(glob.glob(path + '/*.xml')):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        name = ('{:04}'.format(i)+'.jpg')
        
        for member in root.findall('object'):
            value = (name,
                     #root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
        i+=1
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


xml_df = xml_to_csv('WeaponS_bbox/WeaponS_bbox')   #where you place armas (1).xml armas (2).xml.........  
xml_df.to_csv('labels.csv', index=None)
print('Successfully converted xml to csv.')