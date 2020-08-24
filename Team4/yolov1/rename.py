

import os
import glob
#rename xml file 
path = 'WeaponS_bbox/WeaponS_bbox/'   #where you place armas (1).xml armas (2).xml.........  
for xml_file in sorted(glob.glob(os.path.join(path, '*.xml'))):
    check = os.path.basename(xml_file)
    
    new = (check[check.find('(')+1 : check.find(')')])
    
    os.rename(xml_file, os.path.join(path, '{:04}'.format(int(new))) +'.xml')


#rename image file 
path = 'WeaponS/WeaponS/'   #where you place armas (1).jpg armas (2).jpg......... 
for name in sorted(glob.glob(os.path.join(path, '*.jpg'))):
    check = os.path.basename(name)
    
    new = check[check.find('(')+1 : check.find(')')]

    os.rename(name, os.path.join(path, '{:04}'.format(int(new)))+'.jpg')
