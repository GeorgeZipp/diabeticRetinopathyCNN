import os
import cv2
from xml.etree import ElementTree




for i in range (1,90):
    filename = "coordinate" + str(i) +".txt"
    f = open(filename,"w+")
    hard_exudates = []
    soft_exudates = []
    othercoords = []
    for x in range (1,4):
        if i<=9:
            file_name = 'diaretdb1_image00'+str(i)+'_0'+str(x)+'.xml'
        else:
            file_name = 'diaretdb1_image0'+str(i)+'_0'+str(x)+'.xml'


        full_file = os.path.abspath(os.path.join('data', file_name))

        dom = ElementTree.parse(full_file)



        for m in dom.findall('markinglist/marking'):
            type = m.find('markingtype').text
            coords = m.find('representativepoint/coords2d').text
            if (type == 'Hard_exudates'):
                hard_exudates.append(coords)
            elif (type == 'Soft_exudates'):
                soft_exudates.append(coords)
            else:
                othercoords.append(coords)

    f.write('hard_exudates:\n')
    for hard in hard_exudates:
        f.write(hard + '\n')
    f.write('soft_exudates:' + '\n')
    for soft in soft_exudates:
        f.write(soft + '\n')

    f.write('all_coordinates:\n')
    for other in othercoords:
        f.write(other + '\n')