import os
from typing import Tuple
import cv2
import xml.etree.ElementTree as ET


def get_data(file):  # gets the data of the squares from the file
    root = ET.parse(file).getroot()  # gets the xml root
    xmin, ymin, xmax, ymax = [], [], [], []  # initializes empty lists for coordinates

    for child in root.iter('bndbox'):  # collects the data from the objects bndbox table
        xmin.append(int(child.find("xmin").text))
        ymin.append(int(child.find("ymin").text))
        xmax.append(int(child.find("xmax").text))
        ymax.append(int(child.find("ymax").text))

    return ymin, ymax, xmin, xmax  # returns the lists with coordinates for each square


def get_blurs(path, threshold=200) -> Tuple[dict, list]:
    os.chdir(path)
    files = [_ for _ in os.listdir(path) if _.endswith('xml')]
    objects = []
    blur_dict = {}
    for file in files:
        img = cv2.imread(file.replace('xml', 'jpg'))
        cnt_blur, counts = 0, 0

        for ymin, ymax, xmin, xmax in zip(*get_data(file)):
            img_crop = img[ymin:ymax, xmin:xmax]
            gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
            fm = cv2.Laplacian(gray, cv2.CV_64F).var()

            if fm < threshold:  # if the focus measure is less than the supplied threshold,
                cnt_blur += 1  # then the image should be considered "blurry"

            counts += 1
            objects.append(fm)

        blur_dict.update({file: (cnt_blur / counts) * 100})

    blur_dict.update({'total_percent': sum(blur_dict.values()) / len(blur_dict)})

    return blur_dict, objects
