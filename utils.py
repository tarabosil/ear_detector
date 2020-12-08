import os
import cv2
import pandas as pd

import config
from bounding_box import BoundingBox


bounding_box = BoundingBox()


def ground_truth_bounding_boxes_to_csv():
    """ Save coordinates of bounding boxes. """
    images = os.listdir(config.path_to_ground_truth_bb_cnn)

    file = open('cnn/train.csv', 'w')
    file.write('img_name,xMin,yMin,xMax,yMax')

    for img_name in images:
        img = cv2.imread(f'{config.path_to_ground_truth_bb_cnn}/{img_name}')
        boxes = bounding_box.get_bounding_boxes(img)
        for box in boxes:
            file.write(f'\n{img_name},{box[0]},{box[1]},{box[2]},{box[3]}')

    file.close()


def csv_to_txt():
    """ Convert csv file to txt file """
    train = pd.read_csv('cnn/train.csv')

    data = pd.DataFrame()
    data['format'] = train['img_name']

    for i in range(data.shape[0]):
        data['format'][i] = 'cnn/train/' + data['format'][i]

    for i in range(data.shape[0]):
        data['format'][i] = data['format'][i] + ',' + str(train['xMin'][i]) + ',' + str(train['yMin'][i]) + ',' + str(
            train['xMax'][i]) + ',' + str(train['yMax'][i]) + ',Ear'

    data.to_csv('cnn/annotate.txt', header=None, index=None, sep=' ')


ground_truth_bounding_boxes_to_csv()
csv_to_txt()