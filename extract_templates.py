import os
import xml.etree.ElementTree as ET
import numpy as np
from matplotlib import pyplot as plt


def extract_templates(image_id, type):
    """
    Extract templates
    :param image_id: id of image to extract from
    :param type: to decide where to store the image
    """
    global waldo
    global wenda
    global wizard

    storage_folder = "emp"
    if type == "train":
        storage_folder = "updated"
    if type == "test":
        storage_folder = "updated_test"
    image_dir = 'datasets/JPEGImages'
    anno_dir = 'datasets/Annotations'
    image_file = os.path.join(image_dir, '{}.jpg'.format(image_id))
    print(image_file)
    anno_file = os.path.join(anno_dir, '{}.xml'.format(image_id))
    assert os.path.exists(image_file), '{} not found.'.format(image_file)
    assert os.path.exists(anno_file), '{} not found.'.format(anno_file)

    anno_tree = ET.parse(anno_file)
    objs = anno_tree.findall('object')
    occurrences = {'waldo': 0, 'wenda': 0, 'wizard': 0}

    image = np.asarray(plt.imread(image_file))
    for key in occurrences.keys():
        if not os.path.exists(storage_folder + '/' + key):
            os.makedirs(storage_folder + '/' + key)
    for idx, obj in enumerate(objs):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        x1 = int(bbox.find('xmin').text)
        y1 = int(bbox.find('ymin').text)
        x2 = int(bbox.find('xmax').text)
        y2 = int(bbox.find('ymax').text)
        if name == "waldo":
            plt.imsave(storage_folder + '/' + name + '/' + str(waldo) + '.jpg',
                       image[y1:y2, x1:x2])
            waldo = waldo + 1
        if name == "wenda":
            plt.imsave(storage_folder + '/' + name + '/' + str(wenda) + '.jpg',
                       image[y1:y2, x1:x2])
            wenda = wenda + 1
        if name == "wizard":
            plt.imsave(storage_folder + '/' + name + '/' + str(wizard) + '.jpg',
                       image[y1:y2, x1:x2])
            wizard = wizard + 1


filename = 'datasets/ImageSets/val.txt'
with open(filename) as f:
    waldo = 0
    wenda = 0
    wizard = 0
    for img_id in f.readlines():
        extract_templates(img_id.rstrip(), "test")

filename = 'datasets/ImageSets/train.txt'
with open(filename) as f:
    waldo = 0
    wenda = 0
    wizard = 0
    for img_id in f.readlines():
        extract_templates(img_id.rstrip(), "train")
