import os
import xml.etree.ElementTree as ET
import numpy as np
from matplotlib import pyplot as plt


def extract_templates(image_id):
    """
    Extract templates
    :param image_id: id of image to extract from
    """
    image_dir = 'datasets/JPEGImages'
    anno_dir = 'datasets/Annotations'
    image_file = os.path.join(image_dir, '{}.jpg'.format(image_id))
    anno_file = os.path.join(anno_dir, '{}.xml'.format(image_id))
    assert os.path.exists(image_file), '{} not found.'.format(image_file)
    assert os.path.exists(anno_file), '{} not found.'.format(anno_file)

    anno_tree = ET.parse(anno_file)
    objs = anno_tree.findall('object')
    occurrences = {'waldo': 0, 'wenda': 0, 'wizard': 0}
    image = np.asarray(plt.imread(image_file))
    for key in occurrences.keys():
        if not os.path.exists('templates/' + key + '/' + image_id):
            os.makedirs('templates/' + key + '/' + image_id)
    for idx, obj in enumerate(objs):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        x1 = int(bbox.find('xmin').text)
        y1 = int(bbox.find('ymin').text)
        x2 = int(bbox.find('xmax').text)
        y2 = int(bbox.find('ymax').text)
        plt.imsave('templates/'
                   + name
                   + '/'
                   + image_id
                   + '/' + str(occurrences[name])
                   + '.jpg',
                   image[y1:y2, x1:x2])
        occurrences[name] += 1
    for key in occurrences.keys():
        if len(os.listdir('templates/' + key + '/' + image_id)) == 0:
            os.rmdir('templates/' + key + '/' + image_id)


for name in os.listdir("datasets/JPEGImages"):
    # print(name[0:3])
    extract_templates(name[0:3])
