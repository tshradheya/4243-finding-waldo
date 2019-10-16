import xml.etree.ElementTree as ET
import os


def createNewTrainingXml(character):
    """

    <images>
      <image file='frame102.jpg'>
        <box top='252' left='165' width='912' height='933'/>
      </image>
      <image file='frame105.jpg'>
        <box top='216' left='174' width='903' height='963'/>
      </image>
    </images>

    <data>
        <items>
            <item name="item1">item1abc</item>
            <item name="item2">item2abc</item>
        </items>
    </data>
    """

    # image_dir = 'datasets/JPEGImages'
    anno_dir = 'datasets/Annotations'
    # image_file = os.path.join(image_dir, '{}.jpg'.format(image_id))

    images = ET.Element("images")

    for fileName in os.listdir("datasets/JPEGImages"):
        fileNameOnly = fileName[0:3]

        anno_file = os.path.join(anno_dir, '{}.xml'.format(fileNameOnly))
        # assert os.path.exists(image_file), '{} not found.'.format(image_file)
        assert os.path.exists(anno_file), '{} not found.'.format(anno_file)

        anno_tree = ET.parse(anno_file)
        objs = anno_tree.findall('object')
        i = 0
        for idx, obj in enumerate(objs):
            name = obj.find('name').text
            if name == character:
                bbox = obj.find('bndbox')
                x1 = int(bbox.find('xmin').text)
                y1 = int(bbox.find('ymin').text)
                x2 = int(bbox.find('xmax').text)
                y2 = int(bbox.find('ymax').text)

                top = y1
                left = x1
                width = x2 - x1
                height = y2 - y1

                imageElement = ET.SubElement(images, 'image')
                fileLoc = character + '/' + fileNameOnly + "/" + str(i) + ".jpg"
                imageElement.set('file', fileLoc)

                boxElement = ET.SubElement(imageElement, 'box')
                boxElement.set("top", str(top))
                boxElement.set("left", str(left))
                boxElement.set("width", str(width))
                boxElement.set("height", str(height))
                i += 1

    mydata = ET.tostring(images, encoding="unicode")
    myfile = open("training_waldo.xml", "w")
    myfile.write(mydata)


createNewTrainingXml("waldo")
