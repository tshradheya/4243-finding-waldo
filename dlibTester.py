import os
import dlib

training_dir = 'templates/'
testing_dir = 'testing_imgs/'
file_name = 'training_waldo.xml'
detector_name = 'detector.svm'


def main():
    options = dlib.simple_object_detector_training_options()
    options.add_left_right_image_flips = True
    options.C = 4
    options.epsilon = 0.05
    options.num_threads = 8
    options.be_verbose = True

    training_xml_path = os.path.join(training_dir, file_name)
    # testing_xml_path = os.path.join(testing_dir, file_name)

    dlib.train_simple_object_detector(training_xml_path, detector_name, options)

    print("")
    print("================================")
    print("")
    # print(f'Training accuracy: {dlib.test_simple_object_detector(testing_xml_path, detector_name)}')


if __name__ == '__main__':
    main()

"""
dlib.train_simple_object_detector(training_xml_path, detector_name, options)
RuntimeError: An impossible set of object labels was detected. This is happening because none
of the object locations checked by the supplied image scanner is a close enough
match to one of the truth boxes in your training dataset. To resolve this you 
need to either lower the match_eps, adjust the settings of the image scanner so
that it is capable of hitting this truth box, or adjust the offending truth 
rectangle so it can be matched by the current image scanner. Also, if you are 
using the scan_fhog_pyramid object then you could try using a finer image 
pyramid. Additionally, the scan_fhog_pyramid scans a fixed aspect ratio box 
across the image when it searches for objects. So if you are getting this error
and you are using the scan_fhog_pyramid, it's very likely the problem is that 
your training dataset contains truth rectangles of widely varying aspect 
ratios. The solution is to make sure your training boxes all have about the 
same aspect ratio. 

image index              141
match_eps:               0.5
best possible match:     0.493585
truth rect:              [(-5932, 1114) (-5832, 1222)]
truth rect width/height: 0.926606
truth rect area:         11009
nearest detection template rect:              [(-5927, 1074) (-5827, 1273)]
nearest detection template rect width/height: 0.505
nearest detection template rect area:         20200

"""