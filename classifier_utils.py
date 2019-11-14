import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import cyvlfeat as vlfeat
from sklearn.svm import LinearSVC, SVC
import os.path as osp
from skimage import filters
from skimage.feature import corner_peaks
from skimage.io import imread
import pickle
from random import shuffle
from scipy.spatial.distance import cdist
from skimage.feature import hog

from operator import add
from functools import reduce
from utils import *


def bags_of_sifts_spm(image_paths, vocab_filename, depth=3):
    """
    Bags of sifts with spatial pyramid matching.

    :param image_paths: paths to N images
    :param vocab_filename: Path to the precomputed vocabulary.
          This function assumes that vocab_filename exists and contains an
          vocab_size x 128 ndarray 'vocab' where each row is a kmeans centroid
          or visual word. This ndarray is saved to disk rather than passed in
          as a parameter to avoid recomputing the vocabulary every run.
    :param depth: Depth L of spatial pyramid. Divide images and compute (sum)
          bags-of-sifts for all image partitions for all pyramid levels.
          Refer to the explanation in the notebook, tutorial slide and the 
          original paper (Lazebnik et al. 2006.) for more details.

    :return image_feats: N x d matrix, where d is the dimensionality of the
          feature representation. In this case, d will equal the number of
          clusters (vocab_size) times the number of regions in all pyramid levels,
          which is 21 (1+4+16) in this specific case.
    """
    with open(vocab_filename, 'rb') as f:
        vocab = pickle.load(f)
    
    vocab_size = vocab.shape[0]
    feats = []
    weights = [0.25, 0.25, 0.5]


    for path in image_paths:
        img = load_image_gray(path)
        
        H = img.shape[0]
        W = img.shape[1]
        histogram = []
        
        _, descriptors = vlfeat.sift.dsift(img, step=5, fast=True)
        dist = cdist(vocab, descriptors, 'euclidean')
        
        
        # Level Zero
        min_dist_idx = np.argmin(dist, axis = 0)
        hist = np.histogram(min_dist_idx, np.arange(201))[0] * weights[0]

        if np.linalg.norm(histogram) != 0:
            hist = hist / np.linalg.norm(hist)
        
        histogram.extend(hist)
        
        
        # Level 1
        blocks = split(img)
        
        for block in blocks:
            _, descriptors = vlfeat.sift.dsift(block, step=5, fast=True)
            dist = cdist(vocab, descriptors, 'euclidean')
            min_dist_idx = np.argmin(dist, axis = 0)
            hist, _ = np.histogram(min_dist_idx, np.arange(201))
            hist = hist * weights[1]

            if np.linalg.norm(histogram) != 0:
                hist = hist / np.linalg.norm(hist)

            histogram.extend(hist)
            
        
        # Level 2
        
        for block in blocks:
            sub_blocks = split(block)
        
            for sub_block in sub_blocks:
                _, descriptors = vlfeat.sift.dsift(sub_block, step=5, fast=True)
                dist = cdist(vocab, descriptors, 'euclidean')
                min_dist_idx = np.argmin(dist, axis = 0)
                hist, _ = np.histogram(min_dist_idx, np.arange(201))
                hist = hist * weights[2]

                if np.linalg.norm(histogram) != 0:
                    hist = hist / np.linalg.norm(hist)
                
                histogram.extend(hist)

        
        feats.append(histogram)

    return np.array(feats)


def split(arr):
    """Split a matrix into sub-matrices."""

    half_split = np.array_split(arr, 2)

    result = map(lambda x: np.array_split(x, 2, axis=1), half_split)
    result = reduce(add, result)

                
    return result


def bags_of_sifts(image_paths, vocab_filename):

    # load vocabulary
    with open(vocab_filename, 'rb') as f:
        vocab = pickle.load(f)

    # dummy features variable
    feats = np.zeros((len(image_paths),len(vocab)))


    for i, path in enumerate(image_paths):
        
        image = load_image_gray(path)
        frames, descriptors = vlfeat.sift.dsift(image, step=5, fast=True)
        
        
        dist = cdist(vocab, descriptors, 'euclidean')
        min_dist_idx = np.argmin(dist, axis = 0)
        histogram, _ = np.histogram(min_dist_idx, range(len(vocab)+1))
        
        if np.linalg.norm(histogram) == 0:
            feats[i, :] = histogram
        else:
            feats[i, :] = histogram / np.linalg.norm(histogram)

    return feats


def bags_of_sifts_image(image, vocab_filename):

    # load vocabulary
    with open(vocab_filename, 'rb') as f:
        vocab = pickle.load(f)

    # dummy features variable
    feats = np.zeros((1,len(vocab)))


        
    frames, descriptors = vlfeat.sift.dsift(image, step=5, fast=True)


    dist = cdist(vocab, descriptors, 'euclidean')
    min_dist_idx = np.argmin(dist, axis = 0)
    histogram, _ = np.histogram(min_dist_idx, range(len(vocab)+1))

    if np.linalg.norm(histogram) == 0:
        feats[0, :] = histogram
    else:
        feats[0, :] = histogram / np.linalg.norm(histogram)

    return feats

def build_vocabulary(image_paths, vocab_size):

    dim = 128      # length of the SIFT descriptors that you are going to compute.
    vocab = np.zeros((vocab_size,dim))
    total_SIFT_features = np.zeros((20*len(image_paths), dim))
    
    step_size = 5;
    features = []
    
    for path in image_paths:
        img = load_image_gray(path)
        _, descriptors = vlfeat.sift.dsift(img, step=step_size, fast=True)
        
        descriptors = descriptors[np.random.choice(descriptors.shape[0], 20)]
        features.append(descriptors)
    
    features = np.concatenate(features, axis=0).astype('float64')

    vocab = vlfeat.kmeans.kmeans(features, vocab_size) 
    
        
    return vocab


def svm_classify(train_image_feats, train_labels, test_image_feats):    
    clf = LinearSVC(C=2)
    clf.fit(train_image_feats, train_labels)
    test_labels = clf.predict(test_image_feats)

    return test_labels


def test_accuracy(test_labels, predicted_labels):
    num_correct = 0
    for i,label in enumerate(test_labels):
        if (predicted_labels[i] == label):
            num_correct += 1
    return num_correct/len(test_labels)

def svm_probability(train_image_feats, train_labels, test_image_feats):
    
    svc = SVC(C=2, gamma='scale',probability=True)
    svc.fit(train_image_feats, train_labels)
    test_probabilities = svc.predict_proba(test_image_feats)

#     clf = LinearSVC(C=2, probability=True)
#     clf.fit(train_image_feats, train_labels)
#     test_labels = clf.predict(test_image_feats)

    return test_probabilities

def find_characters(vocab_filename, training_feats, train_labels, test_feats):
    
    window = 64
    f = open('datasets/ImageSets/val.txt')
    wa = open('baseline_test/waldo.txt', 'w+')
    we = open('baseline_test/wenda.txt', 'w+')
    wi = open('baseline_test/wizard.txt', 'w+')
    
    image_id = f.readline().rstrip()
    while image_id:
        print(image_id)
        print("processing")
#         image_id = "003"
        image = np.asarray(plt.imread('datasets/JPEGImages/' + image_id + '.jpg'))
        H, W, chan = image.shape
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        test_feats = []

        orb = cv2.ORB_create()
#         orb = cv2.ORB_create(nfeatures=1000, scoreType=cv2.ORB_FAST_SCORE)
        kp, des = orb.detectAndCompute(img_gray, None)

# #         minHessian = 400
# #         detector = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
# #         kp = detector.detect(img_gray)

#             fast = cv2.FastFeatureDetector_create()
#         # find and draw the keypoints
#         kp = fast.detect(img_gray,None)
        img_kp = cv2.drawKeypoints(img_gray, kp, None, color=(0,0,255), flags=cv2.DrawMatchesFlags_DEFAULT)
        
#         plt.figure()
#         plt.imshow(img_kp)
#         plt.show()
        
        for idx in range(len(kp)):
            j,i = kp[idx].pt

            i = int(np.round(i))
            j = int(np.round(j))
            i_end = i+window
            j_end = j+window
            
            i_end = min(i_end, H-1)
            j_end = min(j_end, W-1)

            img = img_gray[i:i_end,j:j_end]
            feats = bags_of_sifts_image(img_gray, vocab_filename)
            test_feats.extend(feats)

            
        numOfMax = 5
        probability = svm_probability(training_feats, train_labels, test_feats)

        locations = np.argpartition(-probability, numOfMax, axis =0)[:numOfMax]

        
        for k in range(len(locations[0])):
            for l in range(numOfMax):

                y, x  = kp[locations[l][k]].pt

                x = int(np.round(x))
                y = int(np.round(y))
                y_end = y+window
                x_end = x+window

                x_end = min(x_end, H-1)
                y_end = min(y_end, W-1)

                patch = img_gray[x:x_end, y:y_end]
#                 plt.imshow(patch)
#                 plt.show()

                if (probability[locations[l][k]][k] > 0.4):
                    if k == 0:
                        res = image_id + ' ' + str(probability[locations[l][k]][k]) + ' ' + str(x) + ' ' + str(y) + ' ' + str(x_end) + ' ' + str(y_end) + '\n'
                        wa.write(res)
                    if k == 1:
                        res = image_id + ' ' + str(np.max(probability[locations[l][k]][k])) + ' ' + str(x) + ' ' + str(y) + ' ' + str(x_end) + ' ' + str(y_end) + '\n'
                        we.write(res)
                    if k == 2:
                        res = image_id + ' ' + str(np.max(probability[locations[l][k]][k])) + ' ' + str(x) + ' ' + str(y) + ' ' + str(x_end) + ' ' + str(y_end) + '\n'
                        wi.write(res)
        image_id = f.readline().rstrip()


        
        
        

def find_characters_second_check(vocab_filename, training_feats, train_labels, test_feats):
    
    waldo_cascade = cv2.CascadeClassifier('cascade.xml')

    window = 64
    f = open('datasets/ImageSets/val.txt')
    wa = open('baseline_test/waldo.txt', 'w+')
    we = open('baseline_test/wenda.txt', 'w+')
    wi = open('baseline_test/wizard.txt', 'w+')
    
    image_id = f.readline().rstrip()
    while image_id:
        print(image_id)
        print("processing")
#         image_id = "003"
        image = np.asarray(plt.imread('datasets/JPEGImages/' + image_id + '.jpg'))
        H, W, chan = image.shape
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


        
        waldo_candidates, _, weights = waldo_cascade.detectMultiScale3(img_gray, 1.03, 20, outputRejectLevels=True)
        for idx, (x,y,w,h) in enumerate(waldo_candidates):
            test_feats = []
            i = y
            i_end = y+h
            j = x
            j_end = x+w
            

            img = img_gray[i:i_end,j:j_end]
            feats = bags_of_sifts_image(img_gray, vocab_filename)
            test_feats.extend(feats)

            
            numOfMax = 5
            probability = svm_probability(training_feats, train_labels, test_feats)
            if probability[0][0] > 0.5:
                res = image_id + ' ' + str(probability[0][0]) + ' ' + str(j) + ' ' + str(i) + ' ' + str(j_end) + ' ' + str(i_end) + '\n'
                wa.write(res)
                

#             locations = np.argpartition(-probability, numOfMax, axis =0)[:numOfMax]

        
#         for k in range(len(locations[0])):
#             for l in range(numOfMax):

#                 y, x  = kp[locations[l][k]].pt

#                 x = int(np.round(x))
#                 y = int(np.round(y))
#                 y_end = y+window
#                 x_end = x+window

#                 x_end = min(x_end, H-1)
#                 y_end = min(y_end, W-1)

#                 patch = img_gray[x:x_end, y:y_end]
# #                 plt.imshow(patch)
# #                 plt.show()

#                 if (probability[locations[l][k]][k] > 0.4):
#                     if k == 0:
#                         res = image_id + ' ' + str(probability[locations[l][k]][k]) + ' ' + str(x) + ' ' + str(y) + ' ' + str(x_end) + ' ' + str(y_end) + '\n'
#                         wa.write(res)
#                     if k == 1:
#                         res = image_id + ' ' + str(np.max(probability[locations[l][k]][k])) + ' ' + str(x) + ' ' + str(y) + ' ' + str(x_end) + ' ' + str(y_end) + '\n'
#                         we.write(res)
#                     if k == 2:
#                         res = image_id + ' ' + str(np.max(probability[locations[l][k]][k])) + ' ' + str(x) + ' ' + str(y) + ' ' + str(x_end) + ' ' + str(y_end) + '\n'
#                         wi.write(res)
        image_id = f.readline().rstrip()


