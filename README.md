Group project for CS4243 Computer Vision and Pattern Recognition
-----------
This project aims to encourage students to propose and implement their own algorithms to detect characters of interest
in images that you would find in the Where's Waldo books.

To facilitate the detection task, we release training and validation data for you to work with and test your algorithm's effectiveness. Our images are scanned from the "Where's Waldo" books and we don't have copyright so please do not distribute the images online.  

In 'datasets/', you will find 'JPEGImages/' and corresponding 'Annotations/'.  The image splits (our recommended training and validation split) are are given in 'ImageSet/'. You can visualize the annotations with the provided script 'vis_anno.py'

We provide baseline detection results in 'baseline/' for each of the three characters ('waldo', 'wenda', 'wizard')
on the validation set and an evaluation script to quantify the performance.  We follow the standard metric of mean
average precision (mAP) and use a threshold of 0.5 as the minimum overlap between the predicted bounding box and
the ground truth bounding box.  Please run the script 'evaluation.py' to see the baseline results. 

As mentioned in lecture, we give bonus marks for the fastest / most innovative / most accurate solutions.  For a fair comparison, we will test your algorithms on hold-out test set. Therefore, for this project, you need to submit your code, along with a readme file to help the TAs run your code.

For your inputs, any input format that follows the provided directory structure is OK. Please refer to the files under 'baseline/' to format your outputs. Specifically, please separately save the predictions for each of the three characters, and strictly name them as 'xx.txt', where xx is the character name (wizard, wenda, waldo). Each row in a file denotes a predicted bounding box, and is given by:

```
image_name score xmin ymin xmax ymax
```
where 'score' denotes the prediction confidence of the bounding box represented by [xmin, ymin, xmax, ymax], i,e,. the top left and bottom right of the rectangle.


# Where's Waldo?

The aim of this project is to  find three characters - Waldo, Wanda and Wizard in images from a popular book called Where's Waldo using any 'non-deep' computer vision algorithm. We explore the different techniques we tried and our final proposed solution and how we optimised it in the context of this project.


## Version

- Python Version 3.6
- Jupyter Notebook


# Scripts

- You can run `extract_templates.py` to generate the test data for positives of each character. For negative we used a random script.
- We have also updated our annotations.


# How to Run?

We have implemented two different methods in the source code

## SVM (slow to run)

The main files are: 

- utils.py
- classifier_utils.py

One can run the `character_classifier.ipynb` to see the results. It will save into the correct directory and that can be updated into the correct txt files and can be checked by running `evaluation.py`


## HAAR

The main file is `character_classifier.ipynb` and running that will save the txt file and can also be used to see visualization for certain images.


Follow below steps to see the the mAP results



# How to see evaluation result?

- Run the evaluation.py after running anyone of above method.
- Change `detpath = 'haar_test/{}.txt'` in evaluation.py to either of them
    - `svm_test` (svm)
    - `svm_haar_test` (haar and then svm)
    - `haar_test` (haar only) [best results]



## Folder directory

Source Codes/
    requirements.txt
    README.md
    cache_anno/
    cascade/ ## XML FILES OF TRAINING
        waldo/
        wenda/
        wizard/
    datasets/
    haar_test/ ## Stores txt result files
    svm_haar_test/ ## Stores txt result files
    svm_test/ ## Stores txt result files
    cascade_detector.ipynb ## HAAR
    character_classifier.ipynb ## SVM
    classifier_utils.py
    utils.py
    evaluation.py
    extract_templates.py ## Run this first














