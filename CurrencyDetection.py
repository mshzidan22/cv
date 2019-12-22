#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

def sift_detector(outsider_image_path, dataset_image_path):
    # Function that compares input image to template
    # It then returns the number of SIFT matches between them
    
    #image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    
    image1 = cv2.imread(outsider_image_path)
    image2 = cv2.imread(dataset_image_path)
    
    # Create SIFT detector object
    sift = cv2.xfeatures2d.SIFT_create()
    
    
    # Obtain the keypoints and descriptors using SIFT
    keypoints_1, descriptors_1 = sift.detectAndCompute(image1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image2, None)

    # Define parameters for our Flann Matcher
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 3)
    search_params = dict(checks = 100)

    # Create the Flann Matcher object
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Obtain matches using K-Nearest Neighbor Method
    # the result 'matchs' is the number of similar matches found in both images
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    # Store good matches using Lowe's ratio test
    good_matches = []
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m) 
    
    if len(good_matches) > 150:
        img3 = cv2.drawMatches(image1, keypoints_1, image2, keypoints_2, good_matches, image1, flags=2)
        plt.figure(figsize=(20,10))
        plt.imshow(img3),plt.show()
        print('Comparing between the Entered image and', dataset_image_path)
        print("Number of keypoints Detected in image 1: ", len(keypoints_1))
        print("Number of keypoints Detected in image 2: ", len(keypoints_2))
        print('Good matches = ', len(good_matches))
        print('[+] The currency is :', dataset_image_path)
    return len(good_matches)


def check_congruence(img_path):
    for image_path in os.listdir('Dataset/original'):
        image_orig = os.path.join('Dataset', 'original', image_path)
        matches = sift_detector(img_path, image_orig)

def main():
    check_congruence('my5.jpg')

if  __name__ == '__main__':
    main()





