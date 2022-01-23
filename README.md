# Image-Feature-Detector

# Synopsis

The goal of feature detection and matching is to identify a pairing between a point in one image and a corresponding point in another image. 
These correspondences can then be used to stitch multiple images together into a panorama.

The project has three parts: feature detection, feature description, and feature matching.

## 1. Feature Detection

In this step, points of interest of the image are identified using Harris corner detection. 
A Harris matrix is computed, and corner strength is traced at every pixel. 

## 2. Feature Description

In this step, two descriptors were implemented, simple and MOPS. 
They are used to create points of interest to compare.  

## 3. Feature Matching

Last is the matching code. Two methods were tried here

1) Sum of squared distances (SSD): The sequared Euclidian distance between two vectors

2) Ratio test: The ratio between closest and second closest features by SSD yielding the closest distances are matched
