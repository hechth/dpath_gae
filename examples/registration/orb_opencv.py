from __future__ import print_function
import cv2
import numpy as np
import sys, argparse, os, math
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import matplotlib.pyplot as plt

 
MAX_FEATURES = 1000
GOOD_MATCH_PERCENT = 0.01
 
 
def alignImages(im1, im2):
 
  # Convert images to grayscale
  #im1Gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
  #im2Gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
   
  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2, None)
   
  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)
   
  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)
 
  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]
 
  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
   
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)
 
  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
   
  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
  h2 = cv2.getAffineTransform(points1[0:3], points2[0:3])
  # Use homography
  height, width, channels = im2.shape
  #im1Reg = cv2.warpPerspective(im1, h, (width, height))
  im1Reg = cv2.warpAffine(im1, h2, (width, height))
   
  return im1Reg, h, imMatches
 
def main(argv):     
    # Read reference image
    refFilename = os.path.join(git_root,'data','images','HE_level_1_cropped_512x512.png')

    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
    b,g,r = cv2.split(imReference)       # get b,g,r
    imReference = cv2.merge([r,g,b])     # switch it to rgb

    # Read image to be aligned
    imFilename = os.path.join(git_root,'data','images','CD3_level_1_cropped_512x512.png')
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
    b,g,r = cv2.split(im)       # get b,g,r
    im = cv2.merge([r,g,b])     # switch it to rgb   
    # Registered image will be resotred in imReg. 
    # The estimated homography will be stored in h. 
    imReg, h, matches = alignImages(im, imReference)

    fig, ax =plt.subplots(2,2)

    ax[0,0].imshow(imReference)
    ax[0,1].imshow(im)

    ax[1,0].imshow(imReg)
    ax[1,1].imshow(matches)

    plt.show()
 
if __name__ == '__main__':
    main(sys.argv[1:])