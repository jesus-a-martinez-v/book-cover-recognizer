import cv2
import numpy as np


class CoverDescriptor(object):
    def __init__(self, use_sift=False):
        self.use_sift = use_sift

    def describe(self, image):
        descriptor = cv2.BRISK_create()

        if self.use_sift:
            descriptor = cv2.xfeatures2d.SIFT_create()

        (keypoints, descriptors) = descriptor.detectAndCompute(image, None)
        keypoints = np.float32([keypoint.pt for keypoint in keypoints])

        return (keypoints, descriptors)
