import numpy as np
import cv2
import math

# Gaussian smoothing
kernel_size = 3

# Canny Edge Detector
low_threshold = 100
high_threshold = 150

# Helper functions
def grayscale(img):
  
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, kernel_size):

    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def  contrast_adjustments(img):
    image = img
    new_image = np.zeros(image.shape, image.dtype)
    alpha = 1.1 
    beta = 20    
    
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
    cv2.imwrite('contrast_adjustments1.jpg',new_image)
    cv2.imshow("contrast adjustments", new_image)
    return new_image

def get_threshold(img_grey):
    ret3,th3 = cv2.threshold(img_grey,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite('threshold.jpg',th3)
    cv2.imshow("threshold", th3)
    cv2.waitKey(0)
    return th3

def get_erosion(th3):
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(th3,kernel,iterations = 1)
 
    cv2.imwrite('erosion.jpg',erosion )
    cv2.imshow("erosion", erosion)
    cv2.waitKey(0)
    return erosion

def get_dilation(erosion):
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(erosion,kernel,iterations = 1)
    cv2.imwrite('dilation.jpg',dilation )
    cv2.imshow("dilation", dilation)
    cv2.waitKey(0)
    return dilation

def get_canny(image,dilate, low_threshold, high_threshold):
    edges = cv2.Canny(dilate, low_threshold, high_threshold)
    cv2.imwrite('edges.jpg',edges )
    cv2.imshow("canny edge", edges)
    return edges
