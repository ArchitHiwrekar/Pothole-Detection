import cv2
import numpy as np
import helpers as hp
import pygame
import time

kernel_size = 3

low_threshold = 50
high_threshold = 200
camera = cv2.VideoCapture(0)
cv2.destroyAllWindows()
if (camera.isOpened() == False):
    print("Error opening camera")

while(camera.isOpened()):
    ret, image= camera.read()

    image3 = hp.gaussian_blur(image, kernel_size)
    cv2.imwrite('gaussian_blur.jpg',image3)
    cv2.waitKey(0)
    image4 = hp.contrast_adjustments(image3)
    cv2.imwrite('contrast_adjustments.jpg',image4)
    cv2.waitKey(0)


    img_grey = cv2.cvtColor(image4, cv2.COLOR_BGR2GRAY)
    # Threshold

    th3 = hp.get_threshold(img_grey)

# Erosion
    erosion = hp.get_erosion(th3)

# Dilation 1

    dilate =  hp.get_dilation(erosion)

# Dilation 1
    dilate = hp.get_dilation(dilate)

# canny
    canny = hp.get_canny(image, dilate, low_threshold, high_threshold)

#plt.show()
    cv2.waitKey(0)

# find contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 600 and area < 50000:
             ellipse = cv2.fitEllipse(contour)
             cv2.ellipse(image, ellipse, (0, 255, 0), 2)
             #cv2.drawContours(image, contour, -1, (0, 255, 0), 3)
             cv2.imshow("output", image)
             cv2.imwrite('output.jpg',image)
             pygame.init()
             pygame.mixer.music.load("pothole.mp3")
             pygame.mixer.music.play()
             time.sleep(5)
        else:
            break
camera.release()
cv2.waitKey(0)
