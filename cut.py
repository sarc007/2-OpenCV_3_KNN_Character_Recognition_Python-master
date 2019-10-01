# Capture the mouse click events in Python and OpenCV
'''
-> draw shape on any image
-> reset shape on selection
-> crop the selection
run the code : python capture_events.py --image image_example.jpg
'''


# import the necessary packages
import argparse
import cv2
import numpy as np

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
ref_point = []
cropping = False

def shape_selection(event, x, y, flags, param):
  # grab references to the global variables
  global ref_point, cropping

  # if the left mouse button was clicked, record the starting
  # (x, y) coordinates and indicate that cropping is being
  # performed
  if event == cv2.EVENT_LBUTTONDOWN:
    ref_point = [(x, y)]
    cropping = True

  # check to see if the left mouse button was released
  elif event == cv2.EVENT_LBUTTONUP:
    # record the ending (x, y) coordinates and indicate that
    # the cropping operation is finished
    ref_point.append((x, y))
    cropping = False

    # draw a rectangle around the region of interest
    cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
    cv2.imshow("image", image)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", shape_selection)

# keep looping until the 'q' key is pressed
while True:
  # display the image and wait for a keypress
  cv2.imshow("image", image)
  key = cv2.waitKey(1) & 0xFF

  # if the 'r' key is pressed, reset the cropping region
  if key == ord("r"):
    image = clone.copy()

  # if the 'c' key is pressed, break from the loop
  elif key == ord("c"):
    break

# if there are two reference points, then crop the region of interest
# from teh image and display it
if len(ref_point) == 2:
  crop_img = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
  kernel = np.array([[-1, -1, -1],
                     [-1, 9, -1],
                     [-1, -1, -1]])
  crop_img = cv2.filter2D(crop_img, -1, kernel)  # applying the sharpening kernel to the input image & displaying it.
  cv2.imshow('Image Sharpening', crop_img)
  cv2.imshow("crop_img", crop_img)
  cv2.imwrite('C:\\Users\\ahs\\Desktop\\dubai lp\\crop.png',crop_img)
  cv2.waitKey(0)

  width = 350
  height = 100
  dim = (width, height)



  # resize image
  resized = cv2.resize(crop_img, dim, interpolation=cv2.INTER_AREA)

  print('Resized Dimensions : ', resized.shape)

  cv2.imshow("Resized image", resized)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

# close all open windows
cv2.destroyAllWindows()
