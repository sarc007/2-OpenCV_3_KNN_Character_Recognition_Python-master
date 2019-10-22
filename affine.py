import scipy.ndimage as ndi
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage



img = cv2.imread('car.jpg')
rows,cols,ch = img.shape

pts1 = np.float32([[486 , 271],[736 , 270],[485 , 332],[738 , 332]])
pts2 = np.float32([[0,0],[1200,0],[0,675],[1200,675]])

M = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(img,M,(1500,800)) #perpesctive transform

cv2.imshow('output',dst)

cv2.waitKey(0)
# plt.axis('off')
# plt.savefig('saved')

# plt.subplot(121),plt.imshow(img),plt.title('Input')
# plt.subplot(122),plt.imshow(dst),plt.title('Output')
# plt.savefig(dst)
# plt.show()
