import cv2
# import glob

# images = glob.glob('*.jpg')
#
# for i in images:
#     img = cv2.imread(i,0)
#
#     re = cv2.resize(img,(150,33))
#
#     cv2.imwrite("resized_"+ i,re)

img = cv2.imread('4 7.jpg',0)
re = cv2.resize(img,(150,33))

cv2.imwrite('resize4_7.jpg',re)