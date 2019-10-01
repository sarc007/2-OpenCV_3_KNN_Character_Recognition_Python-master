# import the necessary packages
import imutils
import cv2

def click(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,',',y)
        font = cv2.FONT_HERSHEY_COMPLEX
        strXY = str(x)+','+str(y)
        cv2.putText(img,strXY,(x,y),font,.5,(255,255,0),2)
        cv2.imshow('image',img)


img = cv2.imread("6.jpg")
cv2.imshow('image',img)
cv2.setMouseCallback('image',click)
cv2.waitKey(0)