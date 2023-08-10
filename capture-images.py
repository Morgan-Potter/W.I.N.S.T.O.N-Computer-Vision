import cv2
vid = cv2.VideoCapture(1)
vid2 = cv2.VideoCapture(2)
imgn=0
while(True):
    ret, img = vid.read()
    ret2, img2 = vid2.read()
    if not ret:
        raise Exception("Camera one was not found")
    if not ret2:
        raise Exception("Camera two was not found")
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img1',img) #display the captured image
    cv2.imshow('img2', img2)
    cv2.imshow('grey', grey)
    cv2.imshow('grey2', grey2)
    response = cv2.waitKey(0)
    if response == int(ord('y')): #save on pressing 'y' 
        cv2.imwrite('calibration-images/left/' + str(imgn) + '.png',img)
        cv2.imwrite('calibration-images/right/' + str(imgn) + '.png',img2)
        cv2.destroyAllWindows()
        imgn += 1
    elif response == int(ord('n')):
        break
