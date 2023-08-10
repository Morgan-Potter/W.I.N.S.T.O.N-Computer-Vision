import harris_corner
import undistort_image
import cv2
import time

Lstream = cv2.VideoCapture(1) # NUMBER IS BASED OFF USB ENUMERATION // LEFT & RIGHT CAMERA MAY DIFFER. 
Rstream = cv2.VideoCapture(0)

time.sleep(1) # delay to wait for auto white balance, focus and exposure
while True:
    ret, left_img = Lstream.read()
    ret2, right_img = Rstream.read()
    print('Image taken.')
    und_left_img, und_right_img = undistort_image.undistort(cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY))
    print('Undistorted.')
    sob_left_img, sob_right_img = harris_corner.sobel(und_left_img, und_right_img)
    print('Done.')
    cv2.imshow('left_gray_undistorted', und_left_img)
    cv2.imshow('right_gray_undistorted', und_right_img)
    cv2.imshow('left_sobel', sob_left_img)
    cv2.imshow('right_sobel', sob_right_img)
    if cv2.waitKey(0) == int(ord('n')):
        break
    cv2.destroyAllWindows()
cv2.destroyAllWindows()