import cv2
import numpy as np

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
gridx = 7
gridy = 8
img_max = 26
points = np.zeros((8*7,3), np.float32)
points[:,:2] = np.mgrid[0:7,0:8].T.reshape(-1,2)
for side in ['left', 'right']:
    objpoints = []
    imgpoints = []
    for imgn in range(img_max):
        img = cv2.imread('calibration-images/' + side + '/' + str(imgn) + '.png')
        greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(greyscale, (gridx,gridy), None)
        if ret == True:
            objpoints.append(points)
            corners2 = cv2.cornerSubPix(greyscale,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            print('yes')
        # cv2.drawChessboardCorners(img, (x,y), corners, ret)
        # cv2.imshow('img', img)
        # cv2.waitKey(500)
    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, greyscale.shape[::-1], None, None)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (640,480), 1, (640,480))
    undistortParams = [mtx.tolist(), dist.tolist(), newcameramtx.tolist()]
    open(side + '-undistort-params.json', 'w').write(str(undistortParams))