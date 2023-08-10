import cv2
import json
import numpy
import json

 # UNDISTORT ARRAY FORMAT - camera matrix, distortion coefficients, optimal camera matrix
left_in = json.loads(open('left-undistort-params.json').read())
right_in = json.loads(open('left-undistort-params.json').read())
left_mtx = numpy.array(left_in[0])
right_mtx = numpy.array(right_in[0])
left_dist = numpy.array(left_in[1])
right_dist = numpy.array(right_in[1])
left_opt = numpy.array(left_in[2])
right_opt = numpy.array(right_in[2])

def undistort(left_img: numpy.ndarray, right_img: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
    ''' Returns undistorted image pairs using the pre-calculated distortion parameters. '''
    left_undistorted = cv2.undistort(left_img, left_mtx, left_dist, left_opt)
    right_undistorted = cv2.undistort(right_img, right_mtx, right_dist, right_opt)
    return left_undistorted, right_undistorted