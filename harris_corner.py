
import numpy
import math

# Load the image in grayscale

def sobel(left_img, right_img):

    # Compute the gradient in x and y direction using Sobel filters
    soimages = []
    for img in [left_img, right_img]:
        out = []
        for x in range(1, len(img)-1): # array of pixels at x
            row = []
            for y in range(1, len(img[0])-1): # array at index 0 = y
                # Compute the gradient in x and y direction using Sobel filters & sharpen
                gx = abs(img[x-1][y+1] + 2*img[x][y+1] + img[x+1][y+1] - img[x-1][y-1] - 2*img[x][y-1] - img[x+1][y-1]) # top row, - bottom row [[-1,1], [0,1], [1,1], [-1,-1], [-1,0], [-1,1]]
                gy = abs(img[x+1][y-1] + 2*img[x+1][y] + img[x+1][y+1] - img[x-1][y-1] - 2*img[x-1][y] - img[x-1][y+1])
                gnorm = math.sqrt(gx*gx + gy*gy) # Consider the partial derivative in x and y directions to be part of a vector and find the magnitude (normalize)
                row.append(gnorm)
            out.append(row)
        soimages.append(numpy.array(out, numpy.uint8))
    return soimages

# Compute the elements of the structure tensor

# You are creating the structure tensor for every pixel in the image through a fairly simple equation on the approximated partial derivitive created with the Sobel operator.
# So you essentially define the edge lines through the sobel operator (which is the derivative of a window, and hence the gradient is known) and then you determine the likelihood of it being a corner
# through the structure tensor and then through the corner response function

#         # Calculate the elements of the structure tensor
#         IxIx = gx[i][j]**2
#         IyIy = gy[i][j]**2
#         IxIy = gx[i][j] * gy[i][j]
#         row_A.append([IxIx, IxIy, IxIy, IyIy])
#     A.append(row_A)

#         # Calculate the determinant and trace of the structure tensor
#         det = IxIx*IyIy - IxIy**2
#         trace = A[i][j][0] + A[i][j][3]
#         response = det - 0.04 * trace**2 # Use the Harris corner response function
#         if response > threshold:
#             corners.append((response, (j,i)))

# # Threshold the response image to obtain the corners
# threshold = ... # Choose a threshold here
# corners = []
# for i in range(1, img.shape[0]-1):
#     for j in range(1, img.shape[1]-1):
#         if response[i][j] > threshold:
#             corners.append((j, i)) # Note: OpenCV and NumPy use (y, x) indexing, whereas Python uses (x, y) indexing
#             # you are going to want to include the response value here to be able to relate the corners in both the captured images
