# Note - the code base for this was built by chatgpt, I am merely adding comments to it to understand how it works.

# Load the image in grayscale
img =  # Load your image here

# Compute the gradient in x and y direction using Sobel filters
threshold = ...
corners = []
gx = []
gy = []
for i in range(1, img.shape[0]-1): # i index = array of pixels at x
    row_gx = []
    row_gy = []
    for j in range(1, img.shape[1]-1): # j = y
        # Compute the gradient in x and y direction using Sobel filters
        Gx = img[i-1][j+1] + 2*img[i][j+1] + img[i+1][j+1] - img[i-1][j-1] - 2*img[i][j-1] - img[i+1][j-1] # top row, - bottom row [[-1,1], [0,1], [1,1], [-1,-1], [-1,0], [-1,1]]
        Gy = img[i+1][j-1] + 2*img[i+1][j] + img[i+1][j+1] - img[i-1][j-1] - 2*img[i-1][j] - img[i-1][j+1]
        row_gx.append(Gx)
        row_gy.append(Gy)
    gx.append(row_gx)
    gy.append(row_gy)

# Compute the elements of the structure tensor

# You are creating the structure tensor for every pixel in the image through a fairly simple equation on the approximated partial derivitive created with the Sobel operator.
# So you essentially define the edge lines through the sobel operator (which is the derivative of a window, and hence the gradient is known) and then you determine the likelihood of it being a corner
# through the structure tensor and then through the corner response function

        # Calculate the elements of the structure tensor
        IxIx = gx[i][j]**2
        IyIy = gy[i][j]**2
        IxIy = gx[i][j] * gy[i][j]
        row_A.append([IxIx, IxIy, IxIy, IyIy])
    A.append(row_A)

        # Calculate the determinant and trace of the structure tensor
        det = IxIx*IyIy - IxIy**2
        trace = A[i][j][0] + A[i][j][3]
        response = det - 0.04 * trace**2 # Use the Harris corner response function
        if response > threshold:
            corners.append((response, (j,i)))

# Threshold the response image to obtain the corners
threshold = ... # Choose a threshold here
corners = []
for i in range(1, img.shape[0]-1):
    for j in range(1, img.shape[1]-1):
        if response[i][j] > threshold:
            corners.append((j, i)) # Note: OpenCV and NumPy use (y, x) indexing, whereas Python uses (x, y) indexing
            # you are going to want to include the response value here to be able to relate the corners in both the captured images
