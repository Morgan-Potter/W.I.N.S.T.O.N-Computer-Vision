import numpy as np

# Step 5: Triangulation
def triangulation(correspondences, P):
    K, RT = decompose_projection_matrix(P)
    R = RT[:, :3]
    t = RT[:, 3]

    image_points_homogeneous = concatenate_ones(correspondences)

    K_inv = inverse_matrix(K)

    camera_coords = dot_product(K_inv, transpose(image_points_homogeneous))

    world_coords = dot_product(transpose(R), subtract(camera_coords, reshape(t, (3, 1))))

    return world_coords


# Step 6: Refinement and Optimization (Optional)


# Step 7: Real-World Coordinates
def get_real_world_coordinates(correspondences, P):
    world_coords = triangulation(correspondences, P)

    # Step 6: Refinement and Optimization (Optional)

    # Step 7: Real-World Coordinates
    return world_coords


# Function to decompose the camera projection matrix into intrinsic and extrinsic matrices
def decompose_projection_matrix(P):
    K, RT = qr_decomposition(inverse_matrix(submatrix(P, (3, 3))))
    K /= K[2, 2]
    RT = inverse_matrix(RT)
    return K, RT


# Function to estimate camera projection matrix using DLT algorithm
def estimate_camera_projection_matrix(image_points, world_points):
    assert len(image_points) == len(world_points)
    num_points = len(image_points)

    A = zeros((2 * num_points, 12))
    for i in range(num_points):
        X, Y, Z = world_points[i]
        x, y = image_points[i]

        A[2 * i] = [-X, -Y, -Z, -1, 0, 0, 0, 0, x * X, x * Y, x * Z, x]
        A[2 * i + 1] = [0, 0, 0, 0, -X, -Y, -Z, -1, y * X, y * Y, y * Z, y]

    _, _, V = singular_value_decomposition(A)

    P = reshape(V[-1], (3, 4))

    return P

# Reshape a matrix
def reshape(matrix, shape):
    reshaped_matrix = []
    idx = 0
    for _ in range(shape[0]):
        row = []
        for _ in range(shape[1]):
            row.append(matrix[idx])
            idx += 1
        reshaped_matrix.append(row)
    return reshaped_matrix


# Perform QR decomposition
def qr_decomposition(matrix):
    rows, cols = len(matrix), len(matrix[0])
    Q = zeros((rows, cols))
    R = deepcopy(matrix)

    for i in range(cols):
        norm = np.linalg.norm(R[i:, i])
        sign = 1 if R[i, i] >= 0 else -1
        v = [R[j, i] / (sign * norm) if j >= i else 0 for j in range(rows)]
        v[i] += 1
        beta = -sign * norm

        R[i:, i:] -= beta * np.outer(v[i:], v[i:])
        Q[i:, i:] -= beta * np.outer(v[i:], Q[i:, i:])

    return Q, R


# Perform singular value decomposition (SVD)
# Perform singular value decomposition (SVD)
def singular_value_decomposition(matrix):
    A = deepcopy(matrix)
    rows, cols = len(A), len(A[0])
    U = zeros((rows, rows))
    V = zeros((cols, cols))
    s = [0] * min(rows, cols)

    B = dot_product(transpose(A), A)
    eigenvalues, eigenvectors = power_iteration(B)

    for i in range(cols):
        V[i] = eigenvectors[i]
        norm = np.linalg.norm(V[i])
        V[i] = [V[i][j] / norm if norm != 0 else 0 for j in range(cols)]

    for i in range(rows):
        s[i] = np.sqrt(eigenvalues[i])
        U[i] = dot_product(A, [V[j] for j in range(cols)])

    return U, s, V


# Perform power iteration to find dominant eigenvalues and eigenvectors
def power_iteration(matrix):
    epsilon = 1e-6
    rows, cols = len(matrix), len(matrix[0])
    x = [1] * cols

    for _ in range(1000):
        x_next = dot_product(matrix, [x])[0]
        norm = np.linalg.norm(x_next)
        x_next = [x_next[i] / norm if norm != 0 else 0 for i in range(cols)]
        
        if np.linalg.norm(subtract([x], [x_next])) < epsilon:
            break

        x = x_next

    eigenvalue = dot_product(dot_product([x], matrix), transpose([x]))[0][0]
    eigenvector = x

    return eigenvalue, eigenvector



# Create a matrix of zeros
def zeros(shape):
    return [[0] * shape[1] for _ in range(shape[0])]

# Deep copy a matrix
def deepcopy(matrix):
    return [row[:] for row in matrix]

# Utility functions for matrix operations
def transpose(matrix):
    return list(map(list, zip(*matrix)))


def dot_product(matrix1, matrix2):
    return [[sum(a * b for a, b in zip(row, col)) for col in transpose(matrix2)] for row in matrix1]


def subtract(matrix1, matrix2):
    return [[a - b for a, b in zip(row1, row2)] for row1, row2 in zip(matrix1, matrix2)]


def inverse_matrix(matrix):
    det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    inv_det = 1.0 / det
    return [[matrix[1][1] * inv_det, -matrix[0][1] * inv_det], [-matrix[1][0] * inv_det, matrix[0][0] * inv_det]]


def submatrix(matrix, size):
    return [row[:size[1]] for row in matrix[:size[0]]]


def concatenate_ones(matrix):
    return [row + [1] for row in matrix]


# Usage example
image_points = [[1,2], [3,4], [9,1]]  # 2D image points
world_points = [[5,6,1], [1,8,9], [1,5,6]]  # 3D world points

# Step 1: Solve correspondences through Harris corner detection (already done)

# Step 2: Match correspondences (already done)

# Step 3: Verify correspondences (already done)

# Step 4: Camera Calibration (already done)

# Step 5: Estimate camera projection matrix using DLT algorithm
P = estimate_camera_projection_matrix(image_points, world_points)

# Steps 5 to 7: Triangulation, Refinement, and Real-World Coordinates
world_coordinates = get_real_world_coordinates(image_points, P)
print(world_coordinates)