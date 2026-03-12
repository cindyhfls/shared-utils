import numpy as np

def is_matrix_orthogonal(matrix, tolerance=1e-9):
    """
    Checks if a square matrix is orthogonal.
    """
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError, "Not a square matrix!"

    # Calculate the product of the matrix and its transpose
    # Use the @ operator for matrix multiplication
    product = matrix @ matrix.T

    # Create an identity matrix of the same shape
    identity = np.identity(matrix.shape[0])

    # Check if the product is close to the identity matrix
    # np.allclose is essential for floating-point comparisons
    return np.allclose(product, identity, atol=tolerance)
