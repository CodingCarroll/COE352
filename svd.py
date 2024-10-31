# svd.py

import numpy as np

def svd(A):
    """
    Compute the Singular Value Decomposition of a matrix A.

    Parameters:
        A (ndarray): The input matrix of shape (m, n).

    Returns:
        U_full (ndarray): The left singular vectors.
        singular_values (ndarray): The singular values.
        Vt_full (ndarray): The right singular vectors transposed.
        condition_number (float): The condition number of the matrix.
        A_inv (ndarray): The inverse of A if it exists.

    Raises:
        ValueError: If the matrix is singular and has no inverse.
    """
    m, n = A.shape
    k = min(m, n)

    # Compute A A^T and A^T A
    AAt = np.dot(A, A.T)
    AtA = np.dot(A.T, A)

    # Compute eigenvalues and eigenvectors of A A^T (for U)
    eigenvalues_U, U = np.linalg.eigh(AAt)
    idx = np.argsort(eigenvalues_U)[::-1]
    eigenvalues_U = eigenvalues_U[idx]
    U = U[:, idx]

    # Compute singular values
    singular_values = np.sqrt(np.maximum(eigenvalues_U[:k], 0))

    # Compute eigenvalues and eigenvectors of A^T A (for V)
    eigenvalues_V, V = np.linalg.eigh(AtA)
    idx = np.argsort(eigenvalues_V)[::-1]
    eigenvalues_V = eigenvalues_V[idx]
    V = V[:, idx]

    # Ensure U and V are full-sized orthogonal matrices
    if m > n:
        U_full = U
    else:
        additional_vectors = np.eye(m)[:, k:]
        U_full = np.hstack((U, additional_vectors))
        U_full, _ = np.linalg.qr(U_full)

    if n > m:
        V_full = V
    else:
        additional_vectors = np.eye(n)[:, k:]
        V_full = np.hstack((V, additional_vectors))
        V_full, _ = np.linalg.qr(V_full)

    Vt_full = V_full.T

    # Compute condition number
    tol = 1e-10
    non_zero_singular_values = singular_values[singular_values > tol]
    if len(non_zero_singular_values) == 0:
        raise ValueError("Matrix is singular and has no inverse.")
    condition_number = singular_values.max() / singular_values.min()

    # Compute inverse of A if it exists
    S_inv = np.diag(1 / singular_values)
    A_inv = V_full @ S_inv @ U_full.T

    return U_full, singular_values, Vt_full, condition_number, A_inv
