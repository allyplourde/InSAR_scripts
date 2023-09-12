import numpy as np



def green_for_lls(A, b):
    """
    Compute Green's function for linear least-squares

    Parameters:
         A (np.array): coefficient matrix
         b (np,array): right-hand side vector

    Returns:
        np.array: Green's function
    """
    print(A)
    q, r = np.linalg.qr(A) # Q R decomposition
    return np.linalg.solve(r, np.dot(q.T, b))


if __name__ == "__main__":

    A = np.array([[1, 2], [3, 4]])
    b = np.array([1, 2])
    print(green_for_lls(A, b))