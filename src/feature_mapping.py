import numpy as np

def map_feature(X1, X2):
    """
    Feature mapping function to polynomial features
    
    Maps the two input features to quadratic features used in the regularization exercise.
    
    Returns a new feature array with more features, comprising of
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    
    Inputs X1, X2 must be the same size
    """
    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)
    
    degree = 6
    out = []
    for i in range(1, degree+1):
        for j in range(i+1):
            out.append((X1**(i-j) * (X2**j)))
    return np.stack(out, axis=1)