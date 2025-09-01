import numpy as np

def load_data(filename):
    """
    Load data from a file into numpy arrays.
    
    Args:
        filename (str): Path to the data file
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target vector
    """
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, :2]
    y = data[:, 2]
    return X, y