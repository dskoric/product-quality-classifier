import numpy as np
import matplotlib.pyplot as plt
from feature_mapping import map_feature
from cost_functions import sigmoid

def plot_data(X, y, pos_label="y=1", neg_label="y=0"):
    """
    Plots the data points X and y into a new figure. Plots the data 
    points with * for the positive examples and o for the negative examples.
    
    Args:
        X (ndarray): Input features, shape (m, n)
        y (ndarray): Target labels, shape (m,)
        pos_label (str): Label for positive examples
        neg_label (str): Label for negative examples
    """
    positive = y == 1
    negative = y == 0
    
    # Plot examples
    plt.plot(X[positive, 0], X[positive, 1], 'k*', label=pos_label)
    plt.plot(X[negative, 0], X[negative, 1], 'yo', label=neg_label)

def plot_decision_boundary(w, b, X, y):
    """
    Plots the decision boundary for logistic regression.
    
    Args:
        w (ndarray): Parameters, shape (n,)
        b (scalar): Bias parameter
        X (ndarray): Input features, shape (m, n)
        y (ndarray): Target labels, shape (m,)
    """
    # Plot the original data
    plot_data(X[:, 0:2], y)
    
    if X.shape[1] <= 2:
        # If only 2 features, create a line plot
        plot_x = np.array([min(X[:, 0]), max(X[:, 0])])
        plot_y = (-1. / w[1]) * (w[0] * plot_x + b)
        
        plt.plot(plot_x, plot_y, c="b")
    else:
        # Create a contour plot for higher dimensional features
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        
        z = np.zeros((len(u), len(v)))
        
        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = sigmoid(np.dot(map_feature(u[i], v[j]), w) + b)
        
        # Important to transpose z before calling contour
        z = z.T
        
        # Plot z = 0
        plt.contour(u, v, z, levels=[0.5], colors="g")