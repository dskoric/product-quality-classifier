import numpy as np

def sigmoid(z):
    """
    Compute the sigmoid of z
    
    Args:
        z (ndarray): A scalar, numpy array of any size.
    
    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
    """
    g = 1/(1+np.exp(-z))
    return g

def compute_cost(X, y, w, b):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
    Returns:
      total_cost : (scalar)     cost 
    """
    m, n = X.shape
    
    # Calculate the predictions
    f_wb = sigmoid(X @ w + b)
    
    # Calculate the cost
    cost = -y * np.log(f_wb) - (1 - y) * np.log(1 - f_wb)
    total_cost = np.sum(cost) / m
    
    return total_cost

def compute_cost_reg(X, y, w, b, lambda_=1):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      lambda_ : (scalar, float) Controls amount of regularization
    Returns:
      total_cost : (scalar)     cost 
    """
    m, n = X.shape
    
    # Calls the compute_cost function that you implemented above
    cost_without_reg = compute_cost(X, y, w, b) 
    
    # You need to calculate this value
    reg_cost = 0.
    
    for j in range(n):
        reg_cost_j = w[j]**2
        reg_cost = reg_cost + reg_cost_j
    
    reg_cost = (lambda_/(2 * m)) * reg_cost
    
    # Add the regularization cost to get the total cost
    total_cost = cost_without_reg + reg_cost
    return total_cost