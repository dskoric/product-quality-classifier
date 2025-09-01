import numpy as np
from cost_functions import sigmoid

def compute_gradient(X, y, w, b):
    """
    Computes the gradient for logistic regression
 
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
    Returns
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b. 
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 
    """
    m, n = X.shape
    
    # Calculate the predictions
    f_wb = sigmoid(X @ w + b)
    
    # Calculate the gradients
    err = f_wb - y
    dj_db = np.sum(err) / m
    dj_dw = (X.T @ err) / m
    
    return dj_db, dj_dw

def compute_gradient_reg(X, y, w, b, lambda_=1): 
    """
    Computes the gradient for logistic regression with regularization
 
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      lambda_ : (scalar,float)  regularization constant
    Returns
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b. 
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 
    """
    m, n = X.shape
    
    dj_db, dj_dw = compute_gradient(X, y, w, b)
    
    for j in range(n): 
        dj_dw_j_reg = (lambda_ / m) * w[j]
        # Add the regularization term to the corresponding element of dj_dw
        dj_dw[j] = dj_dw[j] + dj_dw_j_reg
        
    return dj_db, dj_dw