import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from gradient_functions import compute_gradient, compute_gradient_reg
from feature_mapping import map_feature

def test_compute_gradient():
    # Test data
    X_test = np.array([[1, 2], [3, 4], [5, 6]])
    y_test = np.array([0, 1, 1])
    w_test = np.array([0.1, 0.2])
    b_test = 0.5
    
    # Expected gradients
    expected_dj_db = 0.29806179
    expected_dj_dw = np.array([0.98614423, 1.97228846])
    
    # Compute gradients
    dj_db, dj_dw = compute_gradient(X_test, y_test, w_test, b_test)
    
    # Check if the computed gradients are close to the expected gradients
    assert np.isclose(dj_db, expected_dj_db), f"Expected dj_db {expected_dj_db}, got {dj_db}"
    assert np.allclose(dj_dw, expected_dj_dw), f"Expected dj_dw {expected_dj_dw}, got {dj_dw}"
    print("test_compute_gradient passed!")

def test_compute_gradient_reg():
    # Test data
    X_train = np.array([[0.051267, 0.69956], [-0.092742, 0.68494], [-0.21371, 0.69225]])
    y_train = np.array([1., 1., 1.])
    X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
    initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
    initial_b = 0.5
    lambda_ = 0.5
    
    # Expected gradients
    expected_dj_db = 0.07138288792343662
    expected_dj_dw_first_few = np.array([-0.010386028450548, 0.011409852883280, 0.053627346327457, 0.003140278267313])
    
    # Set random seed for reproducibility
    np.random.seed(1)
    
    # Compute gradients
    dj_db, dj_dw = compute_gradient_reg(X_mapped, y_train, initial_w, initial_b, lambda_)
    
    # Check if the computed gradients are close to the expected gradients
    assert np.isclose(dj_db, expected_dj_db), f"Expected dj_db {expected_dj_db}, got {dj_db}"
    assert np.allclose(dj_dw[:4], expected_dj_dw_first_few), f"Expected first few elements of dj_dw {expected_dj_dw_first_few}, got {dj_dw[:4]}"
    print("test_compute_gradient_reg passed!")

if __name__ == "__main__":
    test_compute_gradient()
    test_compute_gradient_reg()