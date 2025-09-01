import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from cost_functions import compute_cost, compute_cost_reg
from feature_mapping import map_feature

def test_compute_cost():
    # Test data
    X_test = np.array([[1, 2], [3, 4], [5, 6]])
    y_test = np.array([0, 1, 1])
    w_test = np.array([0.1, 0.2])
    b_test = 0.5
    
    # Expected cost
    expected_cost = 0.7330956193913454
    
    # Compute cost
    cost = compute_cost(X_test, y_test, w_test, b_test)
    
    # Check if the computed cost is close to the expected cost
    assert np.isclose(cost, expected_cost), f"Expected {expected_cost}, got {cost}"
    print("test_compute_cost passed!")

def test_compute_cost_reg():
    # Test data
    X_train = np.array([[0.051267, 0.69956], [-0.092742, 0.68494], [-0.21371, 0.69225]])
    y_train = np.array([1., 1., 1.])
    X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
    initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
    initial_b = 0.5
    lambda_ = 0.5
    
    # Expected cost
    expected_cost = 0.6618252552483948
    
    # Set random seed for reproducibility
    np.random.seed(1)
    
    # Compute cost
    cost = compute_cost_reg(X_mapped, y_train, initial_w, initial_b, lambda_)
    
    # Check if the computed cost is close to the expected cost
    assert np.isclose(cost, expected_cost), f"Expected {expected_cost}, got {cost}"
    print("test_compute_cost_reg passed!")

if __name__ == "__main__":
    test_compute_cost()
    test_compute_cost_reg()