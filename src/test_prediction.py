import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.prediction import predict

def test_predict():
    # Test data
    X_test = np.array([[1, 2], [3, 4], [5, 6]])
    w_test = np.array([0.1, 0.2])
    b_test = 0.5
    
    # Expected predictions
    expected_predictions = np.array([1., 1., 1.])
    
    # Compute predictions
    predictions = predict(X_test, w_test, b_test)
    
    # Check if the computed predictions are close to the expected predictions
    assert np.allclose(predictions, expected_predictions), f"Expected {expected_predictions}, got {predictions}"
    print("test_predict passed!")

if __name__ == "__main__":
    test_predict()