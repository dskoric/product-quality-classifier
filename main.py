import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_data
from feature_mapping import map_feature
from cost_functions import compute_cost_reg
from gradient_functions import compute_gradient_reg
from optimization import gradient_descent
from data.prediction import predict
from utils import plot_data, plot_decision_boundary

def main():
    # Load the dataset
    X_train, y_train = load_data("../data/ex2data2.txt")
    
    # Print the first 5 values of X_train and y_train
    print("X_train:", X_train[:5])
    print("Type of X_train:", type(X_train))
    print("y_train:", y_train[:5])
    print("Type of y_train:", type(y_train))
    
    # Check the dimensions of the variables
    print('The shape of X_train is: ' + str(X_train.shape))
    print('The shape of y_train is: ' + str(y_train.shape))
    print('We have m = %d training examples' % (len(y_train)))
    
    # Visualize the data
    plot_data(X_train, y_train, pos_label="Accepted", neg_label="Rejected")
    plt.ylabel('Microchip Test 2') 
    plt.xlabel('Microchip Test 1') 
    plt.legend(loc="upper right")
    plt.savefig('../output/data_visualization.png')
    plt.show()
    
    # Feature mapping
    print("Original shape of data:", X_train.shape)
    X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
    print("Shape after feature mapping:", X_mapped.shape)
    
    # Print the first elements of X_train and mapped_X
    print("X_train[0]:", X_train[0])
    print("mapped X_train[0]:", X_mapped[0])
    
    # Initialize fitting parameters
    np.random.seed(1)
    initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
    initial_b = 1.
    
    # Set regularization parameter lambda_
    lambda_ = 0.01
    
    # Some gradient descent settings
    iterations = 10000
    alpha = 0.01
    
    # Run gradient descent
    w, b, J_history, _ = gradient_descent(X_mapped, y_train, initial_w, initial_b, 
                                         compute_cost_reg, compute_gradient_reg, 
                                         alpha, iterations, lambda_)
    
    # Plot the decision boundary
    plot_decision_boundary(w, b, X_mapped, y_train)
    plt.ylabel('Microchip Test 2') 
    plt.xlabel('Microchip Test 1') 
    plt.legend(loc="upper right")
    plt.savefig('../output/decision_boundary.png')
    plt.show()
    
    # Compute accuracy on the training set
    p = predict(X_mapped, w, b)
    print('Train Accuracy: %f' % (np.mean(p == y_train) * 100))

if __name__ == "__main__":
    main()