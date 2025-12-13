import numpy as np

def train_multiple_linear_regression(X, y):
    """
    X: feature matrix (n_samples x n_features)
    y: target vector
    """
    # Add bias (column of ones)
    X = np.c_[np.ones(X.shape[0]), X]

    # Normal Equation
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta


if __name__ == "__main__":
    n = int(input("Enter number of samples: "))
    m = int(input("Enter number of features: "))

    print("Enter feature values row-wise:")
    X = []
    for _ in range(n):
        X.append(list(map(float, input().split())))

    y = list(map(float, input("Enter Y values: ").split()))

    X = np.array(X)
    y = np.array(y)

    beta = train_multiple_linear_regression(X, y)

    print("\nRegression Coefficients:")
    for i, b in enumerate(beta):
        print(f"b{i} = {b:.4f}")

    print("\nEnter values to predict:")
    x_new = list(map(float, input().split()))
    x_new = np.array([1] + x_new)

    y_pred = x_new @ beta
    print("Predicted Y:", round(y_pred, 4))
