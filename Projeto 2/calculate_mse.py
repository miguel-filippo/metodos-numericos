def calculate_mse(y_true, y_pred):
    """
    Calculate the Mean Squared Error (MSE) between true and predicted values.

    Args:
        y_true (list or np.array): True target values.
        y_pred (list or np.array): Predicted values.

    Returns:
        float: The Mean Squared Error.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Length of true and predicted values must be the same.")

    squared_errors = [(true - pred) ** 2 for true, pred in zip(y_true, y_pred)]
    mse = sum(squared_errors) / len(squared_errors)
    return mse

if __name__ == "__main__":
    # Example usage
    y_true = open('best model predictions.txt', 'r').read().strip().split('\n')
    y_true = [float(x) for x in y_true]
    y_pred = open('group G predictions.txt', 'r').read().strip().split('\n')
    y_pred = [float(x) for x in y_pred]

    mse = calculate_mse(y_true, y_pred)
    print(f"Mean Squared Error: {mse}")
