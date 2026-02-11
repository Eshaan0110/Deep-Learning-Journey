import numpy as np

#Mean Squared Error (Regression Loss)

'''MSE = mean((y_true - y_pred)^2)

    Penalizes large errors more heavily.
    Used in regression tasks.'''

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

#Mean Absolute Error (Regression Loss)

'''MAE = mean(|y_true - y_pred|)

    Penalizes all errors equally.
    Used in regression tasks.
    MAE = mean(|y_true - y_pred|)

    Less sensitive to outliers than MSE'''

def mae_loss(y_true, y_pred):
    return np.mean(np.abs(y_true -  y_pred))

#binary cross-entropy loss (Classification Loss)

def binary_cross_entropy(y_true, y_pred):
    """
    BCE = -mean( y*log(p) + (1-y)*log(1-p) )

    Used for binary classification.
    """
    epsilon = 1e-9  # Prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    return -np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * np.log(1 - y_pred)
    )


# Categorical Cross Entropy (Multi-class)


def categorical_cross_entropy(y_true, y_pred):
    """
    CCE = -sum( y_true * log(y_pred) )

    y_true should be one-hot encoded.
    """
    epsilon = 1e-9
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
