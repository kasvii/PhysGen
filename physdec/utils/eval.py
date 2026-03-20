import torch
import numpy as np

def calculate_metrics(true_values, predicted_values):
    """
    Calculate evaluation metrics for pressure field prediction.

    Args:
        true_values: Ground truth values (tensor or array).
        predicted_values: Predicted values (tensor or array).

    Returns:
        Dictionary of metric names to values.
    """
    if torch.is_tensor(true_values):
        true_values = true_values[0].cpu().numpy()
    if torch.is_tensor(predicted_values):
        predicted_values = predicted_values[0].cpu().numpy()

    mse = np.mean((true_values - predicted_values) ** 2)
    mae = np.mean(np.abs(true_values - predicted_values))
    rmse = np.sqrt(mse)
    max_error = np.max(np.abs(true_values - predicted_values))

    rel_l2 = np.mean(np.linalg.norm(true_values - predicted_values, axis=0) /
                      np.linalg.norm(true_values, axis=0))
    rel_l1 = np.mean(np.sum(np.abs(true_values - predicted_values), axis=0) /
                      np.sum(np.abs(true_values), axis=0))

    y_mean = np.mean(true_values, axis=0)
    ss_tot = np.sum((true_values - y_mean) ** 2)
    ss_res = np.sum((true_values - predicted_values) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'max_error': max_error,
        'rel_l2': rel_l2,
        'rel_l1': rel_l1,
        'r_squared': r_squared
    }

def rel_l2_loss_batchwise(pred, target, eps=1e-6):
    """Batch-wise relative L2 loss."""
    num = torch.norm(pred - target, p=2, dim=1)
    denom = torch.norm(target, p=2, dim=1) + eps
    return torch.mean(num / denom)