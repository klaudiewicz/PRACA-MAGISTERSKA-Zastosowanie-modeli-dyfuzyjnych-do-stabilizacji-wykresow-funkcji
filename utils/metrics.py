# ==========================================
#  METRICS FUNCTIONS
# ==========================================
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import wasserstein_distance

def calculate_metrics(y_true, y_pred, exec_time=None, train_time=None):
    mask = np.abs(y_true) > 1e-3
    mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) if np.any(mask) else 0.0
    
    norm_diff = np.linalg.norm(y_true - y_pred, 2)
    norm_true = np.linalg.norm(y_true, 2)
    l2_error = (norm_diff / (norm_true + 1e-8)) * 100.0

    metrics_dict = {
        'MSE': mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': mape,
        'Wasserstein': wasserstein_distance(y_true, y_pred),
        'L2_Error': l2_error
    }
    
    total_time = 0.0
    if train_time is not None:
        metrics_dict['Train_Time_s'] = train_time
        total_time += train_time
    if exec_time is not None:
        metrics_dict['Sample_Time_s'] = exec_time
        total_time += exec_time
        
    metrics_dict['Total_Time_s'] = total_time
        
    return metrics_dict