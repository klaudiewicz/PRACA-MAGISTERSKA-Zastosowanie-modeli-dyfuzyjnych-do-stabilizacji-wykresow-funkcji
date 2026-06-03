import os
import time
import pickle
import random
import itertools
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import wasserstein_distance, pearsonr

def calculate_metrics(y_true, y_pred, exec_time=None, train_time=None):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    assert y_true.shape == y_pred.shape, (
        f"Kształty się nie zgadzają: y_true={y_true.shape}, y_pred={y_pred.shape}"
    )
    
    mask = np.abs(y_true) > 1e-3
    mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) if np.any(mask) else 0.0
    
    norm_diff = np.linalg.norm(y_true - y_pred, 2)
    norm_true = np.linalg.norm(y_true, 2)
    l2_error = (norm_diff / (norm_true + 1e-8)) * 100.0

    mse = mean_squared_error(y_true, y_pred)
    
    # SNR
    signal_power = np.mean(y_true**2)
    noise_power = mse
    snr = 10 * np.log10(signal_power / (noise_power + 1e-12))

    # Pearson Correlation
    corr, _ = pearsonr(y_true, y_pred)

    metrics_dict = {
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': mape,
        'Wasserstein': wasserstein_distance(y_true, y_pred),
        'L2_Error': l2_error,
        'SNR': snr,
        'Correlation': corr
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


def aggregate_experiment_metrics(test_functions, architectures_config, cache_dir):
    records = []
    
    for func in test_functions:
        for arch_name, info in architectures_config.items():
            cache_file = os.path.join(cache_dir, f"results_cache_{arch_name}_{func}.pkl")
            
            if not os.path.exists(cache_file):
                continue
                
            with open(cache_file, 'rb') as f:
                saved_results = pickle.load(f)
                
            best_metrics = saved_results.get('best_metrics', {})
            
            record = {
                'Function': func,
                'Architecture': arch_name,
                'Class': info['class'].__name__,
                'Capacity': info['capacity'],
                'LR': info['lr'],
                'SNR': best_metrics.get('SNR', 0),
                'Correlation': best_metrics.get('Correlation', 0),
                'Wasserstein': best_metrics.get('Wasserstein', 0), 
                'MSE': best_metrics.get('MSE', best_metrics.get('reconstruction_mse', 0)),
                'MAE': best_metrics.get('MAE', 0),
                'L2_Error': best_metrics.get('L2_Error', 0),
                'Total_Time_s': best_metrics.get('Total_Time_s', 0)
            }
            records.append(record)
            
    df = pd.DataFrame(records)
    return df

def generate_comparison_tables(df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    for func in df['Function'].unique():
        df_func = df[df['Function'] == func].copy()
        df_func = df_func.sort_values(by='SNR', ascending=False)
        
        csv_path = os.path.join(save_dir, f"Metrics_Table_{func}.csv")
        df_func.to_csv(csv_path, index=False)
        
        best_snr_model = df_func.iloc[0]['Architecture']
        worst_snr_model = df_func.iloc[-1]['Architecture']
        
        print(f"\n[{func.upper()}] ANALIZA METRYK:")
        print(f" -> Najlepszy model (najwyższe SNR): {best_snr_model} ({df_func.iloc[0]['SNR']:.2f} dB)")
        print(f" -> Najgorszy model (najniższe SNR): {worst_snr_model} ({df_func.iloc[-1]['SNR']:.2f} dB)")
        
        print_cols = ['Architecture', 'Capacity', 'LR', 'SNR', 'Correlation', 'Wasserstein', 'MSE', 'Total_Time_s']
        df_latex = df_func[print_cols].copy()
        
        df_latex['SNR'] = df_latex['SNR'].map('{:.2f}'.format)
        df_latex['Correlation'] = df_latex['Correlation'].map('{:.4f}'.format)
        df_latex['Wasserstein'] = df_latex['Wasserstein'].map('{:.4f}'.format) 
        df_latex['MSE'] = df_latex['MSE'].map('{:.2e}'.format)
        df_latex['Total_Time_s'] = df_latex['Total_Time_s'].map('{:.2f}'.format)
        
        latex_path = os.path.join(save_dir, f"Metrics_Table_{func}.tex")
        with open(latex_path, 'w') as f:
            f.write(df_latex.to_latex(index=False, caption=f"Porównanie metryki rekonstrukcji dla funkcji {func}", label=f"tab:metrics_{func}"))

