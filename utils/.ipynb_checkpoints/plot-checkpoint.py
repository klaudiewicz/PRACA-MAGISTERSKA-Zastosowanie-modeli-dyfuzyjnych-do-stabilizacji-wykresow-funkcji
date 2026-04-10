import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch

def plot_hyperparameter_impact(history_list, func_name, arch_name, save_path1=None,save_path2=None):
    df = pd.DataFrame(history_list)
    
    metrics_to_plot = ['MSE', 'L2_Error', 'Wasserstein', 'MAPE', 'Total_Time_s']
    titles = ['Błąd MSE', 'Błąd L2 (%)', 'Odległość Wasserstein', 'MAPE', 'Całkowity czas (s)']
    
    # Wykres 1: Wpływ epok 
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f"[{arch_name.upper()}] Funkcja: {func_name} - WPŁYW EPOK", fontsize=16, fontweight='bold', y=1.05)
    
    for i, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        sns.lineplot(data=df, x='epochs', y=metric, hue='T', style='schedule', 
                     markers=True, dashes=False, ax=axes[i], palette='Set1')
        axes[i].set_title(title)
        if metric != 'Total_Time_s':
            axes[i].set_yscale('log')
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()	
    if save_path1:
        plt.savefig(save_path1, bbox_inches='tight', dpi=150) 
    plt.show()

    # Wykres 2: Wpływ T 
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f"[{arch_name.upper()}] Funkcja: {func_name} - WPŁYW KROKÓW DYFUZJI (T)", fontsize=16, fontweight='bold', y=1.05)
    
    for i, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        sns.lineplot(data=df, x='T', y=metric, hue='epochs', style='schedule', 
                     markers=True, dashes=False, ax=axes[i], palette='Set2')
        axes[i].set_title(title)
        if metric != 'Total_Time_s':
            axes[i].set_yscale('log')
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
	
    if save_path2:
        plt.savefig(save_path2, bbox_inches='tight', dpi=150) # Zapisuje wykres
    plt.show()

def plot_best_reconstruction(data, func_name, arch_name, save_path=None):
    """Rekonstrukcja sygnału dla najlepszej konfiguracji."""
    plt.figure(figsize=(10, 6))
    x = data['x']
    
    plt.scatter(x, data['y_noisy'], color='lightgray', alpha=0.5, s=20, label='Zaszumione dane wejściowe')
    plt.plot(x, data['y_true'], label='Oryginalna funkcja (Ground Truth)', color='#1f77b4', linewidth=3)
    plt.plot(x, data['y_denoised'], label='Odszumione (Najlepszy Model)', color='#d62728', linewidth=2.5, linestyle='--')
    
    best_cfg = data['best_config']
    title = (f"[{arch_name.upper()}] Rekonstrukcja funkcji {func_name}\n"
             f"Najlepsze parametry: Epoki={best_cfg['epochs']}, T={best_cfg['T']}, Beta={best_cfg['schedule']}")
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("x", fontsize=12)
    plt.ylabel("f(x)", fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
	
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150) 
    plt.show()

def plot_fundps_heatmaps(results_dict, noise_name):
    valid_funcs = [f for f in results_dict.keys() if results_dict[f]['best_metrics'] is not None]
    if not valid_funcs: return
    
    num_funcs = len(valid_funcs)
    fig, axes = plt.subplots(1, num_funcs, figsize=(6 * num_funcs, 5))
    fig.suptitle(f'Analiza ablacyjna (L2 Error %): FunDPS z szumem {noise_name.upper()}', fontsize=16, fontweight='bold')
    
    if num_funcs == 1: axes = [axes]
        
    for i, func in enumerate(valid_funcs):
        hist = results_dict[func]['metrics_history']
        df = pd.DataFrame(hist)
        max_valid = df.loc[df['L2_Error'] != float('inf'), 'L2_Error'].max()
        df['L2_Error'] = df['L2_Error'].replace(float('inf'), max_valid * 1.5)
        
        pivot_table = df.pivot(index="Zeta", columns="Steps", values="L2_Error")
        
        sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlOrRd", ax=axes[i], 
                    cbar_kws={'label': 'Błąd L2 (%)'}, linewidths=.5)
        axes[i].set_title(f'Funkcja: {func.upper()}')
        axes[i].invert_yaxis()
        
    plt.tight_layout()
    plt.show()
	
def plot_training_loss(results_dict, noise_name):
    valid_funcs = [f for f in results_dict.keys() if 'prior_loss_history' in results_dict[f]]
    if not valid_funcs: return
    
    plt.figure(figsize=(10, 5))
    plt.title(f"Krzywa uczenia modelu priora (FunDPS - Szum {noise_name.upper()})", fontsize=14, fontweight='bold')
    
    for func in valid_funcs:
        loss_curve = results_dict[func]['prior_loss_history']
        plt.plot(loss_curve, label=f'Funkcja: {func.upper()}', alpha=0.8)
        
    plt.yscale('log')
    plt.xlabel('Epoka')
    plt.ylabel('Loss (MSE) - Skala Logarytmiczna')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(loc='best', fontsize=9, ncol=2)
    plt.tight_layout()
    plt.show()

