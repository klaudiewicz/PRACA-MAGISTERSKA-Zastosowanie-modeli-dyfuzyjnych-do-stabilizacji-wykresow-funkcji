import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch

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

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

def plot_hyperparameter_impact(history_list, func_name, arch_name, save_path1=None, save_path2=None):
    df = pd.DataFrame(history_list)
    
    metrics_to_plot = ['MSE', 'L2_Error', 'Wasserstein', 'MAPE', 'Total_Time_s']
    titles = ['Błąd MSE', 'Błąd L2 (%)', 'Odległość Wasserstein', 'MAPE', 'Całkowity czas (s)']
    
    # ==========================================
    # Wykres 1: Wpływ epok 
    # ==========================================
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f"[{arch_name.upper()}] Funkcja: {func_name} - WPŁYW EPOK", fontsize=16, fontweight='bold', y=1.05)
    
    for i, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        sns.lineplot(data=df, x='epochs', y=metric, hue='T', style='schedule', 
                     markers=True, dashes=False, ax=axes[i], palette='Set1')
        axes[i].set_title(title)
        
        # Wymuszenie podziałek co 500 epok
        axes[i].xaxis.set_major_locator(ticker.MultipleLocator(500))
        axes[i].tick_params(axis='x', rotation=45) 
        
        if metric != 'Total_Time_s':
            axes[i].set_yscale('log')
        axes[i].grid(True, alpha=0.3)
        
    plt.tight_layout()    
    if save_path1:
        plt.savefig(save_path1, bbox_inches='tight', dpi=150) 
    plt.show()

    # ==========================================
    # Wykres 2: Wpływ T 
    # ==========================================
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f"[{arch_name.upper()}] Funkcja: {func_name} - WPŁYW KROKÓW DYFUZJI (T)", fontsize=16, fontweight='bold', y=1.05)
    
    # 1. Znajdujemy epokę, która dała najlepszy (najmniejszy) błąd MSE w całym badaniu
    best_epoch = df.loc[df['MSE'].idxmin(), 'epochs']
    
    # 2. Tworzymy przefiltrowany DataFrame dla drugiego wykresu
    # Zostawiamy wiersze, gdzie epoka dzieli się przez 500 (modulo 500 == 0) LUB jest to najlepsza epoka
    df_filtered = df[(df['epochs'] % 500 == 0) | (df['epochs'] == best_epoch)]
    
    # Wyciągamy unikalne wartości T z oryginalnego dataframe'u (żeby osie X się nie popsuły)
    unique_T = sorted(df['T'].unique())
    
    for i, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        # UWAGA: Tutaj przekazujemy `data=df_filtered`, a nie `df`!
        sns.lineplot(data=df_filtered, x='T', y=metric, hue='epochs', style='schedule', 
                     markers=True, dashes=False, ax=axes[i], palette='Set2')
        
        axes[i].set_title(title)
        
        # WYMUSZENIE PODZIAŁEK DLA T
        axes[i].set_xticks(unique_T)
        
        if metric != 'Total_Time_s':
            axes[i].set_yscale('log')
        axes[i].grid(True, alpha=0.3)
        
    plt.tight_layout()
    if save_path2:
        plt.savefig(save_path2, bbox_inches='tight', dpi=150)
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


def plot_average_fundps_heatmap(results_dict, noise_name):
    valid_funcs = [f for f in results_dict.keys() if results_dict[f]['best_metrics'] is not None]
    if not valid_funcs: 
        print(f"Brak poprawnych wyników dla {noise_name}.")
        return
    all_dataframes = []
    for func in valid_funcs:
        df_temp = pd.DataFrame(results_dict[func]['metrics_history'])
        all_dataframes.append(df_temp)
        
    df_all = pd.concat(all_dataframes, ignore_index=True)
    
    max_valid = df_all.loc[df_all['L2_Error'] != float('inf'), 'L2_Error'].max()
    df_all['L2_Error'] = df_all['L2_Error'].replace(float('inf'), max_valid * 1.5)
    
    df_avg = df_all.groupby(['Zeta', 'Steps'])['L2_Error'].mean().reset_index()
    
    best_params = df_avg.loc[df_avg['L2_Error'].idxmin()]
    print(f"--- Najlepsze parametry dla szumu {noise_name.upper()} (Średnia z {len(valid_funcs)} funkcji) ---")
    print(f"Zeta:  {best_params['Zeta']}")
    print(f"Steps: {best_params['Steps']}")
    print(f"Błąd:  {best_params['L2_Error']:.2f}%\n")
    
    pivot_table = df_avg.pivot(index="Zeta", columns="Steps", values="L2_Error")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlOrRd", 
                cbar_kws={'label': 'Średni błąd L2 (%)'}, linewidths=.5)
    
    plt.title(f'Średni błąd L2 dla FunDPS (Szum: {noise_name.upper()})\nUśredniono z {len(valid_funcs)} funkcji', 
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    return df_avg.sort_values(by='L2_Error').head()


import pandas as pd
import matplotlib.pyplot as plt

def plot_fundps_comparison_bars(results_w, results_g, metric_name, metric_title):
    data = []
    funcs = [f for f in results_w.keys() if results_w[f]['best_metrics'] is not None and results_g[f]['best_metrics'] is not None]
    
    if not funcs:
        print(f"Brak danych do wyrysowania dla metryki: {metric_name}")
        return
    
    for f in funcs:
        row = {
            'Funkcja': f.upper(),
            'White Noise': results_w[f]['best_metrics'][metric_name],
            'GRF Noise': results_g[f]['best_metrics'][metric_name]
        }
        data.append(row)
        
    df = pd.DataFrame(data).set_index('Funkcja')  
    
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(kind='bar', ax=ax, width=0.7, color=['#4C72B0', '#DD8452'], edgecolor='black')
    
    ax.set_title(f"Porównanie struktur szumu:\n{metric_title}", fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("Funkcja", fontsize=12, fontweight='bold')
    ax.set_ylabel("Wartość metryki", fontsize=12)
    
    ax.grid(axis='y', linestyle='--', alpha=0.7) 
    plt.xticks(rotation=90) 
    ax.legend(title="Rodzaj szumu", fontsize=10, title_fontsize=11)
    
    plt.tight_layout()
    plt.show()

