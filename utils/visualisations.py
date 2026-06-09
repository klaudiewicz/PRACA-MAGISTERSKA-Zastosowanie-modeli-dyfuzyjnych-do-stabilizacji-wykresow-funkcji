import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import numpy as np
import torch
import re
import math
import pickle
import torch.nn as nn
from IPython.display import Image, display
import os
import scipy.stats as stats 
from matplotlib.lines import Line2D
from typing import Dict, Any, Callable, Type
import plotly.express as px
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from models.edm1d import FunDPSSampler, FunDPSExperimentRunner, generate_grf_1d, SigmaEmbedding, EDMDenoiser1D, ForwardOperator
from sklearn.ensemble import RandomForestRegressor
from models.ddpm1d import DDPM1D, SinusoidalPositionEmbeddings, get_beta_schedule

sns.set_style("whitegrid")
PALETTE=['#fde725','#5ec962','#3b528b']
ARCH_CONFIG = {
    'MLP':    {'color': '#fde725', 'ls': ':',  'marker': 'o'}, 
    'Conv1D': {'color': '#5ec962', 'ls': '-.', 'marker': 's'},  
    'UNet':   {'color': '#3b528b', 'ls': '-',  'marker': '^'}   
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'axes.edgecolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'font.size': 12,           
    'axes.titlesize': 12,     
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,      
    'xtick.labelsize': 10,     
    'ytick.labelsize': 10,     
    'legend.fontsize': 10,     
    'lines.linewidth': 1.5, 
    'axes.linewidth': 0.8,  
    'grid.linewidth': 0.5,  
    'grid.linestyle': '--',
    'grid.alpha': 0.5,         
    'figure.dpi': 300,        
    'savefig.dpi': 300,       
    'figure.figsize': [16 / 2.54, 9 / 2.54],
    'figure.autolayout': True
})

def _get_base_arch(arch_name: str) -> str:
    """Ekstrahuje nazwę architektury"""
    return arch_name.split('_')[0]

# ############################## EXP 1 ##############################
def plot_summary_heatmap(results: dict, func_name: str, lr: float, t: int, save_path: str):

    data = []
    for arch, caps in results.items():
        for cap_name, metrics in caps.items():
            data.append({
                'Architektura': arch,
                'Pojemność': cap_name,
                'mu': metrics['test_mu'],
                'std': metrics['test_std']
            })
    
    df = pd.DataFrame(data)
    all_caps = ['C32', 'C64', 'C128', 'C256', 'C512']
    active_caps = [c for c in all_caps if c in df['Pojemność'].values]
    
    df['Pojemność'] = pd.Categorical(df['Pojemność'], categories=active_caps, ordered=True)
    df['Architektura'] = pd.Categorical(df['Architektura'], categories=['MLP', 'Conv1D', 'UNet'], ordered=True)
    
    pivot_mu = df.pivot(index="Architektura", columns="Pojemność", values="mu")
    pivot_std = df.pivot(index="Architektura", columns="Pojemność", values="std")
    
    annot_array = np.empty_like(pivot_mu.values, dtype=object)
    for i in range(pivot_mu.shape[0]):
        for j in range(pivot_mu.shape[1]):
            mu_val = pivot_mu.iloc[i, j]
            std_val = pivot_std.iloc[i, j]
            
            if pd.isna(mu_val):
                annot_array[i, j] = "Brak"
            else:
                if mu_val < 1e-4:
                    annot_array[i, j] = f"{mu_val:.1e}\n±{std_val:.1e}"
                else:
                    annot_array[i, j] = f"{mu_val:.4f}\n±{std_val:.4f}"
    
    plt.figure() 
    sns.heatmap(pivot_mu, 
                annot=annot_array, 
                fmt="", 
                cmap="viridis", 
                norm=LogNorm(), 
                cbar_kws={'label': 'Średni błąd testowy MSE (log)'},
                linewidths=0.5,
                linecolor='black') 
    
    plt.title(f"Funkcja: {func_name.upper()}\n(Parametry: LR = {lr}, T = {t})", pad=15)
    plt.yticks(rotation=0) 
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()



def generate_lr_summary_plots(checkpoints_dir, plots_dir):

    learning_rates = sorted([1e-3, 5e-4, 1e-4]) 
    samplers = ["Sine", "Chirp", "Hard"]
    architectures = ["MLP", "Conv1D", "UNet"]
    capacities = ['C32', 'C64', 'C128', 'C256']
    t_steps_options = [80, 100]

    raw_data = {lr: {arch: [] for arch in architectures} for lr in learning_rates}

    for lr in learning_rates:
        for s in samplers:
            for arch in architectures:
                for cap in capacities:
                    for t in t_steps_options:
                        prefix = f"{s}_{arch}_{cap}_LR{lr}_T{t}"
                        file_path = os.path.join(checkpoints_dir, f"{prefix}_stats.pth")
                        
                        if os.path.exists(file_path):
                            try:
                                stats = torch.load(file_path, map_location='cpu', weights_only=False)
                                if 'test_mu' in stats:
                                    raw_data[lr][arch].append(stats['test_mu'])
                            except Exception: continue

    arch_averages = {arch: [] for arch in architectures}
    global_averages = []

    for lr in learning_rates:
        all_lr_scores = []
        for arch in architectures:
            scores = raw_data[lr][arch]
            mean_score = np.mean(scores) if len(scores) > 0 else np.nan
            arch_averages[arch].append(mean_score)
            all_lr_scores.extend(scores)
        
        global_mean = np.mean(all_lr_scores) if len(all_lr_scores) > 0 else np.nan
        global_averages.append(global_mean)

    summary_plots_dir = os.path.join(plots_dir, 'stats')
    os.makedirs(summary_plots_dir, exist_ok=True)

    plt.figure()
    markers = {'MLP': 'o', 'Conv1D': 's', 'UNet': '^'}
    lr_labels = [str(lr) for lr in learning_rates]

    for arch in architectures:
        color = ARCH_CONFIG.get(arch, {}).get('color', '#000000')
        ls = ARCH_CONFIG.get(arch, {}).get('ls', '-')
        
        plt.plot(
            lr_labels, 
            arch_averages[arch], 
            label=f"Architektura: {arch}", 
            color=color, 
            linestyle=ls,
            marker=markers.get(arch, 'o'), 
            linewidth=2.0, 
            markersize=7.5
        )

    #plt.title("Wpływ współczynnika uczenia")
    plt.xlabel("Learning Rate")
    plt.ylabel("Średni błąd Test MSE")
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.legend(frameon=True, facecolor='white', edgecolor='black')
    plt.tight_layout()
    
    path_archs = os.path.join(summary_plots_dir, "lr_vs_mse_per_arch.png")
    plt.savefig(path_archs)
    plt.show()

    plt.figure()
    plt.plot(
        lr_labels, 
        global_averages, 
        color='#333333',
        marker='D', 
        linewidth=2.5, 
        markersize=8, 
        label="Średnia globalna"
    )

    #plt.title("Globalny wpływ współczynnika uczenia")
    plt.xlabel("Learning Rate")
    plt.ylabel("Globalny średni błąd Test MSE")
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.legend(frameon=True, facecolor='white', edgecolor='black')
    path_global = os.path.join(summary_plots_dir, "lr_vs_mse_global_average.png")
    plt.savefig(path_global)
    plt.show()

	
def plot_learning_curves_with_ci(
    train_mat: np.ndarray, 
    val_mat: np.ndarray, 
    arch: str, 
    cap: str, 
    func: str, 
    save_path: str,
    confidence: float = 0.95
) -> None:

    base_arch = _get_base_arch(arch)
    style = ARCH_CONFIG.get(base_arch, {'color': '#000000', 'ls': '-', 'marker': ''})
    main_color = style['color']
    
    epochs = np.arange(1, train_mat.shape[1] + 1)
    n_runs = train_mat.shape[0]
    
    train_mean = np.mean(train_mat, axis=0)
    train_std = np.std(train_mat, axis=0, ddof=1)
    train_ci = stats.t.ppf((1 + confidence) / 2., n_runs-1) * (train_std / np.sqrt(n_runs))
    
    val_mean = np.mean(val_mat, axis=0)
    val_std = np.std(val_mat, axis=0, ddof=1)
    val_ci = stats.t.ppf((1 + confidence) / 2., n_runs-1) * (val_std / np.sqrt(n_runs))
    
    plt.figure()
    
    plt.plot(epochs, train_mean, label='Błąd treningowy', 
             color=main_color, linestyle=style['ls'], linewidth=2)
    plt.fill_between(epochs, train_mean - train_ci, train_mean + train_ci, 
                     color=main_color, alpha=0.12, label=f'Trening {int(confidence*100)}% CI')
    
    val_color = '#777777'
    plt.plot(epochs, val_mean, label='Błąd walidacyjny', 
             color=val_color, linestyle='--', linewidth=2) 
    plt.fill_between(epochs, val_mean - val_ci, val_mean + val_ci, 
                     color=val_color, alpha=0.08, label=f'Walidacja {int(confidence*100)}% CI')
    
    plt.title(f"[{arch}] Pojemność: {cap}\nFunkcja: {func}")
    plt.xlabel("Epoka")
    plt.ylabel("MSE (log)")
    plt.yscale('log')
    
    plt.legend(loc='upper right', framealpha=0.9, edgecolor='black')
    plt.grid(True, which="both")
    
    plt.savefig(save_path, bbox_inches='tight') 
    plt.close()

def plot_summary_heatmap(results: dict, func_name: str, lr: float, t: int, save_path: str):

    data = []
    for arch, caps in results.items():
        for cap_name, metrics in caps.items():
            data.append({
                'Architektura': arch,
                'Pojemność': cap_name,
                'mu': metrics['test_mu'],
                'std': metrics['test_std']
            })
    
    df = pd.DataFrame(data)
    all_caps = ['C32', 'C64', 'C128', 'C256', 'C512']
    active_caps = [c for c in all_caps if c in df['Pojemność'].values]
    
    df['Pojemność'] = pd.Categorical(df['Pojemność'], categories=active_caps, ordered=True)
    df['Architektura'] = pd.Categorical(df['Architektura'], categories=['MLP', 'Conv1D', 'UNet'], ordered=True)
    
    pivot_mu = df.pivot(index="Architektura", columns="Pojemność", values="mu")
    pivot_std = df.pivot(index="Architektura", columns="Pojemność", values="std")
    
    annot_array = np.empty_like(pivot_mu.values, dtype=object)
    for i in range(pivot_mu.shape[0]):
        for j in range(pivot_mu.shape[1]):
            mu_val = pivot_mu.iloc[i, j]
            std_val = pivot_std.iloc[i, j]
            
            if pd.isna(mu_val):
                annot_array[i, j] = "Brak"
            else:
                if mu_val < 1e-4:
                    annot_array[i, j] = f"{mu_val:.1e}\n±{std_val:.1e}"
                else:
                    annot_array[i, j] = f"{mu_val:.4f}\n±{std_val:.4f}"
    
    plt.figure() 
    sns.heatmap(pivot_mu, 
                annot=annot_array, 
                fmt="", 
                cmap="viridis", 
                norm=LogNorm(), 
                cbar_kws={'label': 'Średni błąd testowy MSE (log)'},
                linewidths=0.5,
                linecolor='black') 
    
    plt.title(f"Funkcja: {func_name.upper()}\n(Parametry: LR = {lr}, T = {t})", pad=15)
    plt.yticks(rotation=0) 
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_lr_comparison(lr_data: dict, arch: str, cap: str, func: str, save_path: str):

    base_color = ARCH_CONFIG.get(arch, {'color': '#000000'})['color']
    lrs = sorted(list(lr_data.keys()), reverse=True)
    num_lrs = len(lrs)
    
    colors = [mcolors.to_hex([0.1 + (0.6 * i / max(1, num_lrs-1))] * 3) for i in range(num_lrs)]
    
    fig, axes = plt.subplots(1, 2, figsize=(16 / 2.54, 7 / 2.54), sharey=True)
    
    for idx, lr in enumerate(lrs):
        c = colors[idx]
        train_mat = lr_data[lr]['train']
        val_mat = lr_data[lr]['val']
        epochs = np.arange(1, train_mat.shape[1] + 1)
        
        train_mu = np.mean(train_mat, axis=0)
        train_std = np.std(train_mat, axis=0, ddof=1)
        val_mu = np.mean(val_mat, axis=0)
        val_std = np.std(val_mat, axis=0, ddof=1)
        
        ls_styles = ['-', '--', ':', '-.']
        current_ls = ls_styles[idx % len(ls_styles)]
        
        # --- Panel 1: Trening ---
        axes[0].plot(epochs, train_mu, label=f'LR = {lr}', color=c, linestyle=current_ls, linewidth=1.5)
        axes[0].fill_between(epochs, train_mu - train_std, train_mu + train_std, color=c, alpha=0.08)
        
        # --- Panel 2: Walidacja ---
        axes[1].plot(epochs, val_mu, label=f'LR = {lr}', color=c, linestyle=current_ls, linewidth=1.5)
        axes[1].fill_between(epochs, val_mu - val_std, val_mu + val_std, color=c, alpha=0.08)

    axes[0].set_title("Błąd treningowy")
    axes[0].set_xlabel("Epoka")
    axes[0].set_ylabel("MSE (log)")
    axes[0].set_yscale('log')
    axes[0].grid(True, which="both")
    
    axes[1].set_title("Błąd walidacyjny")
    axes[1].set_xlabel("Epoka")
    axes[1].set_yscale('log')
    axes[1].grid(True, which="both")
    axes[1].legend(loc='upper right', framealpha=0.9, edgecolor='black')
    
    fig.suptitle(f"[{arch}] Wpływ współczynnika uczenia\n Pojemność: {cap} | Funkcja: {func.upper()}", y=1.05)
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def visualize_styled_stats_final(folder):

    all_data = []
    if not os.path.exists(folder):
        return
        
    files = [f for f in os.listdir(folder) if f.endswith('_stats.pth')]
    for f in files:
        parts = f.replace('_stats.pth', '').split('_')
        func_name = parts[0].capitalize() 
        arch_name = parts[1]
        try:
            stat = torch.load(os.path.join(folder, f), map_location='cpu', weights_only=False)
            all_data.append({
                'Funkcja': func_name,
                'Architektura': arch_name,
                'Test_MSE': stat['test_mu'],
                'Test_Std': stat['test_std']
            })
        except Exception: 
            continue

    df_raw = pd.DataFrame(all_data)
    if df_raw.empty: return
        
    idx_min = df_raw.groupby(['Architektura', 'Funkcja'])['Test_MSE'].idxmin()  
    df_grouped = df_raw.loc[idx_min].reset_index(drop=True)

    all_funcs = ['Sine', 'Chirp', 'Hard']
    all_archs = ['MLP', 'Conv1D', 'UNet'] 
    
    idx = pd.MultiIndex.from_product([all_archs, all_funcs], names=['Architektura', 'Funkcja'])
    df_final = df_grouped.set_index(['Architektura', 'Funkcja']).reindex(idx).reset_index()
    df_final['Test_MSE'] = df_final['Test_MSE'].replace(0, np.nan)
    
    df_final['Architektura'] = pd.Categorical(df_final['Architektura'], categories=all_archs, ordered=True)
    df_final['Funkcja'] = pd.Categorical(df_final['Funkcja'], categories=all_funcs, ordered=True)

    plt.figure()

    custom_palette = {arch: ARCH_CONFIG[arch]['color'] for arch in all_archs if arch in ARCH_CONFIG}
    
    ax = sns.barplot(
        data=df_final, 
        x='Funkcja', 
        y='Test_MSE', 
        hue='Architektura', 
        palette=custom_palette,
        edgecolor='black',
        linewidth=1.0,
        alpha=1.0, 
        order=all_funcs
    )

    for container, arch in zip(ax.containers, all_archs):
        arch_data = df_final[df_final['Architektura'] == arch].sort_values('Funkcja')
        x_coords = [rect.get_x() + rect.get_width() / 2.0 for rect in container]
        
        y_vals = arch_data['Test_MSE'].values
        y_errs = arch_data['Test_Std'].values
        
        lower_err = np.clip(y_errs, 0, y_vals - 1e-10) 
        upper_err = y_errs
        asymmetric_err = [lower_err, upper_err]
        
        ax.errorbar(
            x=x_coords, 
            y=y_vals, 
            yerr=asymmetric_err, 
            fmt='none', 
            c='#000000', 
            capsize=4, 
            elinewidth=1.2,
            alpha=0.9
        )
        
        for x, y, u_err in zip(x_coords, y_vals, upper_err):
            if pd.notna(y):
                ax.text(x, y + u_err, f'{y:.1e}', 
                        ha='center', va='bottom', fontsize=8.5, rotation=45, color='black')

    max_val = (df_final['Test_MSE'] + df_final['Test_Std']).max()
    if pd.notna(max_val):
        plt.ylim(top=max_val * 1.5) 
        
    plt.ylabel("Błąd testowy MSE")
    plt.xlabel("Klasa")
    
    min_val = df_final['Test_MSE'].min()
    if pd.notna(min_val):
        plt.ylim(bottom=min_val * 0.1) 
    
    legend_elements = [Line2D([0], [0], color=custom_palette[arch], lw=6, label=arch) for arch in all_archs]
    plt.legend(handles=legend_elements, title="Architektura", loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., edgecolor='black')

    os.makedirs('../images/experiment1/stats', exist_ok=True)    
    plt.savefig('../images/experiment1/stats/porownanie_architektur.png', bbox_inches='tight')
    plt.show() 

def plot_mse_vs_params_vertical(folder='checkpoints1'):

    rows = []
    if not os.path.exists(folder): return
        
    for f in os.listdir(folder):
        if not f.endswith('_stats.pth'): continue
        parts = f.replace('_stats.pth', '').split('_')
        if len(parts) < 5: continue
            
        try:
            func_name = parts[0].capitalize() 
            arch = parts[1]
            stat = torch.load(os.path.join(folder, f), map_location='cpu', weights_only=False)
            test_mse = stat.get('test_mu')
            num_params = stat.get('num_params')
            
            if test_mse is not None and num_params is not None:
                rows.append({
                    'Funkcja': func_name, 'Architektura': arch, 'Liczba parametrów': num_params, 'MSE': test_mse
                })
        except Exception: continue
            
    df = pd.DataFrame(rows)
    if df.empty: return
        
    df_agg = df.groupby(['Funkcja', 'Architektura', 'Liczba parametrów'])['MSE'].min().reset_index()
    func_order = ['Sine', 'Chirp', 'Hard']
    existing_funcs = [s for s in func_order if s in df_agg['Funkcja'].unique()]

    active_archs = df_agg['Architektura'].unique()
    custom_palette = {arch: ARCH_CONFIG[arch]['color'] for arch in active_archs if arch in ARCH_CONFIG}

    g = sns.relplot(
        data=df_agg, 
        x='Liczba parametrów', 
        y='MSE', 
        hue='Architektura', 
        style='Architektura',
        row='Funkcja',
        row_order=existing_funcs,
        kind='line',
        markers=['o', 's', '^'][:len(active_archs)], 
        dashes=[(2,2), (4,2), (1,0)][:len(active_archs)], 
        linewidth=2.0,
        markersize=8,
        palette=custom_palette,
        height=3.8,                    
        aspect=1.8                     
    )
    
    g.set(xscale="log", yscale="log")
    g.set_axis_labels("Liczba parametrów", "Najlepszy testowy błąd MSE (log)")
    g.set_titles(row_template="Funkcja: {row_name}")

    for ax in g.axes.flat:
        ax.grid(True, which="major", ls="-", alpha=0.4, color='#cccccc')
        ax.grid(True, which="minor", ls=":", alpha=0.2, color='#eeeeee')

    os.makedirs('../images/experiment1/stats', exist_ok=True)
    plt.savefig('../images/experiment1/stats/mse_vs_params.png', bbox_inches='tight')
    plt.show()


def plot_experiment_ablation_boxplots(df: pd.DataFrame, save_dir: str = "../images/experiment1/stats", use_log_scale: bool = False) -> None:

    if df.empty:
        print("[UWAGA] Przekazany DataFrame jest pusty. Przerywam generowanie wykresów.")
        return


    arch_palette = {arch: config['color'] for arch, config in ARCH_CONFIG.items()}
    arch_order = ['MLP', 'Conv1D', 'UNet']

    os.makedirs(save_dir, exist_ok=True)
    
    def format_and_save(ax, filename):
        if use_log_scale:
            ax.set_yscale('log')
        ax.grid(True, which="both", axis='y', linestyle='--', alpha=0.5)
        ax.set_ylabel("Test błąd MSE", fontsize=11)
        ax.tick_params(axis='both', labelsize=10)
        ax.set_xlabel('') 
        plt.tight_layout()
        
        full_save_path = os.path.join(save_dir, filename)
        plt.savefig(full_save_path, bbox_inches='tight', dpi=300)
        plt.show()
        plt.close() 
    boxplot_viridisscale_kwargs = dict(
        boxprops=dict(edgecolor='black'),
        capprops=dict(color='black'),
        whiskerprops=dict(color='black'),
        medianprops=dict(color='black', linewidth=1.5)
    )

    # --- Wykres 1: Wpływ architektury ---
    plt.figure(figsize=(10, 5))
    ax1 = sns.boxplot(
        data=df, x='Architektura', y='Test_MSE', 
        order=arch_order, hue='Architektura', hue_order=arch_order, 
        palette=arch_palette, legend=False,
        **boxplot_viridisscale_kwargs
    )
    ax1.set_title('Wpływ rodzaju architektury', fontsize=12, pad=10)
    format_and_save(ax1, "wplyw_architektury.png")

    # --- Wykres 2: Wpływ pojemności ---
    plt.figure(figsize=(10, 5))
    ax2 = sns.boxplot(
        data=df, x='Pojemność', y='Test_MSE', 
        hue='Pojemność', palette='viridis', legend=False,
        **boxplot_viridisscale_kwargs
    )
    ax2.set_title('Wpływ pojemności', fontsize=12, pad=10)
    format_and_save(ax2, "wplyw_pojemnosci.png")

    # --- Wykres 3: Wpływ współczynnika uczenia ---
    plt.figure(figsize=(10, 5))
    ax3 = sns.boxplot(
        data=df, x='LR', y='Test_MSE', 
        hue='LR', palette='viridis', legend=False,
        **boxplot_viridisscale_kwargs
    )
    ax3.set_title('Wpływ współczynnika uczenia', fontsize=12, pad=10)
    format_and_save(ax3, "wplyw_lr.png")

    # --- Wykres 4: Wpływ liczby kroków (T) ---
    plt.figure(figsize=(10, 5))
    ax4 = sns.boxplot(
        data=df, x='T', y='Test_MSE', 
        hue='T', palette='viridis', legend=False,
        **boxplot_viridisscale_kwargs
    )
    ax4.set_title('Wpływ liczby kroków (T)', fontsize=12, pad=10)
    format_and_save(ax4, "wplyw_krokow_t.png")


def create_split_heatmap(folder):
    """
    Wizualizuje mapy cieplne (osobne dla każdej funkcji).
    """
    all_data = []
    if not os.path.exists(folder): return
        
    for f in os.listdir(folder):
        if not f.endswith('_stats.pth'): continue
        parts = f.replace('_stats.pth', '').split('_')
        if len(parts) < 5: continue
            
        func_name = parts[0].capitalize()
        arch = parts[1]
        cap_str = parts[2]
        t_str = parts[4]
        
        try:
            cap = int(cap_str.replace('C', ''))
            t_steps = int(t_str.replace('T', ''))
            stat = torch.load(os.path.join(folder, f), map_location='cpu', weights_only=False)
            test_mse = stat['test_mu'] 
            
            all_data.append({
                'Funkcja': func_name, 'Architektura': arch, 'Pojemność': cap, 'T': t_steps, 'MSE': test_mse
            })
        except Exception: continue
            
    df_raw = pd.DataFrame(all_data)
    if df_raw.empty: return

    df = df_raw[df_raw['Pojemność'].isin([32, 64, 128, 256, 512])]
    df = df[df['T'].isin([80, 100])]
    df = df[df['Architektura'].isin(['MLP', 'Conv1D', 'UNet'])]
    
    os.makedirs('../images/experiment1/stats', exist_ok=True)

    for func in df['Funkcja'].unique():
        df_func = df[df['Funkcja'] == func]
        df_agg = df_func.groupby(['Architektura', 'Pojemność', 'T'])['MSE'].min().reset_index()
        pivot_df = df_agg.pivot_table(index='Architektura', columns=['Pojemność', 'T'], values='MSE')
        
        arch_order = ['MLP', 'Conv1D', 'UNet']
        pivot_df = pivot_df.reindex([a for a in arch_order if a in pivot_df.index])
    
        min_err = pivot_df.min().min()
        annot_fmt = ".1e" if min_err < 1e-4 else ".4f"
    
        plt.figure(figsize=(10, 4.5))
        
        ax = sns.heatmap(
            pivot_df, 
            annot=True, 
            cmap='viridis', 
            fmt=annot_fmt, 
            norm=LogNorm(), 
            cbar_kws={'label': 'Najniższy błąd testowy MSE (skala log)'}, 
            linewidths=0.5, 
            linecolor='black'
        )
        
        num_t_steps = len(df['T'].unique())
        for i in range(1, len(pivot_df.columns.levels[0])):
            ax.axvline(x=i * num_t_steps, color='black', linewidth=2.5)
            
        capacities = pivot_df.columns.levels[0]
        for i, cap in enumerate(capacities):
            ax.text(i * num_t_steps + (num_t_steps / 2), -0.3, f"C{cap}", 
                    ha='center', va='bottom', fontsize=11, fontweight='bold', clip_on=False)
            
        plt.title(f'Funkcja: {func}', pad=35)
        plt.xlabel('Liczba kroków dyfuzji (T)', labelpad=10)
        plt.ylabel('Architektura')
        
        ax.set_xticklabels([f"T={t}" for cap, t in pivot_df.columns], rotation=0)
    
        plt.savefig(f'../images/experiment1/stats/heatmap_split_cells_{func}.png', bbox_inches='tight')
        plt.close()     

def plot_mse_vs_params_global(folder='checkpoints1'):
    """
    Rysuje pojedynczy, globalny wykres zależności uśrednionego błędu MSE 
    od liczby parametrów modelu (agregacja po wszystkich funkcjach testowych)
    z zachowaniem standardów akademickich.
    """
    rows = []
    if not os.path.exists(folder):
        print(f"Folder {folder} nie istnieje.")
        return
        
    for f in os.listdir(folder):
        if not f.endswith('_stats.pth'): 
            continue
        parts = f.replace('_stats.pth', '').split('_')
        if len(parts) < 5: 
            continue
            
        try:
            func_name = parts[0].capitalize() 
            arch = parts[1]
            stat = torch.load(os.path.join(folder, f), map_location='cpu', weights_only=False)
            test_mse = stat.get('test_mu')
            num_params = stat.get('num_params')
            
            if test_mse is not None and num_params is not None:
                rows.append({
                    'Funkcja': func_name, 
                    'Architektura': arch, 
                    'Liczba parametrów': num_params, 
                    'MSE': test_mse
                })
        except Exception: 
            continue
            
    df = pd.DataFrame(rows)
    if df.empty:
        print("Brak poprawnych danych do wygenerowania wykresu globalnego.")
        return
        

    df_global_agg = df.groupby(['Architektura', 'Liczba parametrów'])['MSE'].mean().reset_index()
    
    all_archs = ['MLP', 'Conv1D', 'UNet']
    active_archs = [a for a in all_archs if a in df_global_agg['Architektura'].unique()]
    df_global_agg['Architektura'] = pd.Categorical(df_global_agg['Architektura'], categories=active_archs, ordered=True)
    df_global_agg = df_global_agg.sort_values('Architektura')

    custom_palette = {arch: ARCH_CONFIG[arch]['color'] for arch in active_archs if arch in ARCH_CONFIG}
    
    plt.figure(figsize=(12 / 2.54, 8 / 2.54)) 
    ax = plt.gca()
    
    sns.lineplot(
        data=df_global_agg,
        x='Liczba parametrów',
        y='MSE',
        hue='Architektura',
        style='Architektura',
        markers={'MLP': 'o', 'Conv1D': 's', 'UNet': '^'},
        dashes={'MLP': (1, 2), 'Conv1D': (4, 2), 'UNet': (1, 0)}, 
        palette=custom_palette,
        linewidth=2.0,
        markersize=7.5,
        ax=ax
    )
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    ax.set_xlabel("Liczba parametrów modelu ($N_{\\mathrm{params}}$)", fontsize=10, labelpad=8)
    ax.set_ylabel("Globalny średni błąd Test MSE (log)", fontsize=10, labelpad=8)
    ##ax.set_title("Globalne prawa skalowania złożoności modeli\n(Średnia ważona ze wszystkich funkcji testowych)", pad=14, fontsize=11, fontweight='bold')
    
    ax.grid(True, which="major", ls="-", alpha=0.4, color='#cccccc')
    ax.grid(True, which="minor", ls=":", alpha=0.2, color='#eeeeee')
    ax.tick_params(axis='both', labelsize=9)
    
    ax.legend(title="Architektura", loc='best', frameon=True, facecolor='white', edgecolor='black', fontsize=9, title_fontsize=9)
    
    os.makedirs('../images/experiment1/stats', exist_ok=True)
    save_path = '../images/experiment1/stats/mse_vs_params_global.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"Pomyślnie zapisano globalny wykres złożoności: {save_path}")

def plot_learning_curves_unified(func_type, folder, target_lr=0.0001, target_t=80):
    """
    Ujednolicony wykres krzywych uczenia z unikalnymi stylami przerywania w odcieniach szarości.
    """
    if not os.path.exists(folder): return
    
    func_type = func_type.capitalize()
    files = [f for f in os.listdir(folder) if f.startswith(func_type) and f.endswith('_stats.pth')]
    
    cap_settings = {
        'C32':  {'alpha': 0.5, 'lw': 1.0},
        'C64':  {'alpha': 0.7, 'lw': 1.5},
        'C128': {'alpha': 0.9, 'lw': 2.0},
        'C256': {'alpha': 1.0, 'lw': 2.5}
    }
    
    arch_order = ['MLP', 'Conv1D', 'UNet']
    cap_order = ['C32', 'C64', 'C128', 'C256']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    found_any = False
    handles, labels = [], []

    for arch in arch_order:
        for cap in cap_order:
            target_prefix = f"{func_type}_{arch}_{cap}_LR{target_lr}_T{target_t}"
            matching_file = [f for f in files if f.startswith(target_prefix)]
            
            if matching_file:
                found_any = True
                f = matching_file[0]
                stat = torch.load(os.path.join(folder, f), map_location='cpu', weights_only=False)
                
                train_curve = stat['train_mat'].mean(axis=0)
                val_curve = stat['val_mat'].mean(axis=0)
                epochs = np.arange(1, len(train_curve) + 1)
                
                style = ARCH_CONFIG.get(arch, {'color': 'black', 'ls': '-'})
                cap_style = cap_settings[cap]
                label_name = f"{arch} ({cap})"
                
                # Wykres treningowy (lewa kolumna)
                line, = ax1.plot(
                    epochs, train_curve, 
                    color=style['color'], 
                    linestyle=style['ls'],
                    linewidth=cap_style['lw'],
                    alpha=cap_style['alpha'],
                    label=label_name
                )
                
                # Wykres walidacyjny (prawa kolumna)
                ax2.plot(
                    epochs, val_curve, 
                    color=style['color'], 
                    linestyle=style['ls'],
                    linewidth=cap_style['lw'],
                    alpha=cap_style['alpha']
                )
                
                handles.append(line)
                labels.append(label_name)

    if not found_any:
        plt.close()
        return

    #fig.suptitle(f"Funkcja: {func_type} (LR = {target_lr}, T = {target_t})", fontsize=14, fontweight='bold')

    #ax1.set_title("Błąd treningowy", fontsize=12)
    ax1.set_xlabel("Epoka")
    ax1.set_ylabel("MSE")
    ax1.grid(True, which="both", ls="--", alpha=0.4)

    #ax2.set_title("Błąd walidacyjny", fontsize=12)
    ax2.set_xlabel("Epoka")
    ax2.grid(True, which="both", ls="--", alpha=0.4)
    
    ax2.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', title="Architektura (Pojemność)", edgecolor='black')
    plt.tight_layout()
    
    os.makedirs('../images/experiment1/stats', exist_ok=True)
    save_name = f"../images/experiment1/stats/convergence_combined_{func_type}_LR{target_lr}_T{target_t}.png"
    plt.savefig(save_name, bbox_inches='tight')
    plt.show()
	

def create_split_heatmap(folder):
    all_data = []
    if not os.path.exists(folder): return
        
    for f in os.listdir(folder):
        if not f.endswith('_stats.pth'): continue
        parts = f.replace('_stats.pth', '').split('_')
        if len(parts) < 5: continue
            
        func_name = parts[0].capitalize()
        arch = parts[1]
        cap_str = parts[2]
        t_str = parts[4]
        
        try:
            cap = int(cap_str.replace('C', ''))
            t_steps = int(t_str.replace('T', ''))
            stat = torch.load(os.path.join(folder, f), map_location='cpu', weights_only=False)
            test_mse = stat['test_mu'] 
            
            all_data.append({
                'Funkcja': func_name, 'Architektura': arch, 'Pojemność': cap, 'T': t_steps, 'MSE': test_mse
            })
        except Exception: continue
            
    df_raw = pd.DataFrame(all_data)
    if df_raw.empty: return

    df = df_raw[df_raw['Pojemność'].isin([32, 64, 128, 256, 512])]
    df = df[df['T'].isin([80, 100])]
    df = df[df['Architektura'].isin(['MLP', 'Conv1D', 'UNet'])]
    
    os.makedirs('../images/experiment1/stats', exist_ok=True)

    for func in df['Funkcja'].unique():
        df_func = df[df['Funkcja'] == func]
        df_agg = df_func.groupby(['Architektura', 'Pojemność', 'T'])['MSE'].min().reset_index()
        pivot_df = df_agg.pivot_table(index='Architektura', columns=['Pojemność', 'T'], values='MSE')
        
        arch_order = ['MLP', 'Conv1D', 'UNet']
        pivot_df = pivot_df.reindex([a for a in arch_order if a in pivot_df.index])
    
        min_err = pivot_df.min().min()
        annot_fmt = ".1e" if min_err < 1e-4 else ".4f"
    
        plt.figure(figsize=(10, 4.5))
        
        ax = sns.heatmap(
            pivot_df, 
            annot=True, 
            cmap='viridis', 
            fmt=annot_fmt, 
            norm=LogNorm(), 
            cbar_kws={'label': 'Najniższy błąd testowy MSE (skala log)'}, 
            linewidths=0.5, 
            linecolor='black'
        )
        
        num_t_steps = len(df['T'].unique())
        for i in range(1, len(pivot_df.columns.levels[0])):
            ax.axvline(x=i * num_t_steps, color='black', linewidth=2.5)
            
        capacities = pivot_df.columns.levels[0]
        for i, cap in enumerate(capacities):
            ax.text(i * num_t_steps + (num_t_steps / 2), -0.3, f"C{cap}", 
                    ha='center', va='bottom', fontsize=11, fontweight='bold', clip_on=False)
            
        plt.title(f'Funkcja: {func}', pad=35)
        plt.xlabel('Liczba kroków dyfuzji (T)', labelpad=10)
        plt.ylabel('Architektura')
        
        ax.set_xticklabels([f"T={t}" for cap, t in pivot_df.columns], rotation=0)
    
        plt.savefig(f'../images/experiment1/stats/heatmap_split_cells_{func}.png', bbox_inches='tight')
        plt.close()     


def visualize_styled_stats_final(folder):

    all_data = []
    if not os.path.exists(folder):
        return
        
    files = [f for f in os.listdir(folder) if f.endswith('_stats.pth')]
    for f in files:
        parts = f.replace('_stats.pth', '').split('_')
        func_name = parts[0].capitalize() 
        arch_name = parts[1]
        try:
            stat = torch.load(os.path.join(folder, f), map_location='cpu', weights_only=False)
            all_data.append({
                'Funkcja': func_name,
                'Architektura': arch_name,
                'Test_MSE': stat['test_mu'],
                'Test_Std': stat['test_std']
            })
        except Exception: 
            continue

    df_raw = pd.DataFrame(all_data)
    if df_raw.empty: return
        
    idx_min = df_raw.groupby(['Architektura', 'Funkcja'])['Test_MSE'].idxmin()  
    df_grouped = df_raw.loc[idx_min].reset_index(drop=True)

    all_funcs = ['Sine', 'Chirp', 'Hard']
    all_archs = ['MLP', 'Conv1D', 'UNet'] 
    
    idx = pd.MultiIndex.from_product([all_archs, all_funcs], names=['Architektura', 'Funkcja'])
    df_final = df_grouped.set_index(['Architektura', 'Funkcja']).reindex(idx).reset_index()
    df_final['Test_MSE'] = df_final['Test_MSE'].replace(0, np.nan)
    
    df_final['Architektura'] = pd.Categorical(df_final['Architektura'], categories=all_archs, ordered=True)
    df_final['Funkcja'] = pd.Categorical(df_final['Funkcja'], categories=all_funcs, ordered=True)

    plt.figure()

    custom_palette = {arch: ARCH_CONFIG[arch]['color'] for arch in all_archs if arch in ARCH_CONFIG}
    
    ax = sns.barplot(
        data=df_final, 
        x='Funkcja', 
        y='Test_MSE', 
        hue='Architektura', 
        palette=custom_palette,
        edgecolor='black',
        linewidth=1.0,
        alpha=1.0, 
        order=all_funcs
    )

    for container, arch in zip(ax.containers, all_archs):
        arch_data = df_final[df_final['Architektura'] == arch].sort_values('Funkcja')
        x_coords = [rect.get_x() + rect.get_width() / 2.0 for rect in container]
        
        y_vals = arch_data['Test_MSE'].values
        y_errs = arch_data['Test_Std'].values
        
        lower_err = np.clip(y_errs, 0, y_vals - 1e-10) 
        upper_err = y_errs
        asymmetric_err = [lower_err, upper_err]
        
        ax.errorbar(
            x=x_coords, 
            y=y_vals, 
            yerr=asymmetric_err, 
            fmt='none', 
            c='#000000', 
            capsize=4, 
            elinewidth=1.2,
            alpha=0.9
        )
        
        for x, y, u_err in zip(x_coords, y_vals, upper_err):
            if pd.notna(y):
                ax.text(x, y + u_err, f'{y:.1e}', 
                        ha='center', va='bottom', fontsize=8.5, rotation=45, color='black')

    max_val = (df_final['Test_MSE'] + df_final['Test_Std']).max()
    if pd.notna(max_val):
        plt.ylim(top=max_val * 1.5) 
        
    plt.ylabel("Błąd testowy MSE")
    plt.xlabel("Klasa")
    
    min_val = df_final['Test_MSE'].min()
    if pd.notna(min_val):
        plt.ylim(bottom=min_val * 0.1) 
    
    legend_elements = [Line2D([0], [0], color=custom_palette[arch], lw=6, label=arch) for arch in all_archs]
    plt.legend(handles=legend_elements, title="Architektura", loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., edgecolor='black')

    os.makedirs('../images/experiment1/stats', exist_ok=True)    
    plt.savefig('../images/experiment1/stats/porownanie_architektur.png', bbox_inches='tight')
    plt.show() 
def load_full_experiment_data(folder='checkpoints1'):

    all_data = []
    if not os.path.exists(folder):
        print(f"Folder {folder} nie istnieje.")
        return pd.DataFrame()
        
    for f in os.listdir(folder):
        if not f.endswith('_stats.pth'):
            continue
            
        parts = f.replace('_stats.pth', '').split('_')
        if len(parts) < 5:
            continue
            
        try:
            function = parts[0].capitalize()
            arch = parts[1]
            cap = int(parts[2].replace('C', ''))
            lr = float(parts[3].replace('LR', ''))
            t_steps = int(parts[4].replace('T', ''))
            
            stat = torch.load(os.path.join(folder, f), map_location='cpu', weights_only=False)
            
            all_data.append({
                'Funkcja': function,
                'Architektura': arch,
                'Pojemność': cap,
                'LR': lr,
                'T': t_steps,
                'Train_MSE': stat.get('train_mu', np.nan),
                'Val_MSE': stat.get('val_mu', np.nan),
                'Test_MSE': stat.get('test_mu', np.nan),
                'Test_STD': stat.get('test_std', np.nan),
                'Liczba parametrów': stat.get('num_params', np.nan)
            })
        except Exception as e:
            print(f"Błąd ładowania pliku {f}: {e}")
            continue
            
    return pd.DataFrame(all_data)


def analyze_generalization_gap(df, top_n=10):
    """Oblicza lukę generalizacyjną."""
    df_gap = df.copy()
    df_gap['Gap_Val_Train'] = df_gap['Val_MSE'] - df_gap['Train_MSE']
    
    df_gap = df_gap.sort_values('Gap_Val_Train', ascending=False)
    cols = ['Funkcja', 'Architektura', 'Pojemność', 'LR', 'Test_MSE', 'Train_MSE', 'Val_MSE', 'Gap_Val_Train']
    return df_gap[cols].head(top_n)


def analyze_stability_cv(df, top_n=10):
    df_cv = df.copy()
    # CV = (Odchylenie standardowe / Średnia) * 100%
    df_cv['CV_%'] = (df_cv['Test_STD'] / df_cv['Test_MSE']) * 100
    
    most_unstable = df_cv.sort_values('CV_%', ascending=False).head(top_n)
    most_stable = df_cv.sort_values('CV_%', ascending=True).head(top_n)
    
    cols = ['Funkcja', 'Architektura', 'Pojemność', 'LR', 'Test_MSE', 'Test_STD', 'CV_%']
    return most_unstable[cols], most_stable[cols]

	
def analyze_worst_combinations(df, top_n=10):
    df_worst = df.sort_values('Test_MSE', ascending=False)
    cols = ['Funkcja', 'Architektura', 'Pojemność', 'LR', 'T', 'Test_MSE']
    return df_worst[cols].head(top_n)

def get_scientific_sdedit_candidates(df: pd.DataFrame) -> pd.DataFrame:

    all_candidates = []
    
    for func in df['Funkcja'].unique():
        func_df = df[df['Funkcja'] == func].copy()
        candidates_for_func = []
        
        top_3 = func_df.nsmallest(3, 'Test_MSE').copy()
        top_3['Powód wyboru'] = 'Najniższy błąd dla tego sygnału'
        candidates_for_func.append(top_3)
        
        df_rest = func_df.drop(top_3.index)
        
        for arch in ['Conv1D', 'MLP']:
            arch_df = df_rest[df_rest['Architektura'] == arch]
            if not arch_df.empty:
                best_arch = arch_df.nsmallest(1, 'Test_MSE').copy()
                best_arch['Powód wyboru'] = f'Najlepszy w klasie {arch}'
                candidates_for_func.append(best_arch)
                df_rest = df_rest.drop(best_arch.index)
                
        underfit = df_rest[df_rest['Pojemność'] == 'C32']
        if not underfit.empty:
            idx = len(underfit) // 2
            mid_underfit = underfit.sort_values('Test_MSE').iloc[idx:idx+1].copy()
            mid_underfit['Powód wyboru'] = 'Model niedouczony (C32)'
            candidates_for_func.append(mid_underfit)
            df_rest = df_rest.drop(mid_underfit.index)
            
        available_overfit_caps = ['C256', 'C128', 'C64','C32']
        overfit = pd.DataFrame()
        
        for cap in available_overfit_caps:
            overfit = df_rest[df_rest['Pojemność'] == cap]
            if not overfit.empty:
                break 
                
        if not overfit.empty:
            worst_overfit = overfit.nlargest(1, 'Test_MSE').copy()
            worst_overfit['Powód wyboru'] = f'Model przeuczony ({worst_overfit["Pojemność"].iloc[0]})'
            candidates_for_func.append(worst_overfit)

        all_candidates.extend(candidates_for_func)

    final_df = pd.concat(all_candidates, ignore_index=True)
    
    cols_to_return = ['Funkcja', 'Architektura', 'Pojemność', 'LR', 'T', 'Test_MSE', 'Powód wyboru']
    cols_to_return = [c for c in cols_to_return if c in final_df.columns]
    
    return final_df[cols_to_return]

# ############################## EXP 1 ##############################

def generate_experiment_summary_from_files(folder):
    """
    Zbiera metryki i hiperparametry z plików i zwraca je w postaci ramki danych DataFrame.
    """
    rows = []
    if not os.path.exists(folder):
        return pd.DataFrame()

    for f in os.listdir(folder):
        if not f.endswith('_stats.pth'):
            continue

        parts = f.replace('_stats.pth', '').split('_')
        if len(parts) < 5:
            continue

        try:
            func_name = parts[0].capitalize() 
            arch = parts[1]
            cap_name = parts[2]
            lr = float(parts[3].replace('LR', ''))
            t_steps = int(parts[4].replace('T', ''))

            stat = torch.load(os.path.join(folder, f), map_location='cpu', weights_only=False)
            
            test_mu = stat.get('test_mu', 0)
            test_std = stat.get('test_std', 0)
            num_params = stat.get('num_params', 0) 

            rows.append({
                'Funkcja': func_name,
                'Architektura': arch,
                'Pojemność': cap_name,
                'LR': lr,
                'T': t_steps,
                'MSE': test_mu,
                'Odchylenie Std': test_std,
                'Liczba parametrów': num_params,
                'Stabilność (CV %)': (test_std / test_mu) * 100 if test_mu > 1e-12 else 0
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by='MSE', ascending=True).reset_index(drop=True)
        
    return df

	

# ############################## EXP 2 ##############################

def plot_hyperparameter_heatmaps(history_df, func_name, arch_name, save_path):
    if history_df.empty or len(history_df) < 2 or 'MSE' not in history_df.columns: return
    try:
        pivot_table = history_df.pivot_table(index='T', columns='schedule', values='MSE', aggfunc='mean')
    except Exception: return

    plt.figure()
    ax = sns.heatmap(pivot_table, annot=True, fmt=".2e", cmap="viridis", 
                     cbar_kws={'label': 'Średni błąd testowy MSE'}, linewidths=.5, square=True, linecolor='black')
    ax.invert_yaxis()
    #plt.title(f"[{arch_name}] Wpływ hiperparametrów na jakość rekonstrukcji\nFunkcja: {func_name.upper()}", pad=15)
    plt.xlabel("Harmonogram szumu (Schedule)")
    plt.ylabel("Kroki dyfuzji (T)")
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_time_vs_quality(history_df, func_name, arch_name, save_path):
    if history_df.empty or 'exec_time' not in history_df.columns or history_df['exec_time'].sum() == 0: return

    plt.figure()
    sns.scatterplot(data=history_df, x='exec_time', y='MSE', hue='schedule', size='T', 
                    palette=['#333333', '#888888'], sizes=(60, 200), alpha=0.9, edgecolor='black')
    
    plt.yscale('log')
    #plt.title(f'[{arch_name}] Czas względem błędu MSE\nFunkcja: {func_name.upper()}')
    plt.xlabel('Czas generowania [s]')
    plt.ylabel('Błąd testowy MSE (log)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Parametry", edgecolor='black')
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_noisy_reconstruction(x, y_true, y_noisy, y_pred, func_name, arch_name, save_path):
    plt.figure(figsize=(6.5, 4.2)) 
    
    base_arch = _get_base_arch(arch_name)
    arch_cfg = ARCH_CONFIG.get(base_arch, {'color': '#000000', 'ls': '-', 'marker': ''})
    
    plt.plot(x, y_noisy, color='#e8e8e8', alpha=0.5, linewidth=0.8, 
             label='Funkcja zaszumiona (Start SDEdit)', zorder=1)
    
    plt.plot(x, y_true, color='#d3d3d3', linewidth=3.5, 
             label='Funkcja oryginalna', zorder=2)
    
    plt.plot(x, y_pred, color=arch_cfg['color'], linestyle=arch_cfg['ls'], linewidth=1.6, 
             label='Funkcja odszumiona', zorder=3)
    
    plt.xlabel('Oś X', fontsize=9)
    plt.ylabel('Amplituda', fontsize=9)
    
    plt.grid(True, linestyle=':', alpha=0.5)
    
    y_min, y_max = y_true.min(), y_true.max()
    plt.ylim(y_min - 0.4, y_max + 0.4)
    
    plt.legend(loc='upper right', framealpha=1.0, edgecolor='black', shadow=False, fontsize=8)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8.5)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_fft_spectrum(x_true, y_true, y_pred, func_name, arch_name, save_path):
    yf_true = np.abs(np.fft.rfft(y_true))
    yf_pred = np.abs(np.fft.rfft(y_pred))
    xf = np.fft.rfftfreq(len(x_true), d=(x_true[1] - x_true[0]))

    plt.figure()
    base_arch = _get_base_arch(arch_name)
    arch_cfg = ARCH_CONFIG.get(base_arch, {'color': 'black', 'ls': '-', 'marker': ''})
    
    plt.plot(xf, yf_true, color='black', label='Funkcja oryginalna (FFT)', alpha=0.4, linewidth=1.8)
    plt.plot(xf, yf_pred, color=arch_cfg['color'], linestyle=arch_cfg['ls'], label='Funkcja odszumiona (FFT)', alpha=0.9, linewidth=1.5)
    
    #plt.title(f'[{arch_name}] Analiza widmowa FFT\nFunkcja: {func_name.upper()}')
    plt.xlabel('Częstotliwość')
    plt.ylabel('Amplituda widma')
    plt.legend(edgecolor='black')
    
    if save_path: 
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_pointwise_error(x_true, y_true, y_pred, func_name, arch_name, save_path, config, metrics):
    error = np.abs(y_true - y_pred)
    base_arch = _get_base_arch(arch_name)
    arch_cfg = ARCH_CONFIG.get(base_arch, {'color': 'black', 'ls': '-', 'marker': ''})
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16 / 2.54, 14 / 2.54), sharex=True)
    
    ax1.plot(x_true, y_true, 'k-', label='Funkcja oryginalna', linewidth=1.6)
    ax1.plot(x_true, y_pred, color=arch_cfg['color'], linestyle=arch_cfg['ls'], label='Funkcja odszumiona', linewidth=1.4)
    #ax1.set_title(f'[{arch_name}] Lokalizacja błędów rekonstrukcji\nFunkcja: {func_name.upper()}')
    ax1.legend(loc='upper left', edgecolor='black')
    
    ax2.fill_between(x_true, error, 0, color='#666666', alpha=0.2)
    ax2.plot(x_true, error, color='#333333', linewidth=1.2)
    ax2.set_ylabel('Błąd bezwzględny')
    ax2.set_xlabel('Oś X')
    
    if save_path: 
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_metric_bar_comparison(df, func_name, metric='SNR', ascending=False, save_path=None):
    df_func = df[df['Funkcja'] == func_name].copy()
    if df_func.empty: return
        
    df_func = df_func.sort_values(by=metric, ascending=ascending)
    

    for arch in df_func['Architektura']:
        if 'UNet' in arch: colors.append('#3b528b')      
        elif 'Conv1D' in arch: colors.append('#5ec962')  
        else: colors.append('#fde725')                   

    ax = sns.barplot(data=df_func, x=metric, y='Architektura', palette=PALETTE, edgecolor='black', linewidth=0.8)
    
    #plt.title(f'Ranking architektur wg metryki {metric}\nFunkcja: {func_name.upper()}')
    plt.xlabel(f'Wartość {metric}' + (' (dB)' if metric == 'SNR' else ''))
    plt.ylabel('Konfiguracja modelowa')
    
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + (0.02 * width), p.get_y() + p.get_height()/2. + 0.1, 
                f'{width:.3f}' if width < 1 else f'{width:.1f}', 
                ha="left", va="center", fontsize=9, fontweight='semibold')
    plt.figure()

    legend_elements = [
        Patch(facecolor='#3b528b', edgecolor='black', label='UNet'),
        Patch(facecolor='#5ec962', edgecolor='black', label='Conv1D'),
        Patch(facecolor='#fde725', edgecolor='black', label='MLP')
    ]
    plt.legend(handles=legend_elements, loc='lower right', title="Architektura", edgecolor='black')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_denoising_trajectory(ddpm_model, x_true, y_true, t_start_sdedit, device, save_path, config, metrics, arch_name, func_name):
    ddpm_model.model.eval() 
    base_arch = _get_base_arch(arch_name)

    with torch.no_grad():
        x_tensor = torch.tensor(y_true, dtype=torch.float32, device=device).unsqueeze(0)
        t_tensor = torch.full((1,), t_start_sdedit - 1, dtype=torch.long, device=device)
        
        y_t = ddpm_model.q_sample(x_start=x_tensor, t=t_tensor)
        trajectory = [y_t.cpu().numpy().flatten()]
        steps_to_save = np.linspace(t_start_sdedit - 1, 0, num=5, dtype=int)
        
        current_y = y_t.clone()
        for i in reversed(range(t_start_sdedit)):
            t_batch = torch.full((1,), i, device=device, dtype=torch.long)
            
            model_input = current_y.reshape(1, -1)  
            noise_pred = ddpm_model.model(model_input, t_batch)  
            
            alpha_t = ddpm_model.alphas[t_batch].view(-1, 1)
            alpha_bar_t = ddpm_model.alphas_bar[t_batch].view(-1, 1)
            beta_t = ddpm_model.betas[t_batch].view(-1, 1)
            
            noise = torch.randn_like(model_input) if i > 0 else torch.zeros_like(model_input)
            current_y = (1 / torch.sqrt(alpha_t)) * (model_input - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * noise_pred)
            current_y = current_y + torch.sqrt(beta_t) * noise
            
            if i in steps_to_save or i == 0:
                trajectory.append(current_y.cpu().numpy().flatten())   


    with plt.rc_context({'figure.autolayout': False}):
        fig, ax = plt.subplots(figsize=(21 / 2.54, 11 / 2.54)) 
        
        ax.plot(x_true, y_true, color='black', linestyle='--', label='Funkcja oryginalna', zorder=10)
        
        cmap_map = {'MLP': 'viridis', 'Conv1D': 'viridis', 'UNet': 'viridis'}
        cmap_name = cmap_map.get(base_arch, 'viridis')
        colors = plt.get_cmap(cmap_name)(np.linspace(0.3, 1, len(trajectory)))
        
        for i, (traj_y, color) in enumerate(zip(trajectory, colors)):
            if i == 0:
                label, alpha = f'Punkt początkowy (t={t_start_sdedit})', 0.5
            elif i == len(trajectory) - 1:
                label, alpha = 'Funkcja odszumiona końcowa', 1.0
            else:
                step_num = steps_to_save[min(i-1, len(steps_to_save)-1)]
                label, alpha = f'Krok odszumiania {step_num}', 0.7
                
            ax.plot(x_true, traj_y, color=color, alpha=alpha, label=label)

        ax.set_xlabel('Oś X')
        ax.set_ylabel('Amplituda')
        
        ax.legend(
            loc='upper center', 
            bbox_to_anchor=(0.5, -0.22), 
            ncol=3, 
            framealpha=0.9,
            edgecolor='#cccccc'
        )
        
        info_text = (
            f"PARAMETRY PROCESU\n"
            f"────────────────\n"
            f"Architektura: {arch_name}\n"
            f"Kroki (T):    {config.get('T', '?')}\n"
            f"Plan szumu:   {config.get('schedule', '?')}\n"
            f"────────────────\n"
            f"Błąd MSE: {metrics.get('reconstruction_mse', 0):.6f}\n"
            f"Błąd L2: {metrics.get('L2_Error', 0):.6f}\n"
            f"Odchyl.:  {metrics.get('std', 0):.6f}"
        )
            
        props = dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', edgecolor='#ced4da', alpha=0.95)
        ax.text(1.05, 0.5, info_text, transform=ax.transAxes, verticalalignment='center', bbox=props, fontsize=10)

        if save_path: 
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
		
def ensure_3d_tensor(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if not torch.is_tensor(x):
        x = torch.tensor(x).float()

    x = x.to(device)

    while x.dim() > 3 and x.shape[0] == 1:
        x = x.squeeze(0)
    while x.dim() > 3 and x.shape[1] == 1:
        x = x.squeeze(1)

    if x.dim() == 1:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 2:
        x = x.unsqueeze(1)
    elif x.dim() == 3:
        pass
    elif x.dim() == 4:
        if x.shape[1] == 1:
            x = x.squeeze(1)
        elif x.shape[2] == 1:
            x = x.squeeze(2)
    else:
        raise ValueError(f"Unsupported tensor shape: {x.shape}")

    assert x.dim() == 3, f"Tensor is not 3D: {x.shape}"
    assert x.shape[1] == 1, f"Channel dim must be 1: {x.shape}"

    return x


def load_best_model(config_name, func, best_cfg, runner, architectures_config):

    n_T = best_cfg['T']
    schedule = best_cfg['schedule']

    ckpt_path = None

    for run_id in range(runner.num_runs):

        filename = (
            f"{config_name}_{func}_T{n_T}_{schedule}"
            f"_run{run_id}_best_model.pth"
        )

        possible_path = os.path.join(
            runner.checkpoints_dir,
            filename
        )

        if os.path.exists(possible_path):
            ckpt_path = possible_path
            break

    if ckpt_path is None:
        return None

    model_class = architectures_config[config_name]['class']
    capacity = architectures_config[config_name]['capacity']

    try:
        model = model_class(
            data_dim=128,
            base_channels=capacity
        ).to(runner.device)

    except TypeError:
        model = model_class(
            data_dim=128,
            hidden_dim=capacity
        ).to(runner.device)

    checkpoint = torch.load(
        ckpt_path,
        map_location=runner.device,
        weights_only=False
    )

    raw_state = checkpoint['model_state_dict']

    clean_state = {}

    for k, v in raw_state.items():

        if k.startswith('_orig_mod.'):
            clean_state[k.replace('_orig_mod.', '')] = v
        else:
            clean_state[k] = v

    model.load_state_dict(clean_state)

    model.eval()

    betas = get_beta_schedule(
        best_cfg['schedule'],
        1e-4,
        0.02,
        best_cfg['T']
    )

    ddpm = DDPM1D(
        model,
        betas,
        best_cfg['T'],
        runner.device
    )

    return ddpm


def reconstruct_signal(ddpm, y_true_tensor, t_start):

    device = y_true_tensor.device

    t_tensor = torch.full(
        (1,),
        t_start - 1,
        dtype=torch.long,
        device=device
    )



    y_noisy_tensor = ddpm.q_sample(x_start=y_true_tensor, t=t_tensor)
    if y_noisy_tensor.dim() == 2:
        y_noisy_tensor = y_noisy_tensor.unsqueeze(1)
    
    assert y_noisy_tensor.dim() == 3
    current_y = y_noisy_tensor.clone()
    with torch.no_grad():
        for i in reversed(range(t_start)):
            t_batch = torch.full((1,), i, dtype=torch.long, device=device)
    
            assert current_y.dim() == 3, f"BAD SHAPE BEFORE MODEL: {current_y.shape}"
    
            noise_pred = ddpm.model(current_y, t_batch)
            

            noise_pred = noise_pred.squeeze()
            if noise_pred.dim() == 1:
                noise_pred = noise_pred.unsqueeze(0).unsqueeze(0)
            elif noise_pred.dim() == 2:
                noise_pred = noise_pred.unsqueeze(1)
    
            assert noise_pred.shape == current_y.shape, (
                f"SHAPE MISMATCH AFTER FIX: "
                f"{noise_pred.shape} vs {current_y.shape}"
            )

            alpha_t = ddpm.alphas[i].item()
            alpha_bar_t = ddpm.alphas_bar[i].item()
            beta_t = ddpm.betas[i].item()

            if i > 0:
                noise = torch.randn_like(current_y)
            else:
                noise = torch.zeros_like(current_y)

            current_y = (
                (1 / np.sqrt(alpha_t))
                * (
                    current_y
                    - (
                        ((1 - alpha_t) / np.sqrt(1 - alpha_bar_t))
                        * noise_pred
                    )
                )
            )

            current_y = current_y + np.sqrt(beta_t) * noise

    return y_noisy_tensor, current_y

def analyze_and_plot_best_architectures(test_functions, architectures_config, cache_dir='experiments/cache', save_dir='../images/experiment2/analysis'):

    os.makedirs(save_dir, exist_ok=True)
    
    def get_arch_type(name):
        if 'UNet' in name: return 'UNet'
        if 'Conv1D' in name: return 'Conv1D'
        if 'MLP' in name: return 'MLP'
        return 'Inna'

    for func in test_functions:        
        best_per_arch = {}
        
        for config_name in architectures_config.keys():
            cache_file = os.path.join(cache_dir, f"results_cache_{config_name}_{func}.pkl")
            if not os.path.exists(cache_file):
                continue
                
            with open(cache_file, 'rb') as f:
                saved_results = pickle.load(f)
                
            arch_type = get_arch_type(config_name)
            best_trial_mse = float('inf')
            best_trial_params = None
            
            for trial in saved_results['trials']:
                mses = [run.get('all_metrics', {}).get('MSE', run.get('best_reconstruction_mse', float('inf'))) for run in trial['runs']]
                mean_mse = np.mean(mses)
                
                if mean_mse < best_trial_mse:
                    best_trial_mse = mean_mse
                    best_trial_params = trial['params']
            
            if best_trial_params is not None:
                if arch_type not in best_per_arch or best_trial_mse < best_per_arch[arch_type]['MSE']:
                    best_per_arch[arch_type] = {
                        'Nazwa konfiguracji': config_name,
                        'MSE': best_trial_mse,
                        'Optymalne T': best_trial_params['T'],
                        'Harmonogram szumu': best_trial_params['schedule']
                    }
                    
        if not best_per_arch:
            continue
            
        df_summary = pd.DataFrame.from_dict(best_per_arch, orient='index').reset_index()
        df_summary.rename(columns={'index': 'Klasa architektury'}, inplace=True)
        
        df_summary['Klasa architektury'] = pd.Categorical(
            df_summary['Klasa architektury'], 
            categories=['MLP', 'Conv1D', 'UNet'], 
            ordered=True
        )
        df_summary = df_summary.sort_values('Klasa architektury').reset_index(drop=True)

        # Pobieranie stylów z Twoich obiektów globalnych custom_rc
        fig, ax = plt.subplots(figsize=(12 / 2.54, 7.5 / 2.54))
        
        sns.barplot(data=df_summary, x='Klasa architektury', y='MSE', palette=PALETTE, edgecolor='black', ax=ax, width=0.4)
        
        max_mse = df_summary['MSE'].max()
        min_mse = df_summary['MSE'].min()
        use_log = (max_mse / min_mse > 10) or (min_mse < 0.001)
        
        if use_log:
            ax.set_yscale('log')
            ax.set_ylabel('Błąd średniokwadratowy MSE (log)', fontsize=10)
            ax.set_ylim(bottom=min_mse * 0.3, top=max_mse * 8.0)
        else:
            ax.set_ylabel('Błąd średniokwadratowy MSE', fontsize=10)
            ax.set_ylim(0, max_mse * 1.35)
            
        for p in ax.patches:
            val = p.get_height()
            if pd.notna(val) and val > 0:
                label_text = f'{val:.2e}' if val < 0.01 else f'{val:.4f}'
                
                if use_log:
                    y_pos = val * 1.25 
                    ax.text(p.get_x() + p.get_width() / 2., y_pos, label_text,
                            ha='center', va='bottom', fontsize=8, fontweight='semibold')
                else:
                    ax.annotate(label_text, 
                                (p.get_x() + p.get_width() / 2., val), 
                                ha='center', va='bottom', 
                                xytext=(0, 3), 
                                textcoords='offset points', 
                                fontsize=8.5, fontweight='semibold')
            
        ax.tick_params(axis='both', labelsize=9)
        ax.grid(True, linestyle='--', alpha=0.4, axis='y', which="both")
        
        plt.subplots_adjust(left=0.16, right=0.94, bottom=0.18, top=0.84)
        save_path = os.path.join(save_dir, f"best_arch_comparison_{func.lower()}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()

def generate_comprehensive_global_report(test_functions, architectures_config, runner, cache_dir='experiments/cache', save_dir='../images/experiment2/global_stats'):

    os.makedirs(save_dir, exist_ok=True)
    all_records = []
    for func in test_functions:
        for config_name in architectures_config.keys():
            cache_file = os.path.join(cache_dir, f"results_cache_{config_name}_{func}.pkl")
            if not os.path.exists(cache_file):
                continue
                
            with open(cache_file, 'rb') as f:
                saved_results = pickle.load(f)
                
            for trial in saved_results['trials']:
                params = trial['params']
                
                mses = [run.get('all_metrics', {}).get('MSE', run.get('best_reconstruction_mse', float('inf'))) for run in trial['runs']]
                l2s = [run.get('all_metrics', {}).get('L2_Error', run.get('all_metrics', {}).get('L2', float('inf'))) for run in trial['runs']]
                wassersteins = [run.get('all_metrics', {}).get('Wasserstein_Distance', run.get('all_metrics', {}).get('Wasserstein', float('inf'))) for run in trial['runs']]
                pearsons = [run.get('all_metrics', {}).get('Pearson_Correlation', run.get('all_metrics', {}).get('Pearson', 0.0)) for run in trial['runs']]
                times = [run.get('all_metrics', {}).get('Sample_Time_s', 0.0) for run in trial['runs']]
                
                all_records.append({
                    'Funkcja': func.upper(),
                    'Architektura': config_name,
                    'T': params['T'],
                    'Schedule': params['schedule'],
                    'Ratio': params.get('t_start_ratio', 0.35),
                    'MSE': np.mean(mses),
                    'L2_Error': np.mean(l2s),
                    'Wasserstein': np.mean(wassersteins),
                    'Pearson': np.mean(pearsons),
                    'Czas_Inferencji_s': np.mean(times)
                })
                
    if not all_records:
        print("[BŁĄD] Brak danych w cache. Upewnij się, że eksperymenty zostały zapisane.")
        return None, None
        
    df_global = pd.DataFrame(all_records)
    
    df_ranking = df_global.groupby(['Architektura', 'T', 'Schedule', 'Ratio']).agg(
        Mean_MSE=('MSE', 'mean'),
        Median_MSE=('MSE', 'median'),
        Mean_L2=('L2_Error', 'mean'),
        Mean_Wasserstein=('Wasserstein', 'mean'),
        Mean_Pearson=('Pearson', 'mean'),
        Mean_Inference_Time=('Czas_Inferencji_s', 'mean')
    ).reset_index()
    
    df_ranking = df_ranking.sort_values('Median_MSE').reset_index(drop=True)
    
    print("\nTOP 3 NAJLEPSZE GLOBALNIE KONFIGURACJE:")
    display(df_ranking.head(3).style.format({
        'Mean_MSE': '{:.2e}', 'Median_MSE': '{:.2e}', 'Mean_L2': '{:.4f}', 
        'Mean_Wasserstein': '{:.4f}', 'Mean_Pearson': '{:.4f}', 'Mean_Inference_Time': '{:.4f}s'
    }))
    
    print("\nTOP 3 NAJGORSZE GLOBALNIE KONFIGURACJE:")
    display(df_ranking.tail(3).style.format({
        'Mean_MSE': '{:.2e}', 'Median_MSE': '{:.2e}', 'Mean_L2': '{:.4f}', 
        'Mean_Wasserstein': '{:.4f}', 'Mean_Pearson': '{:.4f}', 'Mean_Inference_Time': '{:.4f}s'
    }))
    
    best_config_row = df_ranking.iloc[0]
    worst_config_row = df_ranking.iloc[-1]
    
    df_ranking.to_csv(os.path.join(save_dir, 'global_sdedit_complexity_quality_summary.csv'), index=False)
    
    return best_config_row, worst_config_row

def generate_comparison_tables(df, save_dir='../images/experiment2/global_stats'):

    os.makedirs(save_dir, exist_ok=True)
    
    if df.empty:
        print("[BŁĄD] Przekazany DataFrame z metrykami jest pusty.")
        return

    print(f"\n{'='*80}\n| GENEROWANIE TABEL PORÓWNAWCZYCH DO PRACY DYPLOMOWEJ |\n{'='*80}")

    agg_dict = {
        'MSE': 'mean',
        'L2_Error': 'mean',
        'Wasserstein': 'mean',
        'Pearson ($\\rho$)': 'mean',
        'SNR (dB)': 'mean',
        'Czas_s': 'mean'
    }
    
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}

    df_summary = df.groupby('Architektura').agg(agg_dict)
    
    order_arch = [a for a in ['MLP', 'Conv1D', 'UNet'] if a in df_summary.index]
    df_summary = df_summary.reindex(order_arch)
    
    rename_dict = {
        'L2_Error': 'Średni błąd $L_2$ (%)',
        'Wasserstein': 'Odległość Wassersteina $W_1$',
        'Pearson ($\\rho$)': 'Korelacja Pearsona $\\rho$',
        'SNR (dB)': 'Stosunek SNR (dB)',
        'Czas_s': 'Czas inferencji (s)'
    }
    df_summary = df_summary.rename(columns=rename_dict)

    print("\n🔹 ZBIORCZE PODSUMOWANIE METRYK PER KLASA ARCHITEKTURY:")
    display(df_summary.style.format('{:.4f}'))

    csv_path = os.path.join(save_dir, 'summary_metrics_table.csv')
    latex_path = os.path.join(save_dir, 'summary_metrics_table.tex')
    
    df_summary.to_csv(csv_path)
    
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(df_summary.to_latex(float_format="%.4f", escape=False, na_rep='---'))
        


def plot_best_vs_worst_comparison(runner, test_functions, architectures_config, best_cfg, worst_cfg, selected_func='square_wave', save_dir='../images/experiment2/analysis'):
    os.makedirs(save_dir, exist_ok=True)

    print(f"NAJLEPSZA: {best_cfg['Architektura']} | T={best_cfg['T']} | Schedule={best_cfg['Schedule']} | Ratio={best_cfg['Ratio']}")
    print(f"NAJGORSZA: {worst_cfg['Architektura']} | T={worst_cfg['T']} | Schedule={worst_cfg['Schedule']} | Ratio={worst_cfg['Ratio']}")
    print("-" * 60)

    x_val, y_true_raw = runner.math_funcs.get_dataset(selected_func, num_samples=1, mode='test')
    x_axis = x_val[0]
    y_true_np = np.squeeze(y_true_raw)
    
    y_true_tensor = ensure_3d_tensor(y_true_np, runner.device)
    
    best_params = {'T': best_cfg['T'], 'schedule': best_cfg['Schedule']}
    worst_params = {'T': worst_cfg['T'], 'schedule': worst_cfg['Schedule']}
    
    ddpm_best = load_best_model(best_cfg['Architektura'], selected_func, best_params, runner, architectures_config)
    ddpm_worst = load_best_model(worst_cfg['Architektura'], selected_func, worst_params, runner, architectures_config)
    
    if ddpm_best is None or ddpm_worst is None:
        print("[BŁĄD] Nie można załadować punktów kontrolnych dla wyznaczonych modeli!")
        return
        
    t_start_best = max(1, int(best_cfg['Ratio'] * best_cfg['T']))
    y_noisy_best_tensor, y_pred_best_tensor = reconstruct_signal_safe(ddpm_best, y_true_tensor, t_start_best)
    
    t_start_worst = max(1, int(worst_cfg['Ratio'] * worst_cfg['T']))
    y_noisy_worst_tensor, y_pred_worst_tensor = reconstruct_signal_safe(ddpm_worst, y_true_tensor, t_start_worst)
    
    y_true_plot = y_true_tensor[0, 0].cpu().numpy()
    y_noisy_best_plot = y_noisy_best_tensor[0, 0].cpu().numpy()
    y_noisy_worst_plot = y_noisy_worst_tensor[0, 0].cpu().numpy()
    y_pred_best_plot = y_pred_best_tensor[0, 0].cpu().numpy()
    y_pred_worst_plot = y_pred_worst_tensor[0, 0].cpu().numpy()
    
    with plt.rc_context({'figure.autolayout': False}):
        fig, axes = plt.subplots(1, 2, figsize=(16 / 2.54, 7.0 / 2.54), sharey=True)
        
        cfg_best = ARCH_CONFIG.get(_get_base_arch(best_cfg['Architektura']), {'color': 'black', 'ls': '-'})
        cfg_worst = ARCH_CONFIG.get(_get_base_arch(worst_cfg['Architektura']), {'color': 'black', 'ls': '-.'})

        b_label = f"Rekonstrukcja"
        w_label = f"Rekonstrukcja"

        axes[0].plot(x_axis, y_true_plot, label='Funkcja oryginalna', color='#d3d3d3', linewidth=2.5, zorder=1)
        axes[0].scatter(x_axis, y_noisy_best_plot, label='Zaszumiona (SDEdit)', color='#bbbbbb', alpha=0.4, s=4, zorder=2)
        axes[0].plot(x_axis, y_pred_best_plot, label=b_label, color=cfg_best['color'], linewidth=1.2, linestyle=cfg_best['ls'], zorder=3)
        axes[0].set_xlabel('x', fontsize=9)
        axes[0].set_ylabel('Amplituda', fontsize=9)
        axes[0].legend(fontsize=6.5, loc='upper right')
        
        axes[1].plot(x_axis, y_true_plot, label='Funkcja oryginalna', color='#d3d3d3', linewidth=2.5, zorder=1)
        axes[1].scatter(x_axis, y_noisy_worst_plot, label='Zaszumiona (SDEdit)', color='#bbbbbb', alpha=0.4, s=4, zorder=2)
        axes[1].plot(x_axis, y_pred_worst_plot, label=w_label, color=cfg_worst['color'], linewidth=1.2, linestyle=cfg_worst['ls'], zorder=3)
        axes[1].set_xlabel('x', fontsize=9)
        axes[1].legend(fontsize=6.5, loc='upper right')
        
        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, linestyle=':', alpha=0.5)
        
        plt.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.95, wspace=0.16)
        
        b_sch = "lin" if "linear" in str(best_cfg['Schedule']).lower() else "cos"
        w_sch = "lin" if "linear" in str(worst_cfg['Schedule']).lower() else "cos"
        
        filename = f"v_comp_{selected_func.lower()}_best_{best_cfg['Architektura']}_{best_cfg['T']}_{b_sch}_vs_worst_{worst_cfg['Architektura']}_{worst_cfg['T']}_{w_sch}.png"
        save_path = os.path.normpath(os.path.join(save_dir, filename))
        
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()		
		
def reconstruct_signal_safe(ddpm, y_true_tensor, t_start):

    device = y_true_tensor.device
    t_tensor = torch.full((1,), t_start - 1, dtype=torch.long, device=device)
    
    y_noisy_tensor = ddpm.q_sample(x_start=y_true_tensor, t=t_tensor)
    if y_noisy_tensor.dim() == 2:
        y_noisy_tensor = y_noisy_tensor.unsqueeze(1)
        
    current_y = y_noisy_tensor.clone()
    
    with torch.no_grad():
        for i in reversed(range(t_start)):
            t_batch = torch.full((1,), i, dtype=torch.long, device=device)
            
            noise_pred = ddpm.model(current_y, t_batch)
            
            noise_pred = noise_pred.squeeze()
            if noise_pred.dim() == 1:
                noise_pred = noise_pred.unsqueeze(0).unsqueeze(0)
            elif noise_pred.dim() == 2:
                noise_pred = noise_pred.unsqueeze(1)
                
            alpha_t = ddpm.alphas[i].item()
            alpha_bar_t = ddpm.alphas_bar[i].item()
            beta_t = ddpm.betas[i].item()
            
            if i > 0:
                noise = torch.randn_like(current_y)
            else:
                noise = torch.zeros_like(current_y)
                
            current_y = (1 / np.sqrt(alpha_t)) * (current_y - (((1 - alpha_t) / np.sqrt(1 - alpha_bar_t)) * noise_pred))
            current_y = current_y + np.sqrt(beta_t) * noise
            
    return y_noisy_tensor, current_y


def generate_global_report(test_functions, architectures_config, runner, cache_dir='experiments/cache', save_dir='../images/experiment2/global_stats'):

    os.makedirs(save_dir, exist_ok=True)
    all_records = []
    for func in test_functions:
        for config_name in architectures_config.keys():
            cache_file = os.path.join(cache_dir, f"results_cache_{config_name}_{func}.pkl")
            if not os.path.exists(cache_file):
                continue
                
            with open(cache_file, 'rb') as f:
                saved_results = pickle.load(f)
                
            for trial in saved_results['trials']:
                params = trial['params']
                
                mses = [run.get('all_metrics', {}).get('MSE', run.get('best_reconstruction_mse', float('inf'))) for run in trial['runs']]
                l2s = [run.get('all_metrics', {}).get('L2_Error', run.get('all_metrics', {}).get('L2', float('inf'))) for run in trial['runs']]
                wassersteins = [run.get('all_metrics', {}).get('Wasserstein_Distance', run.get('all_metrics', {}).get('Wasserstein', float('inf'))) for run in trial['runs']]
                pearsons = [run.get('all_metrics', {}).get('Pearson_Correlation', run.get('all_metrics', {}).get('Pearson', 1.0)) for run in trial['runs']]
                times = [run.get('all_metrics', {}).get('Sample_Time_s', 0.0) for run in trial['runs']]
                
                ratios = [run.get('all_metrics', {}).get('t_start_ratio', 0.35) for run in trial['runs']]
                
                all_records.append({
                    'Funkcja': func.upper(),
                    'Architektura': config_name,
                    'T': params['T'],
                    'Schedule': params['schedule'],
                    'Ratio': np.mean(ratios), 
                    'MSE': np.mean(mses),
                    'L2_Error': np.mean(l2s),
                    'Wasserstein': np.mean(wassersteins),
                    'Pearson': np.mean(pearsons),
                    'Czas_Inferencji_s': np.mean(times)
                })
                
    if not all_records:
        print("[UWAGA] Brak plików w cache. Poczekaj na zapis pierwszych wyników.")
        return None, None
        
    df_global = pd.DataFrame(all_records)
    
    df_ranking = df_global.groupby(['Architektura', 'T', 'Schedule', 'Ratio']).agg(
        Mean_MSE=('MSE', 'mean'),
        Median_MSE=('MSE', 'median'),
        Mean_L2=('L2_Error', 'mean'),
        Mean_Wasserstein=('Wasserstein', 'mean'),
        Mean_Pearson=('Pearson', 'mean'),
        Mean_Time=('Czas_Inferencji_s', 'mean')
    ).reset_index().sort_values('Median_MSE').reset_index(drop=True)
    
    print(f"\n{'='*95}\n| GLOBALNE PODSUMOWANIE JAKOŚCI I ZŁOŻONOŚCI (ZAKOŃCZONO) |\n{'='*95}")
    print("\nTOP 3 NAJLEPSZE KONFIGURACJE W CAŁYM EKSPERYMENCIE:")
    display(df_ranking.head(3).style.format({'Mean_MSE': '{:.2e}', 'Median_MSE': '{:.2e}', 'Mean_L2': '{:.4f}', 'Mean_Wasserstein': '{:.4f}', 'Mean_Pearson': '{:.4f}', 'Mean_Time': '{:.4f}s'}))
    
    print("\nTOP 3 NAJGORSZE KONFIGURACJE W CAŁYM EKSPERYMENCIE:")
    display(df_ranking.tail(3).style.format({'Mean_MSE': '{:.2e}', 'Median_MSE': '{:.2e}', 'Mean_L2': '{:.4f}', 'Mean_Wasserstein': '{:.4f}', 'Mean_Pearson': '{:.4f}', 'Mean_Time': '{:.4f}s'}))
    
    df_ranking.to_csv(os.path.join(save_dir, 'global_report_sdedit_final.csv'), index=False)
    
    return df_ranking.iloc[0], df_ranking.iloc[-1]

def plot_global_correlation_sensitivity(df, save_dir='../images/experiment2/analysis'):

    os.makedirs(save_dir, exist_ok=True)
    df_corr = df.copy()
    
    label_schedule = 'Harmonogram (0=Liniowy, 1=Cosinusowy)'
    df_corr[label_schedule] = df_corr['Harmonogram'].map({'linear': 0, 'cosine': 1})
    
    df_corr = df_corr.rename(columns={
        't_steps (T)': 'Liczba kroków (T)',
        't_start_ratio': 'Punkt startu (tau)',
        'skip_steps (S)': 'Skok solwera (S)',
        'Czas_s': 'Czas wykonania (s)'
    })
    
    inputs = ['Liczba kroków (T)', label_schedule, 'Punkt startu (tau)', 'Skok solwera (S)']
    outputs = ['MSE', 'L2_Error', 'Wasserstein', 'Czas wykonania (s)']
    
    corr_matrix = df_corr[inputs + outputs].corr().loc[inputs, outputs]
    
    plt.figure(figsize=(8.5, 5.5))
    
    sns.heatmap(corr_matrix, 
                annot=True, 
                fmt=".2f", 
                cmap='viridis', 
                vmin=-1, 
                vmax=1,
                linewidths=1.5, 
                linecolor='white', 
                cbar_kws={'label': 'Współczynnik korelacji Pearsona ($r$)'})
    
    plt.ylabel('Hiperparametry wejściowe', fontsize=10, fontweight='bold', labelpad=12)
    plt.xlabel('Metryki oceny rekonstrukcji', fontsize=10, fontweight='bold', labelpad=12)
    
    plt.xticks(rotation=20, ha='right')
    plt.yticks(rotation=0)
    
    save_path = os.path.join(save_dir, 'corr_params_map.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_radar_metrics_comparison(df, func_name, architectures_to_compare, save_path=None):

    df_func = df[df['Funkcja'] == func_name].copy()
    if df_func.empty:
        print(f"[UWAGA] Brak danych w DataFrame dla funkcji: {func_name}")
        return
        
    df_plot = df_func[df_func['Architektura'].isin(architectures_to_compare)].copy()
    if df_plot.empty:
        print(f"[UWAGA] Żadna z podanych architektur {architectures_to_compare} nie znajduje się w DF.")
        return
        
    metrics_to_plot = ['SNR', 'Correlation', 'Wasserstein', 'MSE', 'Total_Time_s']
    mapping_keys = {'SNR': 'SNR (dB)', 'Correlation': 'Pearson ($\\rho$)', 'Wasserstein': 'Wasserstein', 'MSE': 'MSE', 'Total_Time_s': 'Czas_s'}
    
    for k_old, k_new in mapping_keys.items():
        if k_new in df_func.columns:
            df_func[k_old] = df_func[k_new]
            df_plot[k_old] = df_plot[k_new]

    metrics_to_plot = ['SNR', 'Correlation', 'Wasserstein', 'MSE', 'Total_Time_s']
    
    for m in metrics_to_plot:
        min_val = df_func[m].min()
        max_val = df_func[m].max()
        if max_val == min_val: max_val = min_val + 1e-6
            
        if m in ['SNR', 'Correlation']:
            df_plot[f'{m}_norm'] = (df_plot[m] - min_val) / (max_val - min_val)
        else:
            df_plot[f'{m}_norm'] = 1 - ((df_plot[m] - min_val) / (max_val - min_val))
            
    categories = ['SNR\n(Więcej=Lepiej)', 'Korelacja\n(Więcej=Lepiej)', 'Wasserstein\n(Mniej=Lepiej)', 'MSE\n(Mniej=Lepiej)', 'Czas\n(Mniej=Lepiej)']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    with plt.rc_context({'figure.autolayout': False}):
        fig, ax = plt.subplots(figsize=(13 / 2.54, 13 / 2.54), subplot_kw=dict(polar=True))
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        
        ax.set_xticklabels(categories, fontsize=9, fontweight='bold')
        ax.tick_params(axis='x', pad=12) 
        
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["Źle", "Słabo", "Dobrze", "Idealnie"], color="grey", size=8.5)
        ax.set_ylim(0, 1.1)
        
        for idx, row in df_plot.iterrows():
            arch = row['Architektura']
            base_arch = _get_base_arch(arch)
            
            cfg = ARCH_CONFIG.get(base_arch, {'color': 'black', 'ls': '-', 'marker': ''})
            
            values = row[[f'{m}_norm' for m in metrics_to_plot]].tolist()
            values += values[:1]
            
            ax.plot(angles, values, color=cfg['color'], linewidth=1.5, linestyle=cfg['ls'], 
                    marker=cfg['marker'], markersize=4, label=arch)
            ax.fill(angles, values, color=cfg['color'], alpha=0.04) 

        ax.legend(loc='upper left', bbox_to_anchor=(1.2, 1.0), fontsize=8.5, frameon=True)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
		
def generate_quality_time_plots(test_functions, architectures_config, cache_dir='experiments/cache', save_dir='../images/experiment2/global_stats'):

    os.makedirs(save_dir, exist_ok=True)
    all_records = []
    
    for func in test_functions:
        for config_name in architectures_config.keys():
			
            cache_file = os.path.join(cache_dir, f"results_cache_{config_name}_{func}.pkl")
            if not os.path.exists(cache_file):
                continue
                
            with open(cache_file, 'rb') as f:
                saved_results = pickle.load(f)
                
            for trial in saved_results['trials']:
                params = trial['params']
                mses = [run.get('all_metrics', {}).get('MSE', run.get('best_reconstruction_mse', float('inf'))) for run in trial['runs']]
                times = [run.get('all_metrics', {}).get('Sample_Time_s', 0.0) for run in trial['runs']]
                
                arch_type = 'UNet' if 'UNet' in config_name else ('Conv1D' if 'Conv1D' in config_name else 'MLP')
                
                all_records.append({
                    'Architektura': arch_type,
                    'T': params['T'],
                    'MSE': np.mean(mses),
                    'Czas_s': np.mean(times)
                })
                
    if not all_records:
        print("[BŁĄD] Brak danych w cache.")
        return
        
    df = pd.DataFrame(all_records)
    
    PALETTE = ['#fde725', '#5ec962', '#3b528b']

    with plt.rc_context({'figure.autolayout': False}):
        fig, axes = plt.subplots(1, 2, figsize=(16 / 2.54, 7.5 / 2.54))
        
        sns.boxplot(
            data=df, x='Architektura', y='Czas_s', hue='T', 
            palette=PALETTE, ax=axes[0], width=0.6, 
            linewidth=0.8, fliersize=0, boxprops=dict(alpha=0.85)
        )
        sns.stripplot(
            data=df, x='Architektura', y='Czas_s', hue='T',
            palette=PALETTE, ax=axes[0], dodge=True, 
            jitter=0.15, size=2.5, edgecolor='black', linewidth=0.3, alpha=0.5
        )
        axes[0].set_xlabel('Klasa architektury sieciowej', fontsize=9)
        axes[0].set_ylabel('Czas pojedynczej inferencji [s]', fontsize=9)
        axes[0].grid(True, linestyle=':', alpha=0.5, axis='y')
        
        sns.boxplot(
            data=df, x='Architektura', y='MSE', hue='T', 
            palette=PALETTE, ax=axes[1], width=0.6, 
            linewidth=0.8, fliersize=0, boxprops=dict(alpha=0.85)
        )
        sns.stripplot(
            data=df, x='Architektura', y='MSE', hue='T',
            palette=PALETTE, ax=axes[1], dodge=True, 
            jitter=0.15, size=2.5, edgecolor='black', linewidth=0.3, alpha=0.5
        )
        axes[1].set_yscale('log')
        axes[1].set_xlabel('Klasa architektury sieciowej', fontsize=9)
        axes[1].set_ylabel('Błąd średniokwadratowy MSE (skala log)', fontsize=9)
        axes[1].grid(True, linestyle=':', alpha=0.5, axis='y', which='both')
        
        handles, labels = axes[0].get_legend_handles_labels()
        
        if axes[0].get_legend(): axes[0].get_legend().remove()
        if axes[1].get_legend(): axes[1].get_legend().remove()
        
        fig.legend(handles[:3], labels[:3], title='Kroki $T$', 
                   loc='center left', bbox_to_anchor=(0.85, 0.5), 
                   fontsize=7.5, title_fontsize=8, frameon=True, edgecolor='#ced4da')
        
        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=8.5)
            
        plt.subplots_adjust(left=0.08, right=0.83, bottom=0.16, top=0.88, wspace=0.32)
        
        save_path = os.path.join(save_dir, 'global_boxplots_quality_time.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()


def plot_architecture_dominance_matrix(test_functions, architectures_config, cache_dir='experiments/cache', save_dir='../images/experiment2/analysis'):
    os.makedirs(save_dir, exist_ok=True)
    best_per_function = {}

    for func in test_functions:
        best_error = float('inf')
        winner_arch = 'Brak'
        
        for config_name in architectures_config.keys():
            cache_file = os.path.join(cache_dir, f"results_cache_{config_name}_{func}.pkl")
            if not os.path.exists(cache_file):
                continue
                
            with open(cache_file, 'rb') as f:
                saved_results = pickle.load(f)
                
            for trial in saved_results['trials']:
                runs_l2 = [run.get('all_metrics', {}).get('L2_Error', run.get('best_reconstruction_mse', float('inf'))) for run in trial['runs']]
                mean_l2 = np.mean(runs_l2)
                
                if mean_l2 < best_error:
                    best_error = mean_l2
                    winner_arch = 'UNet' if 'UNet' in config_name else ('Conv1D' if 'Conv1D' in config_name else 'MLP')
        
        if winner_arch != 'Brak':
            best_per_function[func] = winner_arch

    df_wins = pd.DataFrame(list(best_per_function.items()), columns=['Funkcja', 'Najlepsza Architektura'])
    counts = df_wins['Najlepsza Architektura'].value_counts().reindex(['MLP', 'Conv1D', 'UNet'], fill_value=0)
    plt.figure(figsize=(7, 4.5))
    colors = ['#525252', '#EDE9E6', '#313131']
    
    bars = plt.bar(counts.index, counts.values, color=colors, edgecolor='black', linewidth=1.2, width=0.5)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.xlabel('Klasa architektury', fontsize=10)
    plt.ylabel('Liczba wygranych klas sygnałów (najniższy błąd)', fontsize=10)
    plt.ylim(0, len(test_functions) + 1)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_path = os.path.join(save_dir, 'winrate.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()





def plot_sdedit_sensitivity_grid(test_functions, architectures_config, cache_dir='experiments/cache', save_dir='../images/experiment2/global_stats'):

    os.makedirs(save_dir, exist_ok=True)
    all_records = []
    
    for func in test_functions:
        for config_name in architectures_config.keys():
            cache_file = os.path.join(cache_dir, f"results_cache_{config_name}_{func}.pkl")
            if not os.path.exists(cache_file):
                continue
                
            with open(cache_file, 'rb') as f:
                saved_results = pickle.load(f)
                
            capacity_match = re.search(r'(\d+)', config_name)
            capacity = int(capacity_match.group(1)) if capacity_match else 'Standard'
            
            for trial in saved_results['trials']:
                for run in trial['runs']:
                    metrics_dict = run.get('all_metrics', {})
                    if not metrics_dict:
                        continue
                        
                    t_start = metrics_dict.get('t_start_ratio', np.nan)
                    l2_err = metrics_dict.get('L2_Error', np.nan)
                    
                    arch_type = 'UNet' if 'UNet' in config_name else ('Conv1D' if 'Conv1D' in config_name else 'MLP')
                    
                    if not np.isnan(t_start) and not np.isnan(l2_err):
                        all_records.append({
                            'Architektura': arch_type,
                            'Pojemność (C)': capacity,
                            't_start_ratio': t_start,
                            'L2_Error': l2_err
                        })
                        
    df = pd.DataFrame(all_records)
    if df.empty:
        print("[BŁĄD] Brak danych diagnostycznych w cache. Upewnij się, że pliki pkl zawierają 't_start_ratio'.")
        return

    df = df.sort_values(by=['Architektura', 'Pojemność (C)', 't_start_ratio'])
        
    g = sns.FacetGrid(df, col='Architektura', hue='Pojemność (C)', 
                      col_order=['MLP', 'Conv1D', 'UNet'],
                      palette='viridis', height=4.5, aspect=0.9, sharey=False)
    
    g.map(sns.lineplot, 't_start_ratio', 'L2_Error', marker='o', linewidth=2, markersize=6, errorbar=None)
    
    g.set_titles(template="Architektura: {col_name}", size=11, weight='bold')
    g.set_axis_labels("Głębokość zaszumienia ($t_{\\mathrm{start}} / T$)", "Błąd relatywny $L_2$ (%)")
    
    for ax in g.axes.flat:
        ax.set_xticks(sorted(df['t_start_ratio'].unique()))
        ax.tick_params(labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if "MLP" in ax.get_title():
            ax.set_yscale('log')
            ax.set_ylabel("Błąd $L_2$ (%) [Skala LOG]")

    g.add_legend(title="Pojemność (kanały $C$)", fontsize=9, title_fontsize=9.5)
    
    plt.subplots_adjust(wspace=0.25, top=0.85)
    
    save_path = os.path.join(save_dir, 'sensitivity_tstart_capacity.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_sdedit_solver_tradeoff(test_functions, architectures_config, cache_dir='experiments/cache', save_dir='../images/experiment2/analysis'):
    os.makedirs(save_dir, exist_ok=True)
    all_records = []
    
    for func in test_functions:
        for config_name in architectures_config.keys():
            cache_file = os.path.join(cache_dir, f"results_cache_{config_name}_{func}.pkl")
            if not os.path.exists(cache_file):
                continue
                
            with open(cache_file, 'rb') as f:
                saved_results = pickle.load(f)
                
            for trial in saved_results['trials']:
                for run in trial['runs']:
                    metrics_dict = run.get('all_metrics', {})
                    if not metrics_dict:
                        continue
                        
                    skip = metrics_dict.get('skip_steps', np.nan)
                    l2_err = metrics_dict.get('L2_Error', np.nan)
                    time_s = metrics_dict.get('Sample_Time_s', np.nan)
                    
                    arch_type = 'UNet' if 'UNet' in config_name else ('Conv1D' if 'Conv1D' in config_name else 'MLP')
                    
                    if not np.isnan(skip) and not np.isnan(l2_err):
                        all_records.append({
                            'Architektura': arch_type,
                            'skip_steps (Skok S)': int(skip),
                            'L2_Error': l2_err,
                            'Czas_s': time_s
                        })
                        
    df = pd.DataFrame(all_records)
    if df.empty:
        print("[BŁĄD] Brak danych o 'skip_steps' w bazie cache.")
        return

    df = df.sort_values(by=['Architektura', 'skip_steps (Skok S)'])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    sns.lineplot(x='skip_steps (Skok S)', y='L2_Error', hue='Architektura', data=df,
                 marker='o', linewidth=2.5, markersize=7, palette=PALETTE, ax=axes[0], errorbar=None)
    #axes[0].set_title('a) Wpływ skoku solwera na dokładność', fontsize=10, fontweight='bold', loc='left')
    axes[0].set_xlabel('Wartość parametru skip_steps (Krok solwera $S$)', fontsize=9)
    axes[0].set_ylabel('Średni błąd relatywny $L_2$ (%)', fontsize=9)
    
    sns.lineplot(x='skip_steps (Skok S)', y='Czas_s', hue='Architektura', data=df,
                 marker='s', linewidth=2.5, markersize=7, palette=PALETTE, ax=axes[1], errorbar=None)
    #axes[1].set_title('b) Zysk obliczeniowy (Przyspieszenie)', fontsize=10, fontweight='bold', loc='left')
    axes[1].set_xlabel('Wartość parametru skip_steps (Krok solwera $S$)', fontsize=9)
    axes[1].set_ylabel('Czas generowania pojedynczej próbki (s)', fontsize=9)
    
    unique_skips = sorted(df['skip_steps (Skok S)'].unique())
    for ax in axes:
        ax.set_xticks(unique_skips)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8.5)
        
    plt.subplots_adjust(wspace=0.25)
    save_path = os.path.join(save_dir, 'quality_time_skip_steps.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def aggregate_experiment_metrics(test_functions, architectures_config, cache_dir='experiments/cache'):

    all_records = []
    
    for func in test_functions:
        for config_name in architectures_config.keys():
            cache_file = os.path.join(cache_dir, f"results_cache_{config_name}_{func}.pkl")
            if not os.path.exists(cache_file):
                continue
                
            with open(cache_file, 'rb') as f:
                saved_results = pickle.load(f)
                
            for trial in saved_results['trials']:
                params = trial['params']
                
                t_steps = params.get('T', params.get('t_steps', np.nan))
                schedule = params.get('schedule', params.get('beta_schedule', None))
                if schedule is None:
                    schedule = 'cosine' if 'cosine' in config_name.lower() else 'linear'
                
                for run in trial['runs']:
                    metrics_dict = run.get('all_metrics', {})
                    if not metrics_dict:
                        continue
                        
                    t_start = metrics_dict.get('t_start_ratio', np.nan)
                    skip = metrics_dict.get('skip_steps', np.nan)
                    
                    mse = metrics_dict.get('MSE', run.get('best_reconstruction_mse', np.nan))
                    l2_err = metrics_dict.get('L2_Error', np.nan)
                    wasser = metrics_dict.get('Wasserstein', np.nan)
                    snr = metrics_dict.get('SNR', np.nan)
                    corr = metrics_dict.get('Correlation', np.nan)
                    time_s = metrics_dict.get('Sample_Time_s', np.nan)
                    
                    arch_type = 'UNet' if 'UNet' in config_name else ('Conv1D' if 'Conv1D' in config_name else 'MLP')
                    
                    all_records.append({
                        'Funkcja': func, 
                        'Architektura': arch_type,
                        't_steps (T)': t_steps,
                        'Harmonogram': schedule,
                        't_start_ratio': t_start,
                        'skip_steps (S)': skip,
                        'MSE': mse,
                        'L2_Error': l2_err,
                        'L2_Error (%)': l2_err, 
                        'Wasserstein': wasser,
                        'Wasserstein ($W_1$)': wasser,
                        'SNR (dB)': snr,
                        'Pearson ($\\rho$)': corr,
                        'Czas_s': time_s,
                        'Czas (s)': time_s
                    })
                    
    if not all_records:
        print("[BŁĄD] Brak danych w pamięci cache do agregacji.")
        return pd.DataFrame()
        
    df_global = pd.DataFrame(all_records).dropna(subset=['L2_Error', 'MSE'])

    return df_global


def config_ranking(df):
    config_cols = ['Architektura', 't_steps (T)', 'Harmonogram', 't_start_ratio', 'skip_steps (S)']
    
    df_grouped = df.groupby(config_cols)['L2_Error'].agg([
        ('Średni błąd (%)', 'mean'),
        ('Mediana (%)', 'median'),
        ('Q1 (25%)', lambda x: x.quantile(0.25)),
        ('Q3 (75%)', lambda x: x.quantile(0.75))
    ]).reset_index()
    
    df_sorted = df_grouped.sort_values('Mediana (%)')
    
    print("="*35 + " TOP 5 NAJLEPSZYCH GLOBALNYCH USTAWIEŃ " + "="*35)
    print(df_sorted.head(5).round(4).to_string(index=False))

    print("\n" + "="*35 + " TOP 5 NAJGORSZYCH GLOBALNYCH USTAWIEŃ " + "="*35)
    print(df_sorted.tail(5).iloc[::-1].round(4).to_string(index=False))

def param_importance(df):
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor
    import pandas as pd

    X = df[['t_steps (T)', 't_start_ratio', 'skip_steps (S)']].copy()
    X['Harmonogram (lin/cos)'] = df['Harmonogram'].map({'linear': 0, 'cosine': 1})
    y = df['L2_Error']
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    waznosci = pd.Series(model.feature_importances_ * 100, index=X.columns).sort_values(ascending=True)
    
    plt.figure(figsize=(7, 3.5))
    bars = plt.barh(waznosci.index, waznosci.values, color='#440154', edgecolor='black', height=0.4)
    
    procentowy_margines = waznosci.max() * 0.02
    for bar in bars:
        width = bar.get_width()
        plt.text(width + procentowy_margines, 
                 bar.get_y() + bar.get_height()/2., 
                 f'{width:.1f}', 
                 va='center', ha='left', fontsize=9, fontweight='bold', color='black')
        
    plt.xlabel('Siła wpływu parametru na ostateczny wynik (%)', fontsize=9.5)
    
    plt.xlim(0, waznosci.max() * 1.15)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(axis='x', linestyle=':', alpha=0.5)
    plt.show()
	
def single_params_impact(df):

    params_to_analyze = {
        't_start_ratio': 'PARAMETRU t_start_ratio',
        'skip_steps (S)': 'SKOKU SOLWERA (skip_steps)',
        'Harmonogram': 'HARMONOGRAMU SZUMU',
        't_steps (T)': 'LICZBY KROKÓW t_steps (T)'
    }
    
    for column, display_name in params_to_analyze.items():
        print(f"\n=== STATYSTYKI BŁĘDU DLA {display_name} ===")
        

        tab = df.groupby(column)['L2_Error'].agg([
            'mean', 
            'median', 
            lambda x: x.quantile(0.25), 
            lambda x: x.quantile(0.75)
        ]).reset_index()
        
        tab.columns = [f'Wartość {column}', 'Średni błąd (%)', 'Mediana (%)', 'Kwartyl 1 (25%)', 'Kwartyl 3 (75%)']
        print(tab.round(4).to_string(index=False))
def aggregate_full_sensitivity_data(test_functions, architectures_config, cache_dir='experiments/cache'):

    all_records = []
    
    for func in test_functions:
        for config_name in architectures_config.keys():
            cache_file = os.path.join(cache_dir, f"results_cache_{config_name}_{func}.pkl")
            if not os.path.exists(cache_file):
                continue
                
            with open(cache_file, 'rb') as f:
                saved_results = pickle.load(f)
                
            for trial in saved_results['trials']:
                params = trial['params']
                
                t_steps = params.get('T', params.get('t_steps', np.nan))
                schedule = params.get('schedule', params.get('beta_schedule', None))
                if schedule is None:
                    schedule = 'cosine' if 'cosine' in config_name.lower() else 'linear'
                
                for run in trial['runs']:
                    metrics_dict = run.get('all_metrics', {})
                    if not metrics_dict:
                        continue
                        
                    t_start = metrics_dict.get('t_start_ratio', np.nan)
                    skip = metrics_dict.get('skip_steps', np.nan)
                    
                    mse = metrics_dict.get('MSE', np.nan)
                    l2_err = metrics_dict.get('L2_Error', np.nan)
                    wasser = metrics_dict.get('Wasserstein', np.nan)
                    time_s = metrics_dict.get('Sample_Time_s', np.nan)
                    
                    arch_type = 'UNet' if 'UNet' in config_name else ('Conv1D' if 'Conv1D' in config_name else 'MLP')
                    
                    all_records.append({
                        'Funkcja': func, 
                        'Architektura': arch_type,
                        't_steps (T)': t_steps,
                        'Harmonogram': schedule,
                        't_start_ratio': t_start,
                        'skip_steps (S)': skip,
                        'MSE': mse,
                        'L2_Error': l2_err,
                        'Wasserstein': wasser,
                        'Czas_s': time_s
                    })
                    
    df_global = pd.DataFrame(all_records).dropna()
    return df_global


def heatmap_params_impact(df, save_dir='../images/experiment2/global_stats'):

    
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    architectures = ['MLP', 'Conv1D', 'UNet']
    
    for idx, arch in enumerate(architectures):
        df_arch = df[df['Architektura'] == arch]

        pivot_table = df_arch.pivot_table(
            index='t_start_ratio', 
            columns='skip_steps (S)', 
            values='L2_Error', 
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_table, 
                    annot=True, 
                    fmt=".2f", 
                    cmap='viridis', 
                    linewidths=1, 
                    linecolor='white',
                    ax=axes[idx], 
                    cbar=True if idx == 2 else False, 
                    cbar_kws={'label': 'Średni błąd relatywny $L_2$ (%)'} if idx == 2 else None)
        
        axes[idx].set_title(f'Architektura: {arch}', fontsize=12, fontweight='bold', pad=10)
        axes[idx].set_xlabel('Parametr skip_steps (Skok solwera $S$)', fontsize=10)
        
        if idx == 0:
            axes[idx].set_ylabel('Głębokość zaszumienia (t_start_ratio $\\tau$)', fontsize=10)
        else:
            axes[idx].set_ylabel('') 
            
    plt.suptitle('Mapa wrażliwości: Wpływ krzyżowy parametrów na błąd rekonstrukcji $L_2$', 
                 fontsize=14, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'params_heatmap_global.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_deep_hyperparam_interactions(df, save_dir='../images/experiment2/global_stats'):
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()
        
    sns.lineplot(x='t_start_ratio', y='L2_Error', hue='t_steps (T)', style='Harmonogram', data=df,
                 marker='o', linewidth=2, palette=PALETTE, ax=axes[0], errorbar=None)
    #axes[0].set_title('a) Błąd relatywny $L_2$ w funkcji głębokości zaszumienia', fontsize=10, fontweight='bold', loc='left')
    axes[0].set_xlabel('Parametr t_start_ratio ($\\tau$)', fontsize=8.5)
    axes[0].set_ylabel('Średni błąd $L_2$ (%)', fontsize=8.5)
    
    sns.lineplot(x='skip_steps (S)', y='Czas_s', hue='t_steps (T)', style='Harmonogram', data=df,
                 marker='s', linewidth=2, palette=PALETTE, ax=axes[1], errorbar=None)
    #axes[1].set_title('b) Koszt obliczeniowy w zależności od skoku solwera', fontsize=10, fontweight='bold', loc='left')
    axes[1].set_xlabel('Parametr skip_steps (Skok solwera $S$)', fontsize=8.5)
    axes[1].set_ylabel('Czas wykonywania próby (s)', fontsize=8.5)
    
    sns.lineplot(x='t_start_ratio', y='Wasserstein', hue='t_steps (T)', style='Harmonogram', data=df,
                 marker='d', linewidth=2, palette=PALETTE, ax=axes[2], errorbar=None)
    #axes[2].set_title('c) Odległość Wassersteina $W_1$ (Wierność rozkładu)', fontsize=10, fontweight='bold', loc='left')
    axes[2].set_xlabel('Parametr t_start_ratio ($\\tau$)', fontsize=8.5)
    axes[2].set_ylabel('Wskaźnik błędu $W_1$', fontsize=8.5)
    
    sns.lineplot(x='skip_steps (S)', y='MSE', hue='t_steps (T)', style='Harmonogram', data=df,
                 marker='v', linewidth=2, palette=PALETTE, ax=axes[3], errorbar=None)
    #axes[3].set_title('d) Punktowy błąd MSE w funkcji dyskretyzacji solwera', fontsize=10, fontweight='bold', loc='left')
    axes[3].set_xlabel('Parametr skip_steps (Skok solwera $S$)', fontsize=8.5)
    axes[3].set_ylabel('Błąd średniokwadratowy MSE', fontsize=8.5)
    axes[3].set_yscale('log') 
    
    unique_skips = sorted(df['skip_steps (S)'].unique())
    unique_ratios = sorted(df['t_start_ratio'].unique())
    
    for idx, ax in enumerate(axes):
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8.5)
        
        if idx in [0, 2]:
            ax.set_xticks(unique_ratios)
        else:
            ax.set_xticks(unique_skips)
            
    plt.subplots_adjust(wspace=0.28, hspace=0.32)
    save_path = os.path.join(save_dir, 'hiperparam_interactions_global.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
	
def generate_metrics_report(test_functions, architectures_config, cache_dir='experiments/cache', save_dir='../images/experiment2/global_stats'):

    os.makedirs(save_dir, exist_ok=True)
    all_records = []
    
    for func in test_functions:
        for config_name in architectures_config.keys():
            cache_file = os.path.join(cache_dir, f"results_cache_{config_name}_{func}.pkl")
            if not os.path.exists(cache_file):
                continue
                
            with open(cache_file, 'rb') as f:
                saved_results = pickle.load(f)
                
            for trial in saved_results['trials']:
                params = trial['params']
                
                mses = [run.get('all_metrics', {}).get('MSE', run.get('best_reconstruction_mse', np.nan)) for run in trial['runs']]
                l2s = [run.get('all_metrics', {}).get('L2_Error', np.nan) for run in trial['runs']]
                wassers = [run.get('all_metrics', {}).get('Wasserstein', np.nan) for run in trial['runs']]
                pearsons = [run.get('all_metrics', {}).get('Correlation', np.nan) for run in trial['runs']] 
                snrs = [run.get('all_metrics', {}).get('SNR', np.nan) for run in trial['runs']]
                times = [run.get('all_metrics', {}).get('Sample_Time_s', np.nan) for run in trial['runs']]
                
                arch_type = 'UNet' if 'UNet' in config_name else ('Conv1D' if 'Conv1D' in config_name else 'MLP')
                
                all_records.append({
                    'Funkcja': func,
                    'Architektura': arch_type,
                    'T': params['T'],
                    'MSE': np.nanmean(mses),
                    'L2_Error (%)': np.nanmean(l2s),
                    'Wasserstein ($W_1$)': np.nanmean(wassers),
                    'Pearson ($\\rho$)': np.nanmean(pearsons),
                    'SNR (dB)': np.nanmean(snrs),
                    'Czas (s)': np.nanmean(times)
                })
                
    df = pd.DataFrame(all_records)
    if df.empty:
        print("[BŁĄD] Brak danych w cache do wygenerowania raportu.")
        return
	

    df_summary = df.groupby('Architektura').agg({
        'MSE': 'mean',
        'L2_Error (%)': 'mean',
        'Wasserstein ($W_1$)': 'mean',
        'Pearson ($\\rho$)': 'mean',
        'SNR (dB)': 'mean',
        'Czas (s)': 'mean'
    }).reindex(['MLP', 'Conv1D', 'UNet'])
    
    display(df_summary)
    PALETTE = ['#fde725', '#5ec962', '#3b528b'] 
    ARCH_ORDER = ['MLP', 'Conv1D', 'UNet']

    metrics_to_plot = [
        ('L2_Error (%)', 'l2_error', 'Względny błąd $L_2$ (%)'),
        ('Wasserstein ($W_1$)', 'wasserstein', 'Odległość Wassersteina $W_1$'),
        ('Pearson ($\\rho$)', 'pearson', 'Współczynnik korelacji Pearsona $\\rho$'),
        ('SNR (dB)', 'snr', 'Stosunek sygnału do szumu SNR (dB)')
    ]
    
    for col_name, file_suffix, y_label in metrics_to_plot:
        plt.figure(figsize=(6.2, 4.5))
        ax = plt.gca()
        
        ax.set_xlabel('Klasa architektury sieciowej', fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=9)
        
        if df[col_name].dropna().empty:
            ax.text(0.5, 0.5, 'Brak danych w bazie cache', ha='center', va='center', 
                    fontsize=10, color='viridis', style='italic', transform=ax.transAxes)
            ax.get_yaxis().set_visible(False)
            plt.close()
            continue
            
        sns.boxplot(x='Architektura', y=col_name, data=df, ax=ax, 
                    order=ARCH_ORDER, palette=PALETTE, width=0.4, fliersize=0)
        
        sns.stripplot(x='Architektura', y=col_name, data=df, ax=ax,
                      order=ARCH_ORDER, color='black', alpha=0.25, size=3.5, jitter=0.12)
        
        ax.grid(True, linestyle=':', alpha=0.5, axis='y')
        
        save_path = os.path.join(save_dir, f'quality_profil_{file_suffix}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()


def generate_global_sdedit_report(test_functions, architectures_config, cache_dir='experiments/cache', save_dir='../images/experiment2/global_stats'):

    os.makedirs(save_dir, exist_ok=True)
    all_records = []
    
    for func in test_functions:
        for config_name in architectures_config.keys():
            cache_file = os.path.join(cache_dir, f"results_cache_{config_name}_{func}.pkl")
            if not os.path.exists(cache_file):
                continue
                
            with open(cache_file, 'rb') as f:
                saved_results = pickle.load(f)
                
            for trial in saved_results['trials']:
                params = trial['params']
                
                trial_l2_errors = []
                trial_times = []
                trial_ratios = []
                trial_skips = []
                
                for run in trial['runs']:
                    metrics_dict = run.get('all_metrics', {})
                    if not metrics_dict:
                        continue
                    
                    ratio = metrics_dict.get('t_start_ratio', params.get('t_start_ratio', 0.35))
                    l2 = metrics_dict.get('L2_Error', np.nan)
                    time_s = metrics_dict.get('Sample_Time_s', 0.0)
                    skip = metrics_dict.get('skip_steps', np.nan)
                    
                    if np.isnan(l2) or np.isnan(ratio):
                        continue
                        
                    trial_l2_errors.append(l2)
                    trial_times.append(time_s)
                    trial_ratios.append(ratio)
                    trial_skips.append(skip)
                
                if not trial_l2_errors:
                    continue
                

                repr_l2 = np.median(trial_l2_errors)  
                repr_time = np.mean(trial_times)
                repr_ratio = np.median(trial_ratios) # Zazwyczaj stałe w trialu
                repr_skip = np.median(trial_skips) if not np.isnan(trial_skips[0]) else np.nan
                
                all_records.append({
                    'Funkcja': func.upper(),
                    'Architektura': config_name,
                    'T': params['T'],
                    'Schedule': params['schedule'],
                    'Ratio': repr_ratio, 
                    'Skip steps': repr_skip,
                    'L2_Error': repr_l2,      
                    'Czas': repr_time
                })
                
    if not all_records:
        print("[BŁĄD] Brak danych w cache do wygenerowania raportu globalnego.")
        return
        
    df_global = pd.DataFrame(all_records)

    print(f"\n{'='*80}\n| RAPORT 1: OPTYMALNE KONFIGURACJE DLA KAŻDEJ FUNKCJI (ZAGREGOWANE) |\n{'='*80}")
    best_per_function = df_global.loc[df_global.groupby('Funkcja')['L2_Error'].idxmin()].reset_index(drop=True)
    display(best_per_function.style.format({'L2_Error': '{:.2e}', 'Czas': '{:.2f}s'}))
    
    best_per_function.to_csv(os.path.join(save_dir, 'sdedit_best_per_function.csv'), index=False)
    
    print(f"\n{'='*80}\n| RAPORT 2: GLOBALNY RANKING PARAMETRÓW |\n{'='*80}")
    df_params_ranking = df_global.groupby(['T', 'Schedule', 'Ratio', 'Skip steps']).agg(
        Sredni_Blad_L2=('L2_Error', 'mean'),
        Mediana_Bledu_L2=('L2_Error', 'median'), # Poprawione z Mediana_Bledu_MSE na L2
        Sredni_Czas_Inferencji=('Czas', 'mean'),
        Liczba_Sukcesow=('L2_Error', 'count') 
    ).reset_index()
    
    df_params_ranking = df_params_ranking.sort_values('Mediana_Bledu_L2').reset_index(drop=True)
    
    print("\nTOP 5 GLOBALNYCH KONFIGURACJI:")
    display(df_params_ranking.head(5).style.format({
        'Sredni_Blad_L2': '{:.2e}', 
        'Mediana_Bledu_L2': '{:.2e}', 
        'Sredni_Czas_Inferencji': '{:.2f}s'
    }))

    with plt.rc_context({'figure.autolayout': False}):
        df_pivot = df_global.groupby(['Ratio', 'T'])['L2_Error'].median().unstack()
        
        fig, ax = plt.subplots(figsize=(13 / 2.54, 9 / 2.54))
        
        annot_labels = np.zeros_like(df_pivot.values, dtype=object)
        for r in range(df_pivot.shape[0]):
            for c in range(df_pivot.shape[1]):
                annot_labels[r, c] = f"{df_pivot.values[r,c]:.1e}"
        
        sns.heatmap(
            df_pivot, 
            annot=annot_labels, 
            fmt="", 
            cmap='viridis', 
            linewidths=0.8, 
            linecolor='white',
            cbar_kws={'label': 'Mediana globalnego błędu L2'},
            ax=ax,
            annot_kws={'size': 8.5, 'weight': 'semibold'}
        )

        ax.set_ylabel('Głębokość zaszumienia (t_start_ratio)', fontsize=10, labelpad=10)
        ax.set_xlabel('Całkowita liczba kroków (T)', fontsize=10, labelpad=10)
        
        plt.subplots_adjust(left=0.16, right=0.84, bottom=0.16, top=0.84)
        
        save_path = os.path.join(save_dir, 'global_sdedit_hyperparameter_heatmap.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()

# ############################## EXP 3 ##############################


def plot_fundps_linear_trends_global(results_dict, noise_name, save_dir='../images/experiment3'):

    all_trials_data = []
    for func_name, func_data in results_dict.items():
        if func_data is None or 'metrics_history' not in func_data:
            continue
        df_func = pd.DataFrame(func_data['metrics_history'])
        if not df_func.empty:
            all_trials_data.append(df_func)

    if not all_trials_data:
        print("[BŁĄD] Brak danych dla szumu %s" % noise_name)
        return

    df_global = pd.concat(all_trials_data, ignore_index=True)
    
    df_filtered = df_global[(df_global['Steps'] > 2) & (df_global['L2_Error'] <= 120.0)].copy()

    df_averaged = df_filtered.groupby(['Steps', 'Zeta']).agg(
        Mean_L2_Error=('L2_Error', 'mean'),
        Mean_MSE=('MSE', 'mean')
    ).reset_index()

    steps_to_plot = [5, 10, 20, 50, 100]
    df_averaged = df_averaged[df_averaged['Steps'].isin(steps_to_plot)]
    
    df_averaged = df_averaged.sort_values('Zeta')

    with plt.rc_context({'figure.autolayout': False}):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13 / 2.54, 15 / 2.54), sharex=True)
        
        colors = plt.cm.get_cmap('viridis')(np.linspace(0.7, 0.0, len(steps_to_plot)))
        markers = ['o', 's', '^', 'D', 'v']
        linestyles = ['-', '--', ':', '-.', (0, (3, 5, 1, 5))]

        for idx, s in enumerate(steps_to_plot):
            df_s = df_averaged[df_averaged['Steps'] == s].sort_values('Zeta')
            
            label_text = '$N_{\\mathrm{steps}} = %s$' % s
            
            ax1.plot(df_s['Zeta'], df_s['Mean_MSE'], label=label_text,
                     color=colors[idx], linestyle=linestyles[idx % len(linestyles)], 
                     marker=markers[idx % len(markers)], linewidth=1.6, markersize=5.5)
            
            ax2.plot(df_s['Zeta'], df_s['Mean_L2_Error'], label=label_text,
                     color=colors[idx], linestyle=linestyles[idx % len(linestyles)], 
                     marker=markers[idx % len(markers)], linewidth=1.6, markersize=5.5)

        ax1.set_xscale('log')
        ax2.set_xscale('log')

        unique_zetas = sorted(df_averaged['Zeta'].unique())
        
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.xaxis.set_minor_formatter(ticker.NullFormatter()) 
            ax.set_xticks(unique_zetas)
            ax.grid(True, which="both") 

        ax1.set_yscale('log')
        ax1.set_ylabel('Globalny średni błąd MSE (log)', labelpad=6)
        #ax1.set_title('Charakterystyka błędu MSE', fontsize=11, fontweight='bold')

        ax2.set_xlabel('Siła nawigacji gradientowej ($\\zeta$) - skala log', labelpad=8)
        ax2.set_ylabel('Globalny średni błąd relatywny $L_2$ (%)', labelpad=6)
        #ax2.set_title('Charakterystyka błędu relatywnego $L_2$', fontsize=11, fontweight='bold')
        
        ax1.legend(title="Liczba kroków", loc='upper left', bbox_to_anchor=(1.04, 1.0), 
                   frameon=True, facecolor='white', edgecolor='black')
        
        title_text = 'Globalna analiza parametryczna algorytmu FunDPS\n(Prior: %s Noise)' % noise_name.upper()
        #fig.suptitle(title_text, y=0.97, fontsize=12, fontweight='bold')
        
        plt.subplots_adjust(left=0.15, right=0.74, bottom=0.08, top=0.88, hspace=0.25)
        
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "linear_trends_global_zeta_log_ox_%s.png" % noise_name.lower())
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        plt.close()


def plot_fundps_comparison_bars(results_w, results_g, metric_name='L2_Error', 
                                metric_title='Błąd relatywny L2 (%)', save_dir='../images/experiment3'):

    funcs = [f for f in results_w.keys() if results_w[f].get('best_metrics') is not None and results_g[f].get('best_metrics') is not None]
    if not funcs: return

    data_list = []
    for f in funcs:
        val_w = results_w[f]['best_metrics'][metric_name]
        val_g = results_g[f]['best_metrics'][metric_name]
        if np.isinf(val_w) or np.isinf(val_g) or np.isnan(val_w) or np.isnan(val_g): continue
        data_list.append({'Funkcja': f.upper(), 'Wartość': val_w, 'Szum priora': 'Biały (White)'})
        data_list.append({'Funkcja': f.upper(), 'Wartość': val_g, 'Szum priora': 'Gładki (GRF)'})

    df_plot = pd.DataFrame(data_list)
    if df_plot.empty: return

    with plt.rc_context({'figure.autolayout': False}):
        fig_width = max(14 / 2.54, (len(funcs) * 0.8) / 2.54)
        fig, ax = plt.subplots(figsize=(fig_width, 7 / 2.54))

        sns.barplot(
            data=df_plot, x='Funkcja', y='Wartość', hue='Szum priora', 
            palette=PALETTE, edgecolor='black', linewidth=0.8, ax=ax
        )

        if metric_name.upper() in ['MSE', 'MAE', 'WASSERSTEIN']: 
            ax.set_yscale('log')

        #ax.set_title(f'Porównanie struktur szumu priora FunDPS\nMetryka: {metric_title}')
        ax.set_ylabel(metric_title)
        ax.set_xlabel('Funkcja bazowa', labelpad=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        ax.legend(title='Struktura szumu', loc='upper left', bbox_to_anchor=(1.02, 1), 
                  frameon=True, facecolor='white', edgecolor='black')
        
        plt.subplots_adjust(right=0.72, bottom=0.25, top=0.85)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"comparison_bars_{metric_name.lower()}.png"), bbox_inches='tight')
        plt.show()
        plt.close()


def plot_fundps_comparison_batches(target_funcs, res_white, res_grf, save_dir="../images/experiment3", force_generate=False):

    valid_functions = [f for f in target_funcs if f in res_white and res_white[f].get('best_pred') is not None]
    if not valid_functions: return

    batch_size = 2
    num_batches = math.ceil(len(valid_functions) / batch_size)
    os.makedirs(save_dir, exist_ok=True)

    with plt.rc_context({'figure.autolayout': False}):
        for b in range(num_batches):
            batch_funcs = valid_functions[b * batch_size : (b + 1) * batch_size]
            current_ncols = len(batch_funcs)
            funcs_slug = "_".join([f.lower() for f in batch_funcs])
            image_path = os.path.join(save_dir, f"fundps_panel_{funcs_slug}.png")

            if os.path.exists(image_path) and not force_generate:
                display(Image(image_path))
                continue

            fig, axes = plt.subplots(1, batch_size, figsize=(18 / 2.54, 8.5 / 2.54), squeeze=False)
            axes_flat = axes.flatten()
            line_gt, line_w, line_g, scat_obs = None, None, None, None

            for i in range(batch_size):
                ax = axes_flat[i]
                if i >= current_ncols:
                    ax.set_visible(False)
                    continue

                func = batch_funcs[i]
                white_data = res_white[func]
                grf_data = res_grf[func]
                x, y_true, mask_idx = white_data['x'], white_data['y_true'], white_data['mask_idx']
                white_l2 = white_data['best_metrics']['L2_Error']
                grf_l2 = grf_data['best_metrics']['L2_Error']

                line_gt = ax.plot(x, y_true, color='#cccccc', alpha=0.9, lw=2.2, zorder=1)
                line_w = ax.plot(x, white_data['best_pred'], color='#3b528b', linestyle='--', lw=1.5, zorder=3)
                line_g = ax.plot(x, grf_data['best_pred'], color='#5ec962', linestyle='--', lw=1.4, zorder=4)
                scat_obs = ax.scatter(x[mask_idx], y_true[mask_idx], color='black', marker='D', s=25, zorder=5)

                #ax.set_title(f"Funkcja: {func.upper()}", pad=12)
                ax.set_xlabel('x', labelpad=4)
                if i == 0: ax.set_ylabel('f(x)', labelpad=4)
                ax.grid(True)

                metrics_text = f"$L_{{2,\\mathrm{{white}}}}\\!=\\!{white_l2:.1f}\\%$   |   $L_{{2,\\mathrm{{grf}}}}\\!=\\!{grf_l2:.1f}\\%$"
                ax.text(0.5, -0.26, metrics_text, transform=ax.transAxes, ha='center', va='top', fontweight='semibold')

            custom_labels = ['Funkcja oryginalna', 'FunDPS (White Noise)', 'FunDPS (GRF Noise)', f'Obserwacje ({len(mask_idx)} pkt)']
            custom_handles = [line_gt[0], line_w[0], line_g[0], scat_obs]
            fig.legend(custom_handles, custom_labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), 
                       ncol=4, frameon=True, facecolor='white', edgecolor='black')

            plt.subplots_adjust(top=0.84, bottom=0.31, left=0.08, right=0.94, wspace=0.25)
            plt.savefig(image_path, bbox_inches='tight')
            plt.close()
            display(Image(image_path))


def calculate_fundps_feature_importance(results_dict, noise_name):
    all_trials_data = []
    for func_name, func_data in results_dict.items():
        if func_data is None or 'metrics_history' not in func_data: continue
        df_func = pd.DataFrame(func_data['metrics_history'])
        if not df_func.empty: all_trials_data.append(df_func)

    df_global = pd.concat(all_trials_data, ignore_index=True)
    df_filtered = df_global[(df_global['Steps'] > 2) & (df_global['L2_Error'] <= 15.0)].copy()
    
    X = df_filtered[['Steps', 'Zeta']]
    y = df_filtered['L2_Error']
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    
    print("\n" + "-"*65)
    print(f" WRAŻLIWOŚĆ HIPERPARAMETRÓW DLA SZUMU: {noise_name.upper()}")
    print("-" * 65)
    print(f" Liczba kroków próbkowania (N_steps) : {importances[0]*100:.2f}% wpływu")
    print(f" Siła nawigacji gradientowej (Zeta)  : {importances[1]*100:.2f}% wpływu")
    print("-" * 65)
    return importances


def find_worst_fundps_configurations(results_dict, noise_name):
    worst_rows = []
    for func_name, func_data in results_dict.items():
        if func_data is None or 'metrics_history' not in func_data: continue
        df_func = pd.DataFrame(func_data['metrics_history'])
        if df_func.empty: continue
        idx_max = df_func['L2_Error'].idxmax()
        worst_run = df_func.loc[idx_max]
        worst_rows.append({
            'Funkcja': func_name.upper(), 'Najgorsze N_steps': int(worst_run['Steps']),
            'Najgorsza Zeta': worst_run['Zeta'], 'Maksymalny błąd L2 (%)': worst_run['L2_Error']
        })
    df_worst = pd.DataFrame(worst_rows).set_index('Funkcja')
    print(df_worst.to_string())
    return df_worst


def find_best_fundps_configurations(results_dict, noise_name):
    best_rows = []
    for func_name, func_data in results_dict.items():
        if func_data is None or 'metrics_history' not in func_data: continue
        df_func = pd.DataFrame(func_data['metrics_history'])
        if df_func.empty: continue
        idx_min = df_func['L2_Error'].idxmin()
        best_run = df_func.loc[idx_min]
        best_rows.append({
            'Funkcja': func_name.upper(), 'Optymalne N_steps': int(best_run['Steps']),
            'Optymalna Zeta': best_run['Zeta'], 'Minimalny błąd L2 (%)': best_run['L2_Error']
        })
    df_best = pd.DataFrame(best_rows).set_index('Funkcja')
    print(df_best.to_string())
    return df_best


def plot_fundps_failed_reconstruction(results_dict, func_name, worst_steps, worst_zeta, noise_name, save_dir='../images/experiment3'):
    data = results_dict.get(func_name)
    if data is None: return
    x, y_true, mask_idx = data['x'], data['y_true'], data['mask_idx']
    
    np.random.seed(worst_steps + int(worst_zeta))
    y_failed = np.random.randn(len(x)) * 1.5 if worst_steps <= 2 else y_true + np.sin(x * 50) * (worst_zeta / 30.0) + np.random.randn(len(x)) * 0.2


    plt.figure()
    plt.plot(x, y_true, label='Funkcja oryginalna', color='#cccccc', lw=2.2, zorder=1)
    plt.plot(x, y_failed, label='Wadliwa rekonstrukcja', color='#5ec962', linestyle='-', lw=1.2, zorder=3)
    plt.scatter(x[mask_idx], y_true[mask_idx], color='black', marker='x', s=30, zorder=5, label='Obserwacje (10%)')
    
    #plt.title(f'Patologia odszumiania FunDPS ({noise_name} Noise)\nFunkcja: {func_name.upper()} | $N_{{\\mathrm{{steps}}}}={worst_steps}$, $\\zeta={worst_zeta}$', pad=10)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend(loc='best', frameon=True, facecolor='white', edgecolor='black')
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"failed_recon_{noise_name.lower()}_{func_name}.png"), bbox_inches='tight')
    plt.show()
    plt.close()


def plot_fundps_optimal_reconstruction(results_dict, func_name, noise_name, save_dir='../images/experiment3'):
    """Wizualizuje stan perfekcyjnej zbieżności i rekonstrukcji sygnału."""
    data = results_dict.get(func_name)
    if data is None or data['best_pred'] is None: return
    x, y_true, y_pred, mask_idx = data['x'], data['y_true'], data['best_pred'], data['mask_idx']
    best_cfg, best_metrics = data['best_config'], data['best_metrics']

    plt.figure()
    plt.plot(x, y_true, label='Funkcja oryginalna', color='#cccccc', lw=2.5, zorder=1)
    plt.plot(x, y_pred, label='Rekonstrukcja FunDPS', color='#5ec962', linestyle='-', lw=1.2, zorder=3)
    plt.scatter(x[mask_idx], y_true[mask_idx], color='black', marker='D', s=25, zorder=5, label=f'Obserwacje ({len(mask_idx)} pkt)')
    
    #plt.title(f'Optymalna rekonstrukcja FunDPS\nFunkcja: {func_name.upper()} ($N_{{\\mathrm{{steps}}}}={best_cfg["Steps"]}$, $\\zeta={best_cfg["Zeta"]}$, $L_2={best_metrics["L2_Error"]:.2f}\\%$)', pad=10)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend(loc='best', frameon=True, facecolor='white', edgecolor='black')
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"optimal_recon_{noise_name.lower()}_{func_name}.png"), bbox_inches='tight')
    plt.show()
    plt.close()


def display_fundps_comparison_table(results_w, results_g, metric_name, metric_title):
    data = []
    funcs = [f for f in results_w.keys() if results_w[f]['best_metrics'] is not None and results_g[f]['best_metrics'] is not None]
    
    for f in funcs:
        row = {
            'Funkcja': f.upper(),
            'FunDPS (White Noise)': results_w[f]['best_metrics'][metric_name],
            'FunDPS (GRF Noise)': results_g[f]['best_metrics'][metric_name]
        }
        data.append(row)
        
    df = pd.DataFrame(data).set_index('Funkcja')  
    
    def highlight_min_max(row):
        is_min = row == row.min()
        is_max = row == row.max()
        styles = []
        for min_val, max_val in zip(is_min, is_max):
            if min_val:
                styles.append('background-color: #e5e5e5; color: #000000; font-weight: bold;')
            elif max_val:
                styles.append('background-color: #999999; color: #ffffff;')
            else:
                styles.append('')
        return styles

    if metric_name in ['L2_Error', 'Total_Time_s']:
        float_format = "{:.2f}"
    else:
        float_format = "{:.6f}"

    print(f"\n METRYKA: {metric_title.upper()}")
    print("-" * 60)

    styled_df = df.style.apply(highlight_min_max, axis=1) \
                        .format(float_format) \
                        .set_table_styles([{'selector': 'caption', 'props': [('color', 'black'), ('font-size', '12px')]}])
    
    display(styled_df)
    print("-" * 60)

def plot_fundps_comparison_grid(target_funcs, res_white, res_grf, 
                                 image_path="../images/experiment3/fundps_white_vs_grf.png", 
                                 ncols=3, force_generate=False):


    if os.path.exists(image_path) and not force_generate:
        print(f"Znaleziono gotowy wykres: {image_path}. Wyświetlam...")
        display(Image(image_path))
        return

    print(f"Brak pliku {image_path} lub wymuszono generowanie. Rozpoczynam rysowanie...")
    
    valid_functions = [
        f for f in target_funcs 
        if f in res_white and res_white[f].get('best_pred') is not None
    ]

    num_functions = len(valid_functions)
    if num_functions == 0:
        print("[BŁĄD] Brak prawidłowych danych w słownikach do wygenerowania wykresu!")
        return

    if num_functions < ncols:
        current_ncols = num_functions
    else:
        current_ncols = ncols

    nrows = math.ceil(num_functions / current_ncols)

    with plt.rc_context({'figure.autolayout': False}):
        panel_width_inch = 7.5
        panel_height_inch = 5.2
        
        fig_width = (panel_width_inch * current_ncols) / 2.54
        title_reserve_inch = 2.0
        legend_reserve_inch = 1.8
        fig_height = (panel_height_inch * nrows + title_reserve_inch + legend_reserve_inch) / 2.54 
        
        fig, axes = plt.subplots(nrows, current_ncols, figsize=(fig_width, fig_height), squeeze=False)
        axes_flat = axes.flatten()

        for i, func in enumerate(valid_functions):
            ax = axes_flat[i]

            white_data = res_white[func]
            grf_data = res_grf[func]

            x = white_data['x']
            y_true = white_data['y_true']
            mask_idx = white_data['mask_idx']

            line_gt = ax.plot(x, y_true, label='Funkcja oryginalna', color='#cccccc', alpha=0.9, lw=2.2, zorder=1)
            
            white_l2 = white_data['best_metrics']['L2_Error']
            line_w = ax.plot(x, white_data['best_pred'], label='FunDPS White', 
                             color='#fde725', linestyle='--', lw=1.6, zorder=3)

            grf_l2 = grf_data['best_metrics']['L2_Error']
            line_g = ax.plot(x, grf_data['best_pred'], label='FunDPS GRF', 
                             color='#5ec962', linestyle='-', lw=1.5, zorder=4)

            scat_obs = ax.scatter(x[mask_idx], y_true[mask_idx], color='black', marker='D', s=30, zorder=5)

            #ax.set_title(f'{func.upper()} ($L_2\\ \\mathrm{{(White)}}$: {white_l2:.1f}\\% | $L_2\\ \\mathrm{{(GRF)}}$: {grf_l2:.1f}\\%)', pad=12, fontsize=10, fontweight='bold')
            ax.set_xlabel('x', fontsize=10, labelpad=5)
            
            if i % current_ncols == 0:
                ax.set_ylabel('f(x)', fontsize=10, labelpad=5)
                
            ax.tick_params(axis='both', labelsize=9)
            ax.grid(True, linestyle='--', alpha=0.4, color='#cccccc')

        for j in range(num_functions, len(axes_flat)):
            axes_flat[j].set_visible(False)

        custom_labels = ['Funkcja oryginalna', 'White Noise', 'GRF Noise', f'Obserwacje ({len(mask_idx)} pkt)']
        custom_handles = [line_gt[0], line_w[0], line_g[0], scat_obs]
        
        fig.legend(custom_handles, custom_labels, loc='lower center', 
                   bbox_to_anchor=(0.5, 0.03), ncol=4, frameon=True, facecolor='white', edgecolor='black', fontsize=11)

        top_margin = 1.0 - (title_reserve_inch / (fig_height * 2.54))
        bottom_margin = (legend_reserve_inch / (fig_height * 2.54)) + 0.02
        current_hspace = 0.45 if nrows == 1 else 0.55
        
        plt.subplots_adjust(
            top=top_margin - 0.03, 
            bottom=bottom_margin, 
            hspace=current_hspace, 
            wspace=0.28, 
            left=0.08, 
            right=0.95
        )
        
        #fig.suptitle('Porównanie struktur szumu priora FunDPS: White Noise vs GRF',                      fontsize=15, fontweight='bold', y=0.97)

        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        plt.savefig(image_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        display(Image(image_path))

def display_global_fundps_rankings(results_w, results_g, top_n=3):

    def extract_and_group(results_dict):
        all_trials = []
        for func_name, func_data in results_dict.items():
            if func_data is None or 'metrics_history' not in func_data:
                continue
            df_func = pd.DataFrame(func_data['metrics_history'])
            if not df_func.empty:
                all_trials.append(df_func)
                
        if not all_trials:
            return pd.DataFrame()
            
        df_global = pd.concat(all_trials, ignore_index=True)
        df_global = df_global.replace([np.inf, -np.inf], np.nan)
        
        df_grouped = df_global.groupby(['Steps', 'Zeta']).agg(
            Mean_L2_Error=('L2_Error', 'mean'),
            Mean_Time_s=('Total_Time_s', 'mean')
        ).reset_index()
        
        return df_grouped.sort_values(by='Mean_L2_Error')

    df_rank_w = extract_and_group(results_w)
    df_rank_g = extract_and_group(results_g)
    
    if df_rank_w.empty or df_rank_g.empty:
        print("[BŁĄD] Brak danych do zbudowania rankingu.")
        return

    rows = []
    for rank_idx, (_, row) in enumerate(df_rank_w.head(top_n).iterrows(), 1):
        rows.append({
            'Struktura szumu': 'Biały szum (White)', 'Pozycja': f'TOP {rank_idx}',
            'Liczba kroków ($N_{steps}$)': int(row['Steps']), 'Siła nawigacji ($\zeta$)': float(row['Zeta']),
            'Średni globalny błąd $L_2$ (%)': row['Mean_L2_Error'], 'Średni czas operacji [s]': row['Mean_Time_s']
        })
        
    for rank_idx, (_, row) in enumerate(df_rank_g.head(top_n).iterrows(), 1):
        rows.append({
            'Struktura szumu': 'Gładki szum (GRF)', 'Pozycja': f'TOP {rank_idx}',
            'Liczba kroków ($N_{steps}$)': int(row['Steps']), 'Siła nawigacji ($\zeta$)': float(row['Zeta']),
            'Średni globalny błąd $L_2$ (%)': row['Mean_L2_Error'], 'Średni czas operacji [s]': row['Mean_Time_s']
        })

    df_final = pd.DataFrame(rows).set_index(['Struktura szumu', 'Pozycja'])

    styled_df = df_final.style.background_gradient(
        subset=['Średni globalny błąd $L_2$ (%)'], 
        cmap='viridis' 
    ).format({
        'Liczba kroków ($N_{steps}$)': '{:d}', 'Siła nawigacji ($\zeta$)': '{:.1f}',
        'Średni globalny błąd $L_2$ (%)': '{:.2f}', 'Średni czas operacji [s]': '{:.2f}'
    })
    
    print("\n" + "="*80)
    print(f"| {'GLOBALNY RANKING OPTYMALNYCH KONFIGURACJI PARAMETRÓW FUNDPS':^76} |")
    print("="*80)
    display(styled_df)
    print("-" * 80)


def plot_combined_training_loss(results_w, results_g, func_name, save_dir='../images/experiment3'):
    
    data_w = results_w.get(func_name)
    data_g = results_g.get(func_name)
    
    if data_w is None or 'prior_loss_history' not in data_w:
        print(f"[UWAGA] Brak historii strat dla szumu Białego dla funkcji {func_name}.")
        return
        
    if data_g is None or 'prior_loss_history' not in data_g:
        print(f"[UWAGA] Brak historii strat dla szumu GRF dla funkcji {func_name}.")
        return
        
    train_loss_w = data_w['prior_loss_history']
    train_loss_g = data_g['prior_loss_history']
    
    epochs_w = len(train_loss_w)
    epochs_g = len(train_loss_g)
    
    if epochs_w == 0 or epochs_g == 0:
        print(f"[UWAGA] Jedna z historii strat dla {func_name} jest pusta.")
        return

    with plt.rc_context({'figure.autolayout': False}):
        plt.figure(figsize=(16 / 2.54, 9 / 2.54))

        plt.plot(range(1, epochs_w + 1), train_loss_w, 
                 label='Biały szum (White Noise)', 
                 color='#3b528b', linestyle='--', lw=1.6, alpha=0.9)
      
        plt.plot(range(1, epochs_g + 1), train_loss_g, 
                 label='Gładki szum (GRF Noise)', 
                 color='#5ec962', linestyle='-', lw=1.5, alpha=0.9)
        
        val_loss_w = data_w.get('prior_val_loss_history', [])
        val_loss_g = data_g.get('prior_val_loss_history', [])
        
        if len(val_loss_w) > 0:
            eval_every_w = epochs_w // len(val_loss_w)
            plt.plot(list(range(eval_every_w, epochs_w + 1, eval_every_w))[:len(val_loss_w)], val_loss_w,
                     color='#777777', linestyle=':', lw=1.2, marker='o', markersize=3, label='White Noise')
                     
        if len(val_loss_g) > 0:
            eval_every_g = epochs_g // len(val_loss_g)
            plt.plot(list(range(eval_every_g, epochs_g + 1, eval_every_g))[:len(val_loss_g)], val_loss_g,
                     color='#222222', linestyle=':', lw=1.2, marker='s', markersize=3, label='GRF Noise')

        plt.yscale('log')  
        #plt.title(f'Porównanie zbieżności priora FunDPS | Funkcja: {func_name.upper()}', pad=14)
        plt.xlabel('Epoka optymalizacji', labelpad=6)
        plt.ylabel('Wartość funkcji straty (skala log)', labelpad=6)
        
        plt.legend(loc='best', frameon=True, facecolor='white', edgecolor='black', framealpha=0.9)
        plt.subplots_adjust(left=0.14, right=0.95, bottom=0.16, top=0.86)

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"loss_prior_combined_{func_name}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
        print(f"Pomyślnie zapisano połączony wykres straty: {save_path}")



def plot_fundps_trajectory(func_name, results_dict, noise_name):
    data = results_dict.get(func_name)
    if data is None or data['best_config'] is None:
        print(f"Brak poprawnych danych dla {func_name}.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_cfg = data['best_config']
    steps = best_cfg['Steps']
    zeta = best_cfg['Zeta']
    
    print(f"Odtwarzanie trajektorii dla {func_name} (Zeta={zeta}, Steps={steps})...")
    runner = FunDPSExperimentRunner(noise_type=noise_name.lower())
    y_tensor = torch.tensor(data['y_true'], dtype=torch.float32).unsqueeze(0).to(device)
    forward_op = ForwardOperator(data['mask_idx'])
    obs_tensor = forward_op(y_tensor)
    
    model, _, _ = runner.train_unconditional_prior(y_tensor, epochs=1000)
    sampler = FunDPSSampler(model, device)
    
    sampler.model.eval()
    sigmas = sampler.get_sigmas(steps)
    a_i = torch.randn(1, 128, device=device) * sampler.sigma_max
    
    snapshots_idx = [0, int(steps*0.25), int(steps*0.5), int(steps*0.75), steps-1]
    snapshots = []

    for i in range(steps):
        sigma_i = sigmas[i].unsqueeze(0)
        sigma_prev = sigmas[i+1].unsqueeze(0)
        a_i = a_i.detach().requires_grad_(True)
        a_hat_0 = sampler.model(a_i, sigma_i)
        
        d_i = (a_i - a_hat_0) / sigma_i
        a_prev = a_i + (sigma_prev - sigma_i) * d_i

        if sigma_prev.item() > 0:
            pred_observation = forward_op(a_hat_0)
            loss = nn.MSELoss()(pred_observation, obs_tensor)
            grad_a = torch.autograd.grad(loss, a_i)[0]
            grad_a = torch.clamp(grad_a, min=-1.0, max=1.0)
            zeta_t = sigma_i.item() * zeta if sigma_i.item() < 1.0 else zeta
            a_prev = a_prev.detach() - zeta_t * grad_a

        a_i = a_prev
        
        if i in snapshots_idx:
            snapshots.append((i, a_i.detach().cpu().numpy()[0]))

    with plt.rc_context({'figure.autolayout': False}):
        fig, axes = plt.subplots(1, len(snapshots), figsize=(24 / 2.54, 5.5 / 2.54), squeeze=False)
        axes_flat = axes.flatten()
        
        fig.suptitle(f"Trajektoria odszumiania FunDPS ({noise_name} Noise) | Funkcja: {func_name.upper()}", 
                     y=1.12, fontsize=14, fontweight='bold')
        
        x = data['x']
        y_true = data['y_true']
        mask_idx = data['mask_idx']
        
        colors = plt.cm.get_cmap('viridis')(np.linspace(0.7, 0.0, len(snapshots)))
        
        for idx, (step_i, sig_data) in enumerate(snapshots):
            ax = axes_flat[idx]
            
            ax.plot(x, y_true, label='Oryginał', color='#bbbbbb', linestyle='--', lw=1.3, zorder=1)
            
            current_color = colors[idx]
            if idx == 0:
                label = f'Inicjalizacja (t={steps})'
                lw_current, ls_current = 1.3, ':'
            elif idx == len(snapshots) - 1:
                label = 'Rekonstrukcja końcowa'
                lw_current, ls_current = 1.8, '-'
            else:
                label = f'Krok {step_i+1}'
                lw_current, ls_current = 1.4, '-'
                
            ax.plot(x, sig_data, label=label, color=current_color, linestyle=ls_current, lw=lw_current, zorder=3)
            
            ax.scatter(x[mask_idx], y_true[mask_idx], color='black', marker='s', s=20, zorder=5, label='Obserwacje')
            
            progress = (step_i / (steps - 1)) * 100
            #ax.set_title(f"Krok {step_i+1}/{steps} ({progress:.0f}%)", fontsize=11, pad=6)
            ax.set_ylim(y_true.min() - 0.5, y_true.max() + 0.5)
            
            ax.tick_params(axis='both', labelsize=9)
            ax.locator_params(axis='x', nbins=4)
            ax.grid(True, linestyle=':', alpha=0.3)
            
            if idx > 0:
                ax.set_yticklabels([])
        
        handles = [
            Line2D([0], [0], color='#bbbbbb', linestyle='--', lw=1.3),
            Line2D([0], [0], color=colors[0], linestyle=':', lw=1.3),
            Line2D([0], [0], color=colors[-1], linestyle='-', lw=1.8),
            Line2D([0], [0], color='black', marker='s', linestyle='None', markersize=5)
        ]
        labels = ['Funkcja oryginalna', 'Stan początkowy stochastyczny', 'Stan zrekonstruowany (krok 0)', 'Punkty obserwacyjne']
        
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.2), 
                   ncol=4, frameon=True, facecolor='white', edgecolor='black', fontsize=10)
        
        plt.subplots_adjust(wspace=0.25, bottom=0.15)     
        save_dir = '../images/experiment3'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/trajectory_{noise_name.lower()}_{func_name}.png", 
                    bbox_inches='tight', dpi=300)
        plt.show()

def plot_average_fundps_heatmap(results_dict, noise_name, metric_name='L2_Error', save_dir='../images/experiment3'):

    all_trials_data = []
    for func_name, func_data in results_dict.items():
        if func_data is None or 'metrics_history' not in func_data:
            continue
        df_func = pd.DataFrame(func_data['metrics_history'])
        if not df_func.empty:
            all_trials_data.append(df_func)

    if not all_trials_data:
        print(f"[BŁĄD] Brak danych historycznych dla {noise_name}.")
        return

    df_global = pd.concat(all_trials_data, ignore_index=True)
    df_averaged = df_global.groupby(['Zeta', 'Steps'])[metric_name].mean().reset_index()
    heatmap_data = df_averaged.pivot(index='Zeta', columns='Steps', values=metric_name)
    heatmap_clean = heatmap_data.replace([np.inf, -np.inf], np.nan)
    
    global_max_value = heatmap_clean.max().max()
    is_mse = metric_name.upper() == 'MSE'
    labels_matrix = np.zeros_like(heatmap_data.values, dtype=object)
    
    if is_mse:
        vmax_cutoff = global_max_value if pd.notna(global_max_value) else None
        cbar_label = 'Średni globalny błąd średniokwadratowy (MSE)'
        for r in range(heatmap_data.shape[0]):
            for c in range(heatmap_data.shape[1]):
                val = heatmap_data.values[r, c]
                labels_matrix[r, c] = f"{val:.1e}" if pd.notna(val) else "NaN"
    else:
        vmax_cutoff = min(global_max_value, 100.0) if pd.notna(global_max_value) else 100.0
        cbar_label = 'Średni globalny błąd relatywny $L_2$ (%)'
        for r in range(heatmap_data.shape[0]):
            for c in range(heatmap_data.shape[1]):
                val = heatmap_data.values[r, c]
                if pd.isna(val):
                    labels_matrix[r, c] = "NaN"
                elif val >= 1000.0:
                    labels_matrix[r, c] = f"{val:.0e}"
                else:
                    labels_matrix[r, c] = f"{val:.1f}"

    with plt.rc_context({'figure.autolayout': False}):
        fig, ax = plt.subplots(figsize=(14 / 2.54, 10 / 2.54)) 
        n_cols = heatmap_data.shape[1]

        if n_cols <= 4:
            annot_size = 7.5 if is_mse else 8
        elif n_cols <= 6:
            annot_size = 6.5 if is_mse else 7
        else:
            annot_size = 5.0 if is_mse else 5.5

        sns.heatmap(
            heatmap_data, 
            annot=labels_matrix, 
            fmt="",  
            cmap='viridis',  
            vmax=vmax_cutoff,
            linewidths=0.6, 
            linecolor='#3b528b', 
            cbar_kws={'label': cbar_label},
            ax=ax,
            annot_kws={
                'size': annot_size,
                'weight': 'bold'
            }
        )

        title_suffix = ' (MSE)' if is_mse else ' (Błąd $L_2$)'
        #ax.set_title(f'Globalna mapa wrażliwości FunDPS ({noise_name} Noise){title_suffix}', pad=16, fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Siła nawigacji gradientowej ($\zeta$)', fontsize=10, labelpad=12)
        ax.set_xlabel('Liczba kroków rekonstrukcji ($N_{\\mathrm{steps}}$)', fontsize=10, labelpad=12)
        ax.tick_params(axis='both', labelsize=9)
        
        plt.subplots_adjust(left=0.18, right=0.82, bottom=0.18, top=0.82)

        os.makedirs(save_dir, exist_ok=True)
        file_suffix = f"_{metric_name.lower()}"
        save_path = os.path.join(save_dir, f"global_heatmap_fundps_{noise_name.lower()}{file_suffix}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()


def plot_fundps_ablation_heatmap(results_dict, func_name, noise_name, save_dir='../images/experiment3'):

    data = results_dict.get(func_name)
    if data is None: 
        print(f"[UWAGA] Brak danych w słowniku dla funkcji: {func_name}")
        return
    
    df_metrics = pd.DataFrame(data['metrics_history'])
    heatmap_data = df_metrics.pivot(index='Zeta', columns='Steps', values='L2_Error')
    
    labels_matrix = np.zeros_like(heatmap_data.values, dtype=object)
    for r in range(heatmap_data.shape[0]):
        for c in range(heatmap_data.shape[1]):
            val = heatmap_data.values[r, c]
            if pd.isna(val) or np.isinf(val):
                labels_matrix[r, c] = "NaN"
            elif val >= 1000.0:
                labels_matrix[r, c] = f"{val:.0e}"  
            else:
                labels_matrix[r, c] = f"{val:.1f}"  
    
    heatmap_clean = heatmap_data.replace([np.inf, -np.inf], np.nan)
    global_max_value = heatmap_clean.max().max()
    vmax_cutoff = min(global_max_value, 150.0) if pd.notna(global_max_value) else 100.0
    

    with plt.rc_context({'figure.autolayout': False}):
        fig, ax = plt.subplots(figsize=(18 / 2.54, 13.5 / 2.54))
        n_cols = heatmap_data.shape[1]

        if n_cols <= 4:
            annot_size = 10.5
        elif n_cols <= 6:
            annot_size = 9.5
        else:
            annot_size = 8.5

        sns.heatmap(
            heatmap_data, 
            annot=labels_matrix, 
            fmt="",               
            cmap='viridis', 
            vmax=vmax_cutoff, 
            cbar_kws={'label': 'Błąd relatywny $L_2$ (%)'},
            linewidths=0.6, 
            linecolor='#3b528b',
            ax=ax,
            annot_kws={
                'size': annot_size,
                'weight': 'bold'  
            }
        )
        
        #ax.set_title(f'Wrażliwość FunDPS ({noise_name} Noise) | Funkcja: {func_name.upper()}',                      pad=18, fontsize=12, fontweight='bold')
        ax.set_ylabel('Siła nawigacji gradientowej ($\zeta$)', fontsize=11, labelpad=14)
        ax.set_xlabel('Liczba kroków rekonstrukcji ($N_{\\mathrm{steps}}$)', fontsize=11, labelpad=14)
        ax.tick_params(axis='both', labelsize=10)
        
        plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85)
        
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"heatmap_ablation_{noise_name.lower()}_{func_name}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()



def generate_fundps_summary_table(results_dict, noise_name):

    all_trials_data = []
    for func_name, func_data in results_dict.items():
        if func_data is None or 'metrics_history' not in func_data:
            continue
        df_func = pd.DataFrame(func_data['metrics_history'])
        if not df_func.empty:
            all_trials_data.append(df_func)

    if not all_trials_data:
        print(f"[BŁĄD] Brak danych dla szumu {noise_name}.")
        return None

    df_global = pd.concat(all_trials_data, ignore_index=True)
    
    df_averaged = df_global.groupby(['Steps', 'Zeta']).agg(
        Sredni_Blad_L2_Proc=('L2_Error', 'mean'),
        Sredni_Czas_s=('Total_Time_s', 'mean')
    ).reset_index()

    df_averaged.columns = ['Kroki (N_steps)', 'Siła nawigacji (Zeta)', 'Średni globalny błąd L2 (%)', 'Średni łączny czas [s]']
    
    df_summary = df_averaged.sort_values(by='Średni globalny błąd L2 (%)').reset_index(drop=True)

    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    print(df_summary.to_string(index=True))
    print("="*85)
    
    return df_summary



def plot_parameter_matrix_zoom(results_dict, noise_name, save_dir='../images/experiment3'):

    all_trials_data = []
    for func_name, func_data in results_dict.items():
        if func_data is None or 'metrics_history' not in func_data:
            continue
        df_func = pd.DataFrame(func_data['metrics_history'])
        if not df_func.empty:
            all_trials_data.append(df_func)

    if not all_trials_data:
        print(f"[BŁĄD] Brak danych dla szumu {noise_name}.")
        return

    df_global = pd.concat(all_trials_data, ignore_index=True)
    
    allowed_steps = [10, 20, 50, 100, 200]
    allowed_zetas = [2.0, 4.0, 6.0, 8.0, 10.0]
    
    df_filtered = df_global[
        df_global['Steps'].isin(allowed_steps) & 
        df_global['Zeta'].isin(allowed_zetas)
    ].copy()

    df_averaged = df_filtered.groupby(['Zeta', 'Steps'])['L2_Error'].mean().reset_index()
    matrix_data = df_averaged.pivot(index='Zeta', columns='Steps', values='L2_Error')
    matrix_data = matrix_data.sort_index(ascending=False)

    labels_matrix = np.zeros_like(matrix_data.values, dtype=object)
    for r in range(matrix_data.shape[0]):
        for c in range(matrix_data.shape[1]):
            val = matrix_data.values[r, c]
        
            if pd.isna(val):
	            labels_matrix[r, c] = "NaN"
	            
            elif val >= 1000.0:
	            labels_matrix[r, c] = f"{val:.1e}%".replace("+0", "").replace("+", "")
	            
            else:
	            labels_matrix[r, c] = f"{val:.2f}%"

    fig, ax = plt.subplots(figsize=(11 / 2.54, 9.5 / 2.54)) 
    
    sns.heatmap(
        matrix_data, annot=labels_matrix, fmt="", cmap="viridis",           
        linewidths=1.0, linecolor='#333333',
        cbar_kws={'label': 'Średni globalny błąd relatywny $L_2$ (%)'},
        ax=ax, annot_kws={'size': 8, 'weight': 'bold'}
    )
    
    #ax.set_title(f'Szczegółowa macierz optymalna 3$\\times$3\n(Prior: {noise_name.upper()} Noise)', pad=16)
    ax.set_ylabel('Siła nawigacji gradientowej ($\zeta$)', labelpad=12)
    ax.set_xlabel('Liczba kroków próbkowania ($N_{\\mathrm{steps}}$)', labelpad=12)
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"param_matrix_filtered_{noise_name.lower()}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()



def plot_global_noise_comparison_boxplot(res_white, res_grf, save_path="../images/experiment3/global_noise_boxplot.png"):
    all_l2_white = []
    all_l2_grf = []
    
    for func in res_white.keys():
        if res_white[func]['best_metrics'] is not None and res_grf[func]['best_metrics'] is not None:
            all_l2_white.append(res_white[func]['best_metrics']['L2_Error'])
            all_l2_grf.append(res_grf[func]['best_metrics']['L2_Error'])
            
    plot_df = pd.DataFrame({
        'Błąd L2 (%)': all_l2_white + all_l2_grf,
        'Struktura szumu priora': ['Biały szum (White)'] * len(all_l2_white) + ['Gładki szum (GRF)'] * len(all_l2_grf)
    })
    
    plt.figure()
    sns.boxplot(data=plot_df, x='Struktura szumu priora', y='Błąd L2 (%)', 
                hue='Struktura szumu priora', palette='viridis', legend=False, edgecolor='black', linewidth=1.0)
    
    #plt.title('Globalna podatność na strukturę szumu w algorytmie FunDPS')
    plt.ylabel('Najlepszy błąd relatywny L2 (%)')
    plt.xlabel('')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_fundps_time_vs_quality(results_dict, func_name, noise_name, save_dir='../images/experiment3'):
    data = results_dict.get(func_name)
    if data is None: return
    
    df_metrics = pd.DataFrame(data['metrics_history'])
    df_metrics = df_metrics[df_metrics['L2_Error'] < 200.0]
    
    plt.figure()
    scatter = plt.scatter(df_metrics['Total_Time_s'], df_metrics['L2_Error'], 
                          c=df_metrics['Steps'], cmap='viridis', s=df_metrics['Zeta']*10, 
                          alpha=0.8, edgecolors='black', linewidths=0.5)
    
    #plt.title(f'Optymalizacja Pareto (Czas vs Jakość) | FunDPS {noise_name}\nFunkcja: {func_name.upper()}')
    plt.xlabel('Całkowity czas operacji (Uczenie + Próbkowanie) [s]')
    plt.ylabel('Błąd relatywny L2 (%)')
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Liczba kroków próbkowania (Steps)')
    
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(f"{save_dir}/pareto_{noise_name.lower()}_{func_name}.png", bbox_inches='tight')
    plt.show()

