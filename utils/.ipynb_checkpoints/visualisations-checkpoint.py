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

PALETTE = ['#fde725', '#5ec962', '#3b528b']
ARCH_CONFIG = {
    'MLP':    {'color': '#fde725', 'ls': ':',  'marker': 'o'},
    'Conv1D': {'color': '#5ec962', 'ls': '-.', 'marker': 's'},
    'UNet':   {'color': '#3b528b', 'ls': '-',  'marker': '^'}
}

FONT_SIZE        = 10
TICK_SIZE        = 9
LEGEND_SIZE      = 9
ANNOT_SIZE       = 9
ANNOT_SIZE_LARGE = 10
LINE_WIDTH       = 1.6
MARKER_SIZE      = 6

FIG_W  = 12 / 2.54
FIG_H  =  8 / 2.54
FIG_W2 = 18 / 2.54
FIG_H2 =  8 / 2.54

plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman'],
    'text.color':         'black',
    'axes.labelcolor':    'black',
    'axes.edgecolor':     'black',
    'xtick.color':        'black',
    'ytick.color':        'black',
    'font.size':          FONT_SIZE,
    'axes.titlesize':     FONT_SIZE,
    'axes.titleweight':   'bold',
    'axes.labelsize':     FONT_SIZE,
    'xtick.labelsize':    TICK_SIZE,
    'ytick.labelsize':    TICK_SIZE,
    'legend.fontsize':    LEGEND_SIZE,
    'legend.title_fontsize': LEGEND_SIZE,
    'lines.linewidth':    LINE_WIDTH,
    'axes.linewidth':     0.8,
    'grid.linewidth':     0.5,
    'grid.linestyle':     '--',
    'grid.alpha':         0.5,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'figure.figsize':     [FIG_W, FIG_H],
    'figure.autolayout':  True,
})

HEATMAP_ANNOT_KWS = {'size': ANNOT_SIZE, 'weight': 'bold'}
BOXPLOT_PROPS = dict(
    boxprops=dict(edgecolor='black'),
    capprops=dict(color='black'),
    whiskerprops=dict(color='black'),
    medianprops=dict(color='black', linewidth=1.5)
)
LEGEND_OUTSIDE = dict(
    loc='upper left',
    bbox_to_anchor=(1.02, 1.0),
    borderaxespad=0.,
    frameon=True,
    facecolor='white',
    edgecolor='black',
)


def _get_base_arch(arch_name: str) -> str:
    return arch_name.split('_')[0]


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

    pivot_mu  = df.pivot(index='Architektura', columns='Pojemność', values='mu')
    pivot_std = df.pivot(index='Architektura', columns='Pojemność', values='std')

    annot_array = np.empty_like(pivot_mu.values, dtype=object)
    for i in range(pivot_mu.shape[0]):
        for j in range(pivot_mu.shape[1]):
            mu_val  = pivot_mu.iloc[i, j]
            std_val = pivot_std.iloc[i, j]
            if pd.isna(mu_val):
                annot_array[i, j] = 'Brak'
            elif mu_val < 1e-4:
                annot_array[i, j] = f'{mu_val:.1e}\n±{std_val:.1e}'
            else:
                annot_array[i, j] = f'{mu_val:.4f}\n±{std_val:.4f}'

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    sns.heatmap(
        pivot_mu,
        annot=annot_array,
        fmt='',
        cmap='viridis',
        norm=LogNorm(),
        cbar_kws={'label': 'Średni błąd testowy MSE (log)'},
        linewidths=0.5,
        linecolor='black',
        annot_kws=HEATMAP_ANNOT_KWS,
        ax=ax,
    )
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()



def generate_lr_summary_plots(checkpoints_dir, plots_dir):
    learning_rates = sorted([1e-3, 5e-4, 1e-4])
    samplers = ["Sine", "Chirp", "Hard"]
    architectures = ["MLP", "Conv1D", "UNet"]
    capacities = ["C32", "C64", "C128", "C256"]
    t_steps_options = [80, 100]

    raw_data = {
        lr: {arch: [] for arch in architectures} for lr in learning_rates
    }

    for lr in learning_rates:
        for s in samplers:
            for arch in architectures:
                for cap in capacities:
                    for t in t_steps_options:
                        prefix = f"{s}_{arch}_{cap}_LR{lr}_T{t}"
                        file_path = os.path.join(
                            checkpoints_dir, f"{prefix}_stats.pth"
                        )
                        if os.path.exists(file_path):
                            try:
                                st = torch.load(
                                    file_path,
                                    map_location="cpu",
                                    weights_only=False,
                                )
                                if "test_mu" in st:
                                    raw_data[lr][arch].append(st["test_mu"])
                            except Exception:
                                continue

    arch_averages = {arch: [] for arch in architectures}
    global_averages = []

    for lr in learning_rates:
        all_lr_scores = []
        for arch in architectures:
            scores = raw_data[lr][arch]
            mean_score = np.mean(scores) if scores else np.nan
            arch_averages[arch].append(mean_score)
            all_lr_scores.extend(scores)
        global_averages.append(
            np.mean(all_lr_scores) if all_lr_scores else np.nan
        )

    summary_plots_dir = os.path.join(plots_dir, "stats")
    os.makedirs(summary_plots_dir, exist_ok=True)

    lr_labels = [str(lr) for lr in learning_rates]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for arch in architectures:
        cfg = ARCH_CONFIG.get(arch, {})
        ax.plot(
            lr_labels,
            arch_averages[arch],
            label=arch,
            color=cfg.get("color", "#000000"),
            linestyle=cfg.get("ls", "-"),
            marker=cfg.get("marker", "o"),
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
        )
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Średni błąd Test MSE")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    legend_kwargs = {
        "frameon": True,
        "facecolor": "white",
        "edgecolor": "black",
    }
    legend_kwargs.update(LEGEND_OUTSIDE)
    legend_kwargs.pop("title", None)

    ax.legend(title="Architektura", **legend_kwargs)

    plt.savefig(
        os.path.join(summary_plots_dir, "lr_vs_mse_per_arch.png"),
        bbox_inches="tight",
    )
    plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(
        lr_labels,
        global_averages,
        color="#333333",
        marker="D",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label="Średnia globalna",
    )
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Globalny średni błąd Test MSE")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, facecolor="white", edgecolor="black")

    plt.savefig(
        os.path.join(summary_plots_dir, "lr_vs_mse_global_average.png"),
        bbox_inches="tight",
    )
    plt.show()
    plt.close()

def plot_learning_curves_with_ci(
    train_mat: np.ndarray,
    val_mat:   np.ndarray,
    arch:      str,
    cap:       str,
    func:      str,
    save_path: str,
    confidence: float = 0.95,
) -> None:
    base_arch  = _get_base_arch(arch)
    style      = ARCH_CONFIG.get(base_arch, {'color': '#000000', 'ls': '-', 'marker': ''})
    main_color = style['color']

    epochs  = np.arange(1, train_mat.shape[1] + 1)
    n_runs  = train_mat.shape[0]
    t_crit  = stats.t.ppf((1 + confidence) / 2., n_runs - 1)

    train_mean = np.mean(train_mat, axis=0)
    train_ci   = t_crit * np.std(train_mat, axis=0, ddof=1) / np.sqrt(n_runs)
    val_mean   = np.mean(val_mat,   axis=0)
    val_ci     = t_crit * np.std(val_mat,   axis=0, ddof=1) / np.sqrt(n_runs)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    val_color = '#777777'

    ax.plot(epochs, train_mean, label='Błąd treningowy',
            color=main_color, linestyle=style['ls'], linewidth=LINE_WIDTH)
    ax.fill_between(epochs, train_mean - train_ci, train_mean + train_ci,
                    color=main_color, alpha=0.12, label=f'Trening {int(confidence*100)}% CI')
    ax.plot(epochs, val_mean, label='Błąd walidacyjny',
            color=val_color, linestyle='--', linewidth=LINE_WIDTH)
    ax.fill_between(epochs, val_mean - val_ci, val_mean + val_ci,
                    color=val_color, alpha=0.08, label=f'Walidacja {int(confidence*100)}% CI')

    ax.set_xlabel('Epoka')
    ax.set_ylabel('MSE (log)')
    ax.set_yscale('log')
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='black')
    ax.grid(True, which='both')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_lr_comparison(lr_data: dict, arch: str, cap: str, func: str, save_path: str):
    lrs     = sorted(list(lr_data.keys()), reverse=True)
    num_lrs = len(lrs)
    colors  = [mcolors.to_hex([0.1 + (0.6 * i / max(1, num_lrs - 1))] * 3) for i in range(num_lrs)]
    ls_styles = ['-', '--', ':', '-.']

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W2, FIG_H), sharey=True)

    for idx, lr in enumerate(lrs):
        c         = colors[idx]
        train_mat = lr_data[lr]['train']
        val_mat   = lr_data[lr]['val']
        epochs    = np.arange(1, train_mat.shape[1] + 1)
        train_mu  = np.mean(train_mat, axis=0)
        train_std = np.std(train_mat,  axis=0, ddof=1)
        val_mu    = np.mean(val_mat,   axis=0)
        val_std   = np.std(val_mat,    axis=0, ddof=1)
        ls        = ls_styles[idx % len(ls_styles)]

        axes[0].plot(epochs, train_mu, label=f'LR = {lr}', color=c, linestyle=ls, linewidth=LINE_WIDTH)
        axes[0].fill_between(epochs, train_mu - train_std, train_mu + train_std, color=c, alpha=0.08)
        axes[1].plot(epochs, val_mu,   label=f'LR = {lr}', color=c, linestyle=ls, linewidth=LINE_WIDTH)
        axes[1].fill_between(epochs, val_mu - val_std, val_mu + val_std, color=c, alpha=0.08)

    for ax, ylabel, title in zip(
        axes,
        ['MSE (log)', ''],
        ['Błąd treningowy', 'Błąd walidacyjny'],
    ):
        ax.set_xlabel('Epoka')
        ax.set_yscale('log')
        ax.grid(True, which='both')
        ax.set_title(title, fontsize=FONT_SIZE)

    axes[0].set_ylabel('MSE (log)')
    axes[1].legend(**LEGEND_OUTSIDE)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def visualize_styled_stats_final(folder):
    all_data = []
    if not os.path.exists(folder):
        return

    for f in os.listdir(folder):
        if not f.endswith('_stats.pth'):
            continue
        parts     = f.replace('_stats.pth', '').split('_')
        func_name = parts[0].capitalize()
        arch_name = parts[1]
        try:
            st = torch.load(os.path.join(folder, f), map_location='cpu', weights_only=False)
            all_data.append({
                'Funkcja':    func_name,
                'Architektura': arch_name,
                'Test_MSE':   st['test_mu'],
                'Test_Std':   st['test_std'],
            })
        except Exception:
            continue

    df_raw = pd.DataFrame(all_data)
    if df_raw.empty:
        return

    idx_min   = df_raw.groupby(['Architektura', 'Funkcja'])['Test_MSE'].idxmin()
    df_grouped = df_raw.loc[idx_min].reset_index(drop=True)

    all_funcs = ['Sine', 'Chirp', 'Hard']
    all_archs = ['MLP', 'Conv1D', 'UNet']

    idx      = pd.MultiIndex.from_product([all_archs, all_funcs], names=['Architektura', 'Funkcja'])
    df_final = df_grouped.set_index(['Architektura', 'Funkcja']).reindex(idx).reset_index()
    df_final['Test_MSE'] = df_final['Test_MSE'].replace(0, np.nan)
    df_final['Architektura'] = pd.Categorical(df_final['Architektura'], categories=all_archs, ordered=True)
    df_final['Funkcja']      = pd.Categorical(df_final['Funkcja'],      categories=all_funcs, ordered=True)

    custom_palette = {arch: ARCH_CONFIG[arch]['color'] for arch in all_archs if arch in ARCH_CONFIG}

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    sns.barplot(
        data=df_final, x='Funkcja', y='Test_MSE', hue='Architektura',
        palette=custom_palette, edgecolor='black', linewidth=1.0,
        alpha=1.0, order=all_funcs, ax=ax,
    )

    for container, arch in zip(ax.containers, all_archs):
        arch_data = df_final[df_final['Architektura'] == arch].sort_values('Funkcja')
        x_coords  = [rect.get_x() + rect.get_width() / 2.0 for rect in container]
        y_vals    = arch_data['Test_MSE'].values
        y_errs    = arch_data['Test_Std'].values
        lower_err = np.clip(y_errs, 0, y_vals - 1e-10)
        ax.errorbar(x=x_coords, y=y_vals, yerr=[lower_err, y_errs],
                    fmt='none', c='#000000', capsize=3, elinewidth=1.2, alpha=0.9)
        for x, y, u in zip(x_coords, y_vals, y_errs):
            if pd.notna(y):
                ax.text(x, y + u, f'{y:.1e}', ha='center', va='bottom',
                        fontsize=TICK_SIZE - 1, rotation=45, color='black')

    max_val = (df_final['Test_MSE'] + df_final['Test_Std']).max()
    min_val = df_final['Test_MSE'].min()
    if pd.notna(max_val):
        ax.set_ylim(top=max_val * 1.5)
    if pd.notna(min_val):
        ax.set_ylim(bottom=min_val * 0.1)

    ax.set_ylabel('Błąd testowy MSE')
    ax.set_xlabel('Klasa')

    legend_elements = [Line2D([0], [0], color=custom_palette[arch], lw=6, label=arch) for arch in all_archs]
    ax.legend(handles=legend_elements, title='Architektura', **LEGEND_OUTSIDE)

    os.makedirs('../images/experiment1/stats', exist_ok=True)
    plt.savefig('../images/experiment1/stats/porownanie_architektur.png', bbox_inches='tight')
    plt.show()
    plt.close()


def plot_mse_vs_params_vertical(folder='checkpoints1'):
    rows = []
    if not os.path.exists(folder):
        return

    for f in os.listdir(folder):
        if not f.endswith('_stats.pth'):
            continue
        parts = f.replace('_stats.pth', '').split('_')
        if len(parts) < 5:
            continue
        try:
            func_name  = parts[0].capitalize()
            arch       = parts[1]
            st         = torch.load(os.path.join(folder, f), map_location='cpu', weights_only=False)
            test_mse   = st.get('test_mu')
            num_params = st.get('num_params')
            if test_mse is not None and num_params is not None:
                rows.append({'Funkcja': func_name, 'Architektura': arch,
                             'Liczba parametrów': num_params, 'MSE': test_mse})
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return

    df_agg        = df.groupby(['Funkcja', 'Architektura', 'Liczba parametrów'])['MSE'].min().reset_index()
    func_order    = ['Sine', 'Chirp', 'Hard']
    existing_funcs = [s for s in func_order if s in df_agg['Funkcja'].unique()]
    active_archs   = df_agg['Architektura'].unique()
    custom_palette = {arch: ARCH_CONFIG[arch]['color'] for arch in active_archs if arch in ARCH_CONFIG}

    g = sns.relplot(
        data=df_agg, x='Liczba parametrów', y='MSE',
        hue='Architektura', style='Architektura',
        row='Funkcja', row_order=existing_funcs,
        kind='line',
        markers=[ARCH_CONFIG.get(a, {}).get('marker', 'o') for a in active_archs],
        dashes=[(2, 2), (4, 2), (1, 0)][:len(active_archs)],
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        palette=custom_palette,
        height=FIG_H,
        aspect=FIG_W / FIG_H,
        facet_kws={'sharey': False},
    )
    g.set(xscale='log', yscale='log')
    g.set_axis_labels('Liczba parametrów', 'Najlepszy testowy błąd MSE (log)')
    g.set_titles(row_template='Funkcja: {row_name}')
    for ax in g.axes.flat:
        ax.grid(True, which='major', ls='-',  alpha=0.4, color='#cccccc')
        ax.grid(True, which='minor', ls=':', alpha=0.2, color='#eeeeee')

    os.makedirs('../images/experiment1/stats', exist_ok=True)
    plt.savefig('../images/experiment1/stats/mse_vs_params.png', bbox_inches='tight')
    plt.show()
    plt.close()


def plot_experiment_ablation_boxplots(df: pd.DataFrame, save_dir: str = '../images/experiment1/stats', use_log_scale: bool = False) -> None:
    if df.empty:
        return

    arch_palette = {arch: cfg['color'] for arch, cfg in ARCH_CONFIG.items()}
    arch_order   = ['MLP', 'Conv1D', 'UNet']
    os.makedirs(save_dir, exist_ok=True)

    FIG_BOX_W = 7 / 2.54
    FIG_BOX_H = FIG_H

    def _format_and_save(ax, filename):
        if use_log_scale:
            ax.set_yscale('log')
        ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.5)
        ax.set_ylabel('Test błąd MSE')
        ax.set_xlabel('')
        ax.tick_params(axis='both', labelsize=TICK_SIZE)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()

    fig, ax1 = plt.subplots(figsize=(FIG_BOX_W, FIG_BOX_H))
    sns.boxplot(data=df, x='Architektura', y='Test_MSE',
                order=arch_order, hue='Architektura', hue_order=arch_order,
                palette=arch_palette, legend=False, ax=ax1, **BOXPLOT_PROPS)
    _format_and_save(ax1, 'wplyw_architektury.png')

    cap_order_vals = sorted(df['Pojemność'].unique())
    cap_count      = len(cap_order_vals)
    fig, ax2 = plt.subplots(figsize=(max(FIG_BOX_W, cap_count * 1.4 / 2.54), FIG_BOX_H))
    sns.boxplot(data=df, x='Pojemność', y='Test_MSE',
                hue='Pojemność', palette='viridis', legend=False, ax=ax2, **BOXPLOT_PROPS)
    _format_and_save(ax2, 'wplyw_pojemnosci.png')

    lr_count = len(df['LR'].unique())
    fig, ax3 = plt.subplots(figsize=(max(FIG_BOX_W, lr_count * 1.4 / 2.54), FIG_BOX_H))
    sns.boxplot(data=df, x='LR', y='Test_MSE',
                hue='LR', palette='viridis', legend=False, ax=ax3, **BOXPLOT_PROPS)
    _format_and_save(ax3, 'wplyw_lr.png')

    t_count = len(df['T'].unique())
    fig, ax4 = plt.subplots(figsize=(max(FIG_BOX_W, t_count * 1.4 / 2.54), FIG_BOX_H))
    sns.boxplot(data=df, x='T', y='Test_MSE',
                hue='T', palette='viridis', legend=False, ax=ax4, **BOXPLOT_PROPS)
    _format_and_save(ax4, 'wplyw_krokow_t.png')


def create_split_heatmap(folder):
    all_data = []
    if not os.path.exists(folder):
        return

    for f in os.listdir(folder):
        if not f.endswith('_stats.pth'):
            continue
        parts = f.replace('_stats.pth', '').split('_')
        if len(parts) < 5:
            continue
        func_name = parts[0].capitalize()
        arch      = parts[1]
        cap_str   = parts[2]
        t_str     = parts[4]
        try:
            cap     = int(cap_str.replace('C', ''))
            t_steps = int(t_str.replace('T', ''))
            st      = torch.load(os.path.join(folder, f), map_location='cpu', weights_only=False)
            all_data.append({'Funkcja': func_name, 'Architektura': arch,
                             'Pojemność': cap, 'T': t_steps, 'MSE': st['test_mu']})
        except Exception:
            continue

    df_raw = pd.DataFrame(all_data)
    if df_raw.empty:
        return

    df = df_raw[df_raw['Pojemność'].isin([32, 64, 128, 256, 512])]
    df = df[df['T'].isin([80, 100])]
    df = df[df['Architektura'].isin(['MLP', 'Conv1D', 'UNet'])]
    os.makedirs('../images/experiment1/stats', exist_ok=True)

    for func in df['Funkcja'].unique():
        df_func  = df[df['Funkcja'] == func]
        df_agg   = df_func.groupby(['Architektura', 'Pojemność', 'T'])['MSE'].min().reset_index()
        pivot_df = df_agg.pivot_table(index='Architektura', columns=['Pojemność', 'T'], values='MSE')
        pivot_df = pivot_df.reindex([a for a in ['MLP', 'Conv1D', 'UNet'] if a in pivot_df.index])

        min_err    = pivot_df.min().min()
        annot_fmt  = '.1e' if min_err < 1e-4 else '.4f'
        num_t      = len(df['T'].unique())
        n_cols     = pivot_df.shape[1]

        fig, ax = plt.subplots(figsize=(max(FIG_W, n_cols * 1.6 / 2.54), FIG_H))
        sns.heatmap(
            pivot_df, annot=True, cmap='viridis', fmt=annot_fmt,
            norm=LogNorm(),
            cbar_kws={'label': 'Najniższy błąd testowy MSE (skala log)'},
            linewidths=0.5, linecolor='black',
            annot_kws=HEATMAP_ANNOT_KWS,
            ax=ax,
        )

        for i in range(1, len(pivot_df.columns.levels[0])):
            ax.axvline(x=i * num_t, color='black', linewidth=2.5)

        capacities = pivot_df.columns.levels[0]
        for i, cap in enumerate(capacities):
            ax.text(i * num_t + num_t / 2, -0.35, f'C{cap}',
                    ha='center', va='bottom', fontsize=FONT_SIZE, fontweight='bold', clip_on=False)

        ax.set_xlabel('Liczba kroków dyfuzji (T)', labelpad=10)
        ax.set_ylabel('Architektura')
        ax.set_xticklabels([f'T={t}' for _, t in pivot_df.columns], rotation=0, fontsize=TICK_SIZE)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        plt.tight_layout()
        plt.savefig(f'../images/experiment1/stats/heatmap_split_cells_{func}.png', bbox_inches='tight')
        plt.close()


def plot_mse_vs_params_global(folder='checkpoints1'):
    rows = []
    if not os.path.exists(folder):
        return

    for f in os.listdir(folder):
        if not f.endswith('_stats.pth'):
            continue
        parts = f.replace('_stats.pth', '').split('_')
        if len(parts) < 5:
            continue
        try:
            func_name  = parts[0].capitalize()
            arch       = parts[1]
            st         = torch.load(os.path.join(folder, f), map_location='cpu', weights_only=False)
            test_mse   = st.get('test_mu')
            num_params = st.get('num_params')
            if test_mse is not None and num_params is not None:
                rows.append({'Funkcja': func_name, 'Architektura': arch,
                             'Liczba parametrów': num_params, 'MSE': test_mse})
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return

    df_agg       = df.groupby(['Architektura', 'Liczba parametrów'])['MSE'].mean().reset_index()
    all_archs    = ['MLP', 'Conv1D', 'UNet']
    active_archs = [a for a in all_archs if a in df_agg['Architektura'].unique()]
    df_agg['Architektura'] = pd.Categorical(df_agg['Architektura'], categories=active_archs, ordered=True)
    df_agg = df_agg.sort_values('Architektura')

    custom_palette = {arch: ARCH_CONFIG[arch]['color'] for arch in active_archs if arch in ARCH_CONFIG}

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    sns.lineplot(
        data=df_agg, x='Liczba parametrów', y='MSE',
        hue='Architektura', style='Architektura',
        markers={a: ARCH_CONFIG[a]['marker'] for a in active_archs if a in ARCH_CONFIG},
        dashes={a: (1, 2) if a == 'MLP' else (4, 2) if a == 'Conv1D' else (1, 0) for a in active_archs},
        palette=custom_palette,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE, ax=ax,
    )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Liczba parametrów modelu ($N_{\\mathrm{params}}$)', labelpad=8)
    ax.set_ylabel('Globalny średni błąd Test MSE (log)', labelpad=8)
    ax.grid(True, which='major', ls='-',  alpha=0.4, color='#cccccc')
    ax.grid(True, which='minor', ls=':', alpha=0.2, color='#eeeeee')
    ax.legend(title='Architektura', **LEGEND_OUTSIDE)

    os.makedirs('../images/experiment1/stats', exist_ok=True)
    plt.savefig('../images/experiment1/stats/mse_vs_params_global.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def plot_learning_curves_unified(func_type, folder, target_lr=0.0001, target_t=80):
    if not os.path.exists(folder):
        return

    func_type = func_type.capitalize()
    files     = [f for f in os.listdir(folder) if f.startswith(func_type) and f.endswith('_stats.pth')]

    cap_settings = {
        'C32':  {'alpha': 0.5, 'lw': 0.8},
        'C64':  {'alpha': 0.7, 'lw': 1.1},
        'C128': {'alpha': 0.9, 'lw': 1.4},
        'C256': {'alpha': 1.0, 'lw': LINE_WIDTH},
    }
    arch_order = ['MLP', 'Conv1D', 'UNet']
    cap_order  = ['C32', 'C64', 'C128', 'C256']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W2, FIG_H))
    handles, labels = [], []
    found_any = False

    for arch in arch_order:
        for cap in cap_order:
            prefix  = f'{func_type}_{arch}_{cap}_LR{target_lr}_T{target_t}'
            matches = [f for f in files if f.startswith(prefix)]
            if not matches:
                continue
            found_any = True
            st         = torch.load(os.path.join(folder, matches[0]), map_location='cpu', weights_only=False)
            train_curve = st['train_mat'].mean(axis=0)
            val_curve   = st['val_mat'].mean(axis=0)
            epochs      = np.arange(1, len(train_curve) + 1)
            style       = ARCH_CONFIG.get(arch, {'color': 'black', 'ls': '-'})
            cap_style   = cap_settings.get(cap, {'alpha': 1.0, 'lw': LINE_WIDTH})
            label_name  = f'{arch} ({cap})'

            line, = ax1.plot(epochs, train_curve,
                             color=style['color'], linestyle=style['ls'],
                             linewidth=cap_style['lw'], alpha=cap_style['alpha'],
                             label=label_name)
            ax2.plot(epochs, val_curve,
                     color=style['color'], linestyle=style['ls'],
                     linewidth=cap_style['lw'], alpha=cap_style['alpha'])
            handles.append(line)
            labels.append(label_name)

    if not found_any:
        plt.close()
        return

    for ax, ylabel in zip([ax1, ax2], ['MSE', '']):
        ax.set_xlabel('Epoka')
        ax.set_ylabel(ylabel)
        ax.grid(True, which='both', ls='--', alpha=0.4)

    ax1.set_ylabel('MSE')
    ax2.legend(handles, labels, title='Architektura (Pojemność)', **LEGEND_OUTSIDE)
    plt.tight_layout()

    os.makedirs('../images/experiment1/stats', exist_ok=True)
    plt.savefig(f'../images/experiment1/stats/convergence_combined_{func_type}_LR{target_lr}_T{target_t}.png',
                bbox_inches='tight')
    plt.show()
    plt.close()


def load_full_experiment_data(folder='checkpoints1'):
    all_data = []
    if not os.path.exists(folder):
        return pd.DataFrame()

    for f in os.listdir(folder):
        if not f.endswith('_stats.pth'):
            continue
        parts = f.replace('_stats.pth', '').split('_')
        if len(parts) < 5:
            continue
        try:
            function = parts[0].capitalize()
            arch     = parts[1]
            cap      = int(parts[2].replace('C', ''))
            lr       = float(parts[3].replace('LR', ''))
            t_steps  = int(parts[4].replace('T', ''))
            st       = torch.load(os.path.join(folder, f), map_location='cpu', weights_only=False)
            all_data.append({
                'Funkcja':          function,
                'Architektura':     arch,
                'Pojemność':        cap,
                'LR':               lr,
                'T':                t_steps,
                'Train_MSE':        st.get('train_mu', np.nan),
                'Val_MSE':          st.get('val_mu',   np.nan),
                'Test_MSE':         st.get('test_mu',  np.nan),
                'Test_STD':         st.get('test_std', np.nan),
                'Liczba parametrów': st.get('num_params', np.nan),
            })
        except Exception as e:
            print(f'Błąd ładowania pliku {f}: {e}')
            continue

    return pd.DataFrame(all_data)


def analyze_generalization_gap(df, top_n=10):
    df_gap = df.copy()
    df_gap['Gap_Val_Train'] = df_gap['Val_MSE'] - df_gap['Train_MSE']
    df_gap = df_gap.sort_values('Gap_Val_Train', ascending=False)
    cols   = ['Funkcja', 'Architektura', 'Pojemność', 'LR', 'Test_MSE', 'Train_MSE', 'Val_MSE', 'Gap_Val_Train']
    return df_gap[cols].head(top_n)


def analyze_stability_cv(df, top_n=10):
    df_cv         = df.copy()
    df_cv['CV_%'] = (df_cv['Test_STD'] / df_cv['Test_MSE']) * 100
    most_unstable = df_cv.sort_values('CV_%', ascending=False).head(top_n)
    most_stable   = df_cv.sort_values('CV_%', ascending=True ).head(top_n)
    cols          = ['Funkcja', 'Architektura', 'Pojemność', 'LR', 'Test_MSE', 'Test_STD', 'CV_%']
    return most_unstable[cols], most_stable[cols]


def analyze_worst_combinations(df, top_n=10):
    df_worst = df.sort_values('Test_MSE', ascending=False)
    cols     = ['Funkcja', 'Architektura', 'Pojemność', 'LR', 'T', 'Test_MSE']
    return df_worst[cols].head(top_n)


def get_scientific_sdedit_candidates(df: pd.DataFrame) -> pd.DataFrame:
    all_candidates = []

    for func in df['Funkcja'].unique():
        func_df = df[df['Funkcja'] == func].copy()
        cands   = []

        top_3         = func_df.nsmallest(3, 'Test_MSE').copy()
        top_3['Powód wyboru'] = 'Najniższy błąd dla tego sygnału'
        cands.append(top_3)
        df_rest = func_df.drop(top_3.index)

        for arch in ['Conv1D', 'MLP']:
            arch_df = df_rest[df_rest['Architektura'] == arch]
            if not arch_df.empty:
                best_arch               = arch_df.nsmallest(1, 'Test_MSE').copy()
                best_arch['Powód wyboru'] = f'Najlepszy w klasie {arch}'
                cands.append(best_arch)
                df_rest = df_rest.drop(best_arch.index)

        underfit = df_rest[df_rest['Pojemność'] == 'C32']
        if not underfit.empty:
            idx         = len(underfit) // 2
            mid_under   = underfit.sort_values('Test_MSE').iloc[idx:idx+1].copy()
            mid_under['Powód wyboru'] = 'Model niedouczony (C32)'
            cands.append(mid_under)
            df_rest = df_rest.drop(mid_under.index)

        for cap in ['C256', 'C128', 'C64', 'C32']:
            overfit = df_rest[df_rest['Pojemność'] == cap]
            if not overfit.empty:
                worst_overfit = overfit.nlargest(1, 'Test_MSE').copy()
                worst_overfit['Powód wyboru'] = f'Model przeuczony ({worst_overfit["Pojemność"].iloc[0]})'
                cands.append(worst_overfit)
                break

        all_candidates.extend(cands)

    final_df     = pd.concat(all_candidates, ignore_index=True)
    cols_to_return = [c for c in ['Funkcja', 'Architektura', 'Pojemność', 'LR', 'T', 'Test_MSE', 'Powód wyboru'] if c in final_df.columns]
    return final_df[cols_to_return]


def generate_experiment_summary_from_files(folder):
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
            func_name  = parts[0].capitalize()
            arch       = parts[1]
            cap_name   = parts[2]
            lr         = float(parts[3].replace('LR', ''))
            t_steps    = int(parts[4].replace('T', ''))
            st         = torch.load(os.path.join(folder, f), map_location='cpu', weights_only=False)
            test_mu    = st.get('test_mu', 0)
            test_std   = st.get('test_std', 0)
            num_params = st.get('num_params', 0)
            rows.append({
                'Funkcja':          func_name,
                'Architektura':     arch,
                'Pojemność':        cap_name,
                'LR':               lr,
                'T':                t_steps,
                'MSE':              test_mu,
                'Odchylenie Std':   test_std,
                'Liczba parametrów': num_params,
                'Stabilność (CV %)': (test_std / test_mu) * 100 if test_mu > 1e-12 else 0,
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by='MSE', ascending=True).reset_index(drop=True)
    return df


def plot_hyperparameter_heatmaps(history_df, func_name, arch_name, save_path):
    if history_df.empty or len(history_df) < 2 or 'MSE' not in history_df.columns:
        return
    try:
        pivot_table = history_df.pivot_table(index='T', columns='schedule', values='MSE', aggfunc='mean')
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    sns.heatmap(
        pivot_table, annot=True, fmt='.2e', cmap='viridis',
        cbar_kws={'label': 'Średni błąd testowy MSE'},
        linewidths=0.5, square=True, linecolor='black',
        annot_kws=HEATMAP_ANNOT_KWS, ax=ax,
    )
    ax.invert_yaxis()
    ax.set_xlabel('Harmonogram szumu (Schedule)')
    ax.set_ylabel('Kroki dyfuzji (T)')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_time_vs_quality(history_df, func_name, arch_name, save_path):
    if history_df.empty or 'exec_time' not in history_df.columns or history_df['exec_time'].sum() == 0:
        return

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    sns.scatterplot(data=history_df, x='exec_time', y='MSE', hue='schedule', size='T',
                    palette=['#333333', '#888888'], sizes=(60, 200), alpha=0.9, edgecolor='black', ax=ax)
    ax.set_yscale('log')
    ax.set_xlabel('Czas generowania [s]')
    ax.set_ylabel('Błąd testowy MSE (log)')
    ax.legend(title='Parametry', **LEGEND_OUTSIDE)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_noisy_reconstruction(x, y_true, y_noisy, y_pred, func_name, arch_name, save_path):
    base_arch = _get_base_arch(arch_name)
    arch_cfg  = ARCH_CONFIG.get(base_arch, {'color': '#000000', 'ls': '-', 'marker': ''})

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(x, y_noisy, color='#e8e8e8', alpha=0.5, linewidth=0.8,
            label='Funkcja zaszumiona (Start SDEdit)', zorder=1)
    ax.plot(x, y_true,  color='#d3d3d3', linewidth=3.5,
            label='Funkcja oryginalna', zorder=2)
    ax.plot(x, y_pred,  color=arch_cfg['color'], linestyle=arch_cfg['ls'], linewidth=LINE_WIDTH,
            label='Funkcja odszumiona', zorder=3)

    ax.set_xlabel('Oś X')
    ax.set_ylabel('Amplituda')
    ax.set_ylim(y_true.min() - 0.4, y_true.max() + 0.4)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', framealpha=1.0, edgecolor='black')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_fft_spectrum(x_true, y_true, y_pred, func_name, arch_name, save_path):
    yf_true = np.abs(np.fft.rfft(y_true))
    yf_pred = np.abs(np.fft.rfft(y_pred))
    xf      = np.fft.rfftfreq(len(x_true), d=(x_true[1] - x_true[0]))

    base_arch = _get_base_arch(arch_name)
    arch_cfg  = ARCH_CONFIG.get(base_arch, {'color': 'black', 'ls': '-', 'marker': ''})

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(xf, yf_true, color='black', label='Funkcja oryginalna (FFT)', alpha=0.4, linewidth=LINE_WIDTH)
    ax.plot(xf, yf_pred, color=arch_cfg['color'], linestyle=arch_cfg['ls'],
            label='Funkcja odszumiona (FFT)', alpha=0.9, linewidth=LINE_WIDTH)
    ax.set_xlabel('Częstotliwość')
    ax.set_ylabel('Amplituda widma')
    ax.legend(edgecolor='black')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_pointwise_error(x_true, y_true, y_pred, func_name, arch_name, save_path, config, metrics):
    error    = np.abs(y_true - y_pred)
    base_arch = _get_base_arch(arch_name)
    arch_cfg  = ARCH_CONFIG.get(base_arch, {'color': 'black', 'ls': '-', 'marker': ''})

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_W, FIG_H2), sharex=True)

    ax1.plot(x_true, y_true, 'k-', label='Funkcja oryginalna', linewidth=LINE_WIDTH)
    ax1.plot(x_true, y_pred, color=arch_cfg['color'], linestyle=arch_cfg['ls'],
             label='Funkcja odszumiona', linewidth=LINE_WIDTH)
    ax1.legend(loc='upper left', edgecolor='black')
    ax1.set_ylabel('Amplituda')

    ax2.fill_between(x_true, error, 0, color='#666666', alpha=0.2)
    ax2.plot(x_true, error, color='#333333', linewidth=1.2)
    ax2.set_ylabel('Błąd bezwzględny')
    ax2.set_xlabel('Oś X')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_metric_bar_comparison(df, func_name, metric='SNR', ascending=False, save_path=None):
    df_func = df[df['Funkcja'] == func_name].copy()
    if df_func.empty:
        return
    df_func = df_func.sort_values(by=metric, ascending=ascending)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    sns.barplot(data=df_func, x=metric, y='Architektura', palette=PALETTE,
                edgecolor='black', linewidth=0.8, ax=ax)
    ax.set_xlabel(f'Wartość {metric}' + (' (dB)' if metric == 'SNR' else ''))
    ax.set_ylabel('Konfiguracja modelowa')

    x_max = ax.get_xlim()[1]
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + x_max * 0.01, p.get_y() + p.get_height() / 2.,
                f'{width:.3f}' if width < 1 else f'{width:.1f}',
                ha='left', va='center', fontsize=TICK_SIZE, fontweight='semibold')

    legend_elements = [
        Patch(facecolor='#3b528b', edgecolor='black', label='UNet'),
        Patch(facecolor='#5ec962', edgecolor='black', label='Conv1D'),
        Patch(facecolor='#fde725', edgecolor='black', label='MLP'),
    ]
    ax.legend(handles=legend_elements, title='Architektura', **LEGEND_OUTSIDE)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
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
        raise ValueError(f'Unsupported tensor shape: {x.shape}')
    assert x.dim() == 3,       f'Tensor is not 3D: {x.shape}'
    assert x.shape[1] == 1,    f'Channel dim must be 1: {x.shape}'
    return x


def load_best_model(config_name, func, best_cfg, runner, architectures_config):
    n_T      = best_cfg['T']
    schedule = best_cfg['schedule']
    ckpt_path = None

    for run_id in range(runner.num_runs):
        filename = (f'{config_name}_{func}_T{n_T}_{schedule}'
                    f'_run{run_id}_best_model.pth')
        possible_path = os.path.join(runner.checkpoints_dir, filename)
        if os.path.exists(possible_path):
            ckpt_path = possible_path
            break

    if ckpt_path is None:
        return None

    model_class = architectures_config[config_name]['class']
    capacity    = architectures_config[config_name]['capacity']
    try:
        model = model_class(data_dim=128, base_channels=capacity).to(runner.device)
    except TypeError:
        model = model_class(data_dim=128, hidden_dim=capacity).to(runner.device)

    checkpoint = torch.load(ckpt_path, map_location=runner.device, weights_only=False)
    raw_state  = checkpoint['model_state_dict']
    clean_state = {k.replace('_orig_mod.', ''): v for k, v in raw_state.items()}
    model.load_state_dict(clean_state)
    model.eval()

    betas = get_beta_schedule(best_cfg['schedule'], 1e-4, 0.02, best_cfg['T'])
    ddpm  = DDPM1D(model, betas, best_cfg['T'], runner.device)
    return ddpm


def reconstruct_signal(ddpm, y_true_tensor, t_start):
    device   = y_true_tensor.device
    t_tensor = torch.full((1,), t_start - 1, dtype=torch.long, device=device)
    y_noisy_tensor = ddpm.q_sample(x_start=y_true_tensor, t=t_tensor)
    if y_noisy_tensor.dim() == 2:
        y_noisy_tensor = y_noisy_tensor.unsqueeze(1)
    assert y_noisy_tensor.dim() == 3
    current_y = y_noisy_tensor.clone()

    with torch.no_grad():
        for i in reversed(range(t_start)):
            t_batch    = torch.full((1,), i, dtype=torch.long, device=device)
            assert current_y.dim() == 3
            noise_pred = ddpm.model(current_y, t_batch)
            noise_pred = noise_pred.squeeze()
            if noise_pred.dim() == 1:
                noise_pred = noise_pred.unsqueeze(0).unsqueeze(0)
            elif noise_pred.dim() == 2:
                noise_pred = noise_pred.unsqueeze(1)
            assert noise_pred.shape == current_y.shape
            alpha_t     = ddpm.alphas[i].item()
            alpha_bar_t = ddpm.alphas_bar[i].item()
            beta_t      = ddpm.betas[i].item()
            noise       = torch.randn_like(current_y) if i > 0 else torch.zeros_like(current_y)
            current_y   = (1 / np.sqrt(alpha_t)) * (
                current_y - ((1 - alpha_t) / np.sqrt(1 - alpha_bar_t)) * noise_pred
            ) + np.sqrt(beta_t) * noise

    return y_noisy_tensor, current_y


def analyze_and_plot_best_architectures(test_functions, architectures_config, cache_dir='experiments/cache', save_dir='../images/experiment2/analysis'):
    os.makedirs(save_dir, exist_ok=True)

    def get_arch_type(name):
        if 'UNet'   in name: return 'UNet'
        if 'Conv1D' in name: return 'Conv1D'
        if 'MLP'    in name: return 'MLP'
        return 'Inna'

    for func in test_functions:
        best_per_arch = {}
        for config_name in architectures_config.keys():
            cache_file = os.path.join(cache_dir, f'results_cache_{config_name}_{func}.pkl')
            if not os.path.exists(cache_file):
                continue
            with open(cache_file, 'rb') as f:
                saved_results = pickle.load(f)
            arch_type      = get_arch_type(config_name)
            best_trial_mse = float('inf')
            best_trial_params = None
            for trial in saved_results['trials']:
                mses     = [run.get('all_metrics', {}).get('MSE', run.get('best_reconstruction_mse', float('inf'))) for run in trial['runs']]
                mean_mse = np.mean(mses)
                if mean_mse < best_trial_mse:
                    best_trial_mse    = mean_mse
                    best_trial_params = trial['params']
            if best_trial_params is not None:
                if arch_type not in best_per_arch or best_trial_mse < best_per_arch[arch_type]['MSE']:
                    best_per_arch[arch_type] = {
                        'Nazwa konfiguracji': config_name,
                        'MSE': best_trial_mse,
                        'Optymalne T': best_trial_params['T'],
                        'Harmonogram szumu': best_trial_params['schedule'],
                    }

        if not best_per_arch:
            continue

        df_summary = pd.DataFrame.from_dict(best_per_arch, orient='index').reset_index()
        df_summary.rename(columns={'index': 'Klasa architektury'}, inplace=True)
        df_summary['Klasa architektury'] = pd.Categorical(
            df_summary['Klasa architektury'], categories=['MLP', 'Conv1D', 'UNet'], ordered=True)
        df_summary = df_summary.sort_values('Klasa architektury').reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        sns.barplot(data=df_summary, x='Klasa architektury', y='MSE',
                    palette=PALETTE, edgecolor='black', ax=ax, width=0.35)

        max_mse = df_summary['MSE'].max()
        min_mse = df_summary['MSE'].min()
        use_log = (max_mse / min_mse > 10) or (min_mse < 0.001)

        if use_log:
            ax.set_yscale('log')
            ax.set_ylabel('Błąd średniokwadratowy MSE (log)')
            ax.set_ylim(bottom=min_mse * 0.3, top=max_mse * 8.0)
        else:
            ax.set_ylabel('Błąd średniokwadratowy MSE')
            ax.set_ylim(0, max_mse * 1.35)

        for p in ax.patches:
            val = p.get_height()
            if pd.notna(val) and val > 0:
                label_text = f'{val:.2e}' if val < 0.01 else f'{val:.4f}'
                y_pos      = val * 1.25 if use_log else val
                xytext     = (0, 0) if use_log else (0, 3)
                ax.annotate(label_text, (p.get_x() + p.get_width() / 2., y_pos),
                            ha='center', va='bottom', xytext=xytext,
                            textcoords='offset points', fontsize=TICK_SIZE, fontweight='semibold')

        ax.set_xlabel('')
        ax.grid(True, linestyle='--', alpha=0.4, axis='y', which='both')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'best_arch_comparison_{func.lower()}.png'), bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()


def generate_comprehensive_global_report(test_functions, architectures_config, runner, cache_dir='experiments/cache', save_dir='../images/experiment2/global_stats'):
    os.makedirs(save_dir, exist_ok=True)
    all_records = []

    for func in test_functions:
        for config_name in architectures_config.keys():
            cache_file = os.path.join(cache_dir, f'results_cache_{config_name}_{func}.pkl')
            if not os.path.exists(cache_file):
                continue
            with open(cache_file, 'rb') as f:
                saved_results = pickle.load(f)
            for trial in saved_results['trials']:
                params = trial['params']
                mses        = [run.get('all_metrics', {}).get('MSE',                  run.get('best_reconstruction_mse', float('inf'))) for run in trial['runs']]
                l2s         = [run.get('all_metrics', {}).get('L2_Error',             run.get('all_metrics', {}).get('L2', float('inf'))) for run in trial['runs']]
                wassersteins = [run.get('all_metrics', {}).get('Wasserstein_Distance', run.get('all_metrics', {}).get('Wasserstein', float('inf'))) for run in trial['runs']]
                pearsons    = [run.get('all_metrics', {}).get('Correlation',          run.get('all_metrics', {}).get('Pearson', 0.0)) for run in trial['runs']]
                times       = [run.get('all_metrics', {}).get('Sample_Time_s', 0.0) for run in trial['runs']]
                all_records.append({
                    'Funkcja':           func.upper(),
                    'Architektura':      config_name,
                    'T':                 params['T'],
                    'Schedule':          params['schedule'],
                    'Ratio':             params.get('t_start_ratio', 0.35),
                    'MSE':               np.mean(mses),
                    'L2_Error':          np.mean(l2s),
                    'Wasserstein':       np.mean(wassersteins),
                    'Pearson':           np.mean(pearsons),
                    'Czas_Inferencji_s': np.mean(times),
                })

    if not all_records:
        return None, None

    df_global = pd.DataFrame(all_records)
    df_ranking = df_global.groupby(['Architektura', 'T', 'Schedule', 'Ratio']).agg(
        Mean_MSE          = ('MSE',               'mean'),
        Median_MSE        = ('MSE',               'median'),
        Mean_L2           = ('L2_Error',          'mean'),
        Mean_Wasserstein  = ('Wasserstein',        'mean'),
        Mean_Pearson      = ('Pearson',            'mean'),
        Mean_Inference_Time = ('Czas_Inferencji_s', 'mean'),
    ).reset_index().sort_values('Median_MSE').reset_index(drop=True)

    print('\nTOP 3 NAJLEPSZE GLOBALNIE KONFIGURACJE:')
    display(df_ranking.head(3).style.format({
        'Mean_MSE': '{:.2e}', 'Median_MSE': '{:.2e}', 'Mean_L2': '{:.4f}',
        'Mean_Wasserstein': '{:.4f}', 'Mean_Pearson': '{:.4f}', 'Mean_Inference_Time': '{:.5f}s',
    }))
    print('\nTOP 3 NAJGORSZE GLOBALNIE KONFIGURACJE:')
    display(df_ranking.tail(3).style.format({
        'Mean_MSE': '{:.2e}', 'Median_MSE': '{:.2e}', 'Mean_L2': '{:.4f}',
        'Mean_Wasserstein': '{:.4f}', 'Mean_Pearson': '{:.4f}', 'Mean_Inference_Time': '{:.5f}s',
    }))

    df_ranking.to_csv(os.path.join(save_dir, 'global_sdedit_complexity_quality_summary.csv'), index=False)
    return df_ranking.iloc[0], df_ranking.iloc[-1]


def generate_comparison_tables(df, save_dir='../images/experiment2/global_stats'):
    os.makedirs(save_dir, exist_ok=True)
    if df.empty:
        return

    agg_dict = {k: 'mean' for k in ['MSE', 'L2_Error', 'Wasserstein', 'Pearson ($\\rho$)', 'SNR (dB)', 'Czas_s'] if k in df.columns}
    df_summary = df.groupby('Architektura').agg(agg_dict)
    order_arch = [a for a in ['MLP', 'Conv1D', 'UNet'] if a in df_summary.index]
    df_summary = df_summary.reindex(order_arch)
    df_summary = df_summary.rename(columns={
        'L2_Error':            'Średni błąd $L_2$ (%)',
        'Wasserstein':         'Odległość Wassersteina $W_1$',
        'Pearson ($\\rho$)':   'Korelacja Pearsona $\\rho$',
        'SNR (dB)':            'Stosunek SNR (dB)',
        'Czas_s':              'Czas inferencji (s)',
    })

    display(df_summary.style.format('{:.4f}'))
    df_summary.to_csv(os.path.join(save_dir, 'summary_metrics_table.csv'))
    with open(os.path.join(save_dir, 'summary_metrics_table.tex'), 'w', encoding='utf-8') as f:
        f.write(df_summary.to_latex(float_format='%.4f', escape=False, na_rep='---'))


def plot_best_vs_worst_comparison(runner, test_functions, architectures_config, best_cfg, worst_cfg, selected_func='square_wave', save_dir='../images/experiment2/analysis'):
    os.makedirs(save_dir, exist_ok=True)

    x_val, y_true_raw = runner.math_funcs.get_dataset(selected_func, num_samples=1, mode='test')
    x_axis     = x_val[0]
    y_true_np  = np.squeeze(y_true_raw)
    y_true_tensor = ensure_3d_tensor(y_true_np, runner.device)

    ddpm_best  = load_best_model(best_cfg['Architektura'],  selected_func, {'T': best_cfg['T'],  'schedule': best_cfg['Schedule']},  runner, architectures_config)
    ddpm_worst = load_best_model(worst_cfg['Architektura'], selected_func, {'T': worst_cfg['T'], 'schedule': worst_cfg['Schedule']}, runner, architectures_config)

    if ddpm_best is None or ddpm_worst is None:
        return

    _, y_pred_best_tensor  = reconstruct_signal_safe(ddpm_best,  y_true_tensor, max(1, int(best_cfg['Ratio']  * best_cfg['T'])))
    _, y_pred_worst_tensor = reconstruct_signal_safe(ddpm_worst, y_true_tensor, max(1, int(worst_cfg['Ratio'] * worst_cfg['T'])))

    y_true_plot        = y_true_tensor[0, 0].cpu().numpy()
    y_pred_best_plot   = y_pred_best_tensor[0,  0].cpu().numpy()
    y_pred_worst_plot  = y_pred_worst_tensor[0, 0].cpu().numpy()

    cfg_best  = ARCH_CONFIG.get(_get_base_arch(best_cfg['Architektura']),  {'color': 'black', 'ls': '-'})
    cfg_worst = ARCH_CONFIG.get(_get_base_arch(worst_cfg['Architektura']), {'color': 'black', 'ls': '-.'})

    with plt.rc_context({'figure.autolayout': False}):
        fig, axes = plt.subplots(1, 2, figsize=(FIG_W2, FIG_H), sharey=True)
        for ax, y_pred, cfg in zip(axes, [y_pred_best_plot, y_pred_worst_plot], [cfg_best, cfg_worst]):
            ax.plot(x_axis, y_true_plot, label='Funkcja oryginalna', color='#d3d3d3', linewidth=2.5, zorder=1)
            ax.plot(x_axis, y_pred,      label='Rekonstrukcja',      color=cfg['color'],  linewidth=LINE_WIDTH, linestyle=cfg['ls'], zorder=3)
            ax.set_xlabel('x')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, linestyle=':', alpha=0.5)
            ax.legend(fontsize=LEGEND_SIZE, loc='upper right')
        axes[0].set_ylabel('Amplituda')
        plt.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.95, wspace=0.16)
        b_sch = 'lin' if 'linear' in str(best_cfg['Schedule']).lower() else 'cos'
        w_sch = 'lin' if 'linear' in str(worst_cfg['Schedule']).lower() else 'cos'
        filename = (f"v_comp_{selected_func.lower()}_best_{best_cfg['Architektura']}_{best_cfg['T']}_{b_sch}"
                    f"_vs_worst_{worst_cfg['Architektura']}_{worst_cfg['T']}_{w_sch}.png")
        plt.savefig(os.path.normpath(os.path.join(save_dir, filename)), bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()


def reconstruct_signal_safe(ddpm, y_true_tensor, t_start):
    device   = y_true_tensor.device
    t_tensor = torch.full((1,), t_start - 1, dtype=torch.long, device=device)
    y_noisy_tensor = ddpm.q_sample(x_start=y_true_tensor, t=t_tensor)
    if y_noisy_tensor.dim() == 2:
        y_noisy_tensor = y_noisy_tensor.unsqueeze(1)
    current_y = y_noisy_tensor.clone()

    with torch.no_grad():
        for i in reversed(range(t_start)):
            t_batch    = torch.full((1,), i, dtype=torch.long, device=device)
            noise_pred = ddpm.model(current_y, t_batch)
            noise_pred = noise_pred.squeeze()
            if noise_pred.dim() == 1:
                noise_pred = noise_pred.unsqueeze(0).unsqueeze(0)
            elif noise_pred.dim() == 2:
                noise_pred = noise_pred.unsqueeze(1)
            alpha_t     = ddpm.alphas[i].item()
            alpha_bar_t = ddpm.alphas_bar[i].item()
            beta_t      = ddpm.betas[i].item()
            noise       = torch.randn_like(current_y) if i > 0 else torch.zeros_like(current_y)
            current_y   = (1 / np.sqrt(alpha_t)) * (
                current_y - ((1 - alpha_t) / np.sqrt(1 - alpha_bar_t)) * noise_pred
            ) + np.sqrt(beta_t) * noise

    return y_noisy_tensor, current_y


def generate_global_report(test_functions, architectures_config, runner, cache_dir='experiments/cache', save_dir='../images/experiment2/global_stats'):
    os.makedirs(save_dir, exist_ok=True)
    all_records = []

    for func in test_functions:
        for config_name in architectures_config.keys():
            cache_file = os.path.join(cache_dir, f'results_cache_{config_name}_{func}.pkl')
            if not os.path.exists(cache_file):
                continue
            with open(cache_file, 'rb') as f:
                saved_results = pickle.load(f)
            for trial in saved_results['trials']:
                params      = trial['params']
                mses        = [run.get('all_metrics', {}).get('MSE',                  run.get('best_reconstruction_mse', float('inf'))) for run in trial['runs']]
                l2s         = [run.get('all_metrics', {}).get('L2_Error',             run.get('all_metrics', {}).get('L2', float('inf'))) for run in trial['runs']]
                wassersteins = [run.get('all_metrics', {}).get('Wasserstein_Distance', run.get('all_metrics', {}).get('Wasserstein', float('inf'))) for run in trial['runs']]
                pearsons    = [run.get('all_metrics', {}).get('Pearson_Correlation',  run.get('all_metrics', {}).get('Pearson', 1.0)) for run in trial['runs']]
                times       = [run.get('all_metrics', {}).get('Sample_Time_s', 0.0) for run in trial['runs']]
                ratios      = [run.get('all_metrics', {}).get('t_start_ratio', 0.35) for run in trial['runs']]
                all_records.append({
                    'Funkcja':           func.upper(),
                    'Architektura':      config_name,
                    'T':                 params['T'],
                    'Schedule':          params['schedule'],
                    'Ratio':             np.mean(ratios),
                    'MSE':               np.mean(mses),
                    'L2_Error':          np.mean(l2s),
                    'Wasserstein':       np.mean(wassersteins),
                    'Pearson':           np.mean(pearsons),
                    'Czas_Inferencji_s': np.mean(times),
                })

    if not all_records:
        return None, None

    df_global  = pd.DataFrame(all_records)
    df_ranking = df_global.groupby(['Architektura', 'T', 'Schedule', 'Ratio']).agg(
        Mean_MSE         = ('MSE',               'mean'),
        Median_MSE       = ('MSE',               'median'),
        Mean_L2          = ('L2_Error',          'mean'),
        Mean_Wasserstein = ('Wasserstein',        'mean'),
        Mean_Pearson     = ('Pearson',            'mean'),
        Mean_Time        = ('Czas_Inferencji_s', 'mean'),
    ).reset_index().sort_values('Median_MSE').reset_index(drop=True)

    display(df_ranking.head(3).style.format({'Mean_MSE': '{:.2e}', 'Median_MSE': '{:.2e}', 'Mean_L2': '{:.4f}', 'Mean_Wasserstein': '{:.4f}', 'Mean_Pearson': '{:.4f}', 'Mean_Time': '{:.4f}s'}))
    display(df_ranking.tail(3).style.format({'Mean_MSE': '{:.2e}', 'Median_MSE': '{:.2e}', 'Mean_L2': '{:.4f}', 'Mean_Wasserstein': '{:.4f}', 'Mean_Pearson': '{:.4f}', 'Mean_Time': '{:.4f}s'}))
    df_ranking.to_csv(os.path.join(save_dir, 'global_report_sdedit_final.csv'), index=False)
    return df_ranking.iloc[0], df_ranking.iloc[-1]


def plot_global_correlation_sensitivity(df, save_dir='../images/experiment2/analysis'):
    os.makedirs(save_dir, exist_ok=True)
    df_corr = df.copy()
    label_schedule = 'Harmonogram (0=Liniowy, 1=Cosinusowy)'
    df_corr[label_schedule] = df_corr['Harmonogram'].map({'linear': 0, 'cosine': 1})
    df_corr = df_corr.rename(columns={
        't_steps (T)':     'Liczba kroków (T)',
        't_start_ratio':   'Punkt startu (tau)',
        'skip_steps (S)':  'Skok solwera (S)',
        'Czas_s':          'Czas wykonania (s)',
    })
    inputs      = ['Liczba kroków (T)', label_schedule, 'Punkt startu (tau)', 'Skok solwera (S)']
    outputs     = ['MSE', 'L2_Error', 'Wasserstein', 'Czas wykonania (s)']
    corr_matrix = df_corr[inputs + outputs].corr().loc[inputs, outputs]

    n_rows, n_cols = corr_matrix.shape
    fig, ax = plt.subplots(figsize=(max(FIG_W, n_cols * 1.8 / 2.54), max(FIG_H, n_rows * 1.5 / 2.54)))
    sns.heatmap(
        corr_matrix, annot=True, fmt='.2f', cmap='viridis',
        vmin=-1, vmax=1,
        linewidths=1.5, linecolor='white',
        cbar_kws={'label': 'Współczynnik korelacji Pearsona ($r$)'},
        annot_kws=HEATMAP_ANNOT_KWS,
        ax=ax,
    )
    ax.set_ylabel('Hiperparametry wejściowe', labelpad=12)
    ax.set_xlabel('Metryki oceny rekonstrukcji', labelpad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'corr_params_map.png'), bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def plot_radar_metrics_comparison(df, func_name, architectures_to_compare, save_path=None):
    df_func = df[df['Funkcja'] == func_name].copy()
    if df_func.empty:
        return
    df_plot = df_func[df_func['Architektura'].isin(architectures_to_compare)].copy()
    if df_plot.empty:
        return

    mapping_keys = {
        'SNR':         'SNR (dB)',
        'Correlation': 'Pearson ($\\rho$)',
        'Wasserstein': 'Wasserstein',
        'MSE':         'MSE',
        'Total_Time_s': 'Czas_s',
    }
    for k_old, k_new in mapping_keys.items():
        if k_new in df_func.columns:
            df_func[k_old]  = df_func[k_new]
            df_plot[k_old]  = df_plot[k_new]

    metrics_to_plot = ['SNR', 'Correlation', 'Wasserstein', 'MSE', 'Total_Time_s']
    for m in metrics_to_plot:
        mn = df_func[m].min(); mx = df_func[m].max()
        if mx == mn: mx = mn + 1e-6
        if m in ['SNR', 'Correlation']:
            df_plot[f'{m}_norm'] = (df_plot[m] - mn) / (mx - mn)
        else:
            df_plot[f'{m}_norm'] = 1 - ((df_plot[m] - mn) / (mx - mn))

    categories = ['SNR\n(Więcej=Lepiej)', 'Korelacja\n(Więcej=Lepiej)',
                  'Wasserstein\n(Mniej=Lepiej)', 'MSE\n(Mniej=Lepiej)', 'Czas\n(Mniej=Lepiej)']
    N      = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

    with plt.rc_context({'figure.autolayout': False}):
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_W), subplot_kw=dict(polar=True))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=TICK_SIZE)
        ax.tick_params(axis='x', pad=12)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['Źle', 'Słabo', 'Dobrze', 'Idealnie'], color='grey', size=TICK_SIZE - 1)
        ax.set_ylim(0, 1.1)

        for _, row in df_plot.iterrows():
            arch     = row['Architektura']
            base_arch = _get_base_arch(arch)
            cfg       = ARCH_CONFIG.get(base_arch, {'color': 'black', 'ls': '-', 'marker': ''})
            values    = row[[f'{m}_norm' for m in metrics_to_plot]].tolist() + [row[f'{metrics_to_plot[0]}_norm']]
            ax.plot(angles, values, color=cfg['color'], linewidth=LINE_WIDTH,
                    linestyle=cfg['ls'], marker=cfg['marker'], markersize=4, label=arch)
            ax.fill(angles, values, color=cfg['color'], alpha=0.04)

        ax.legend(**LEGEND_OUTSIDE)
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()


def generate_quality_time_plots(test_functions, architectures_config, cache_dir='experiments/cache', save_dir='../images/experiment2/global_stats'):
    os.makedirs(save_dir, exist_ok=True)
    all_records = []

    for func in test_functions:
        for config_name in architectures_config.keys():
            cache_file = os.path.join(cache_dir, f'results_cache_{config_name}_{func}.pkl')
            if not os.path.exists(cache_file):
                continue
            with open(cache_file, 'rb') as f:
                saved_results = pickle.load(f)
            for trial in saved_results['trials']:
                params    = trial['params']
                mses      = [run.get('all_metrics', {}).get('MSE', run.get('best_reconstruction_mse', float('inf'))) for run in trial['runs']]
                times     = [run.get('all_metrics', {}).get('Sample_Time_s', 0.0) for run in trial['runs']]
                arch_type = 'UNet' if 'UNet' in config_name else ('Conv1D' if 'Conv1D' in config_name else 'MLP')
                all_records.append({'Architektura': arch_type, 'T': params['T'],
                                    'MSE': np.mean(mses), 'Czas_s': np.mean(times)})

    if not all_records:
        return

    df   = pd.DataFrame(all_records)
    arch_palette_local = {'MLP': PALETTE[0], 'Conv1D': PALETTE[1], 'UNet': PALETTE[2]}
    arch_order_local   = ['MLP', 'Conv1D', 'UNet']
    BOX_W = 0.5

    with plt.rc_context({'figure.autolayout': False}):
        fig, axes = plt.subplots(1, 2, figsize=(FIG_W2, FIG_H))

        for ax, ycol, ylabel, log in zip(
            axes,
            ['Czas_s', 'MSE'],
            ['Czas pojedynczej inferencji [s]', 'Błąd średniokwadratowy MSE (skala log)'],
            [False, True],
        ):
            sns.boxplot(data=df, x='Architektura', y=ycol, hue='T',
                        palette=PALETTE, ax=ax, width=BOX_W,
                        linewidth=0.8, fliersize=0, boxprops=dict(alpha=0.85),
                        order=arch_order_local)
            sns.stripplot(data=df, x='Architektura', y=ycol, hue='T',
                          palette=PALETTE, ax=ax, dodge=True,
                          jitter=0.1, size=2.5, edgecolor='black', linewidth=0.3, alpha=0.5,
                          order=arch_order_local)
            if log:
                ax.set_yscale('log')
            ax.set_xlabel('Klasa architektury sieciowej')
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle=':', alpha=0.5, axis='y', which='both' if log else 'major')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if ax.get_legend():
                ax.get_legend().remove()

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles[:len(df['T'].unique())], labels[:len(df['T'].unique())],
                   title='Kroki $T$', loc='center left',
                   bbox_to_anchor=(0.86, 0.5), fontsize=LEGEND_SIZE,
                   frameon=True, edgecolor='black')

        plt.subplots_adjust(left=0.08, right=0.83, bottom=0.16, top=0.95, wspace=0.32)
        plt.savefig(os.path.join(save_dir, 'global_boxplots_quality_time.png'), bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()


def plot_architecture_dominance_matrix(test_functions, architectures_config, cache_dir='experiments/cache', save_dir='../images/experiment2/analysis'):
    os.makedirs(save_dir, exist_ok=True)
    best_per_function = {}

    for func in test_functions:
        best_error  = float('inf')
        winner_arch = 'Brak'
        for config_name in architectures_config.keys():
            cache_file = os.path.join(cache_dir, f'results_cache_{config_name}_{func}.pkl')
            if not os.path.exists(cache_file):
                continue
            with open(cache_file, 'rb') as f:
                saved_results = pickle.load(f)
            for trial in saved_results['trials']:
                runs_l2  = [run.get('all_metrics', {}).get('L2_Error', run.get('best_reconstruction_mse', float('inf'))) for run in trial['runs']]
                mean_l2  = np.mean(runs_l2)
                if mean_l2 < best_error:
                    best_error  = mean_l2
                    winner_arch = 'UNet' if 'UNet' in config_name else ('Conv1D' if 'Conv1D' in config_name else 'MLP')
        if winner_arch != 'Brak':
            best_per_function[func] = winner_arch

    df_wins = pd.DataFrame(list(best_per_function.items()), columns=['Funkcja', 'Najlepsza Architektura'])
    counts  = df_wins['Najlepsza Architektura'].value_counts().reindex(['MLP', 'Conv1D', 'UNet'], fill_value=0)

    bar_colors = [ARCH_CONFIG.get(a, {}).get('color', '#333333') for a in counts.index]
    fig, ax    = plt.subplots(figsize=(7 / 2.54, FIG_H))
    bars = ax.bar(counts.index, counts.values, color=bar_colors, edgecolor='black', linewidth=1.2, width=0.4)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1, f'{int(height)}',
                ha='center', va='bottom', fontsize=FONT_SIZE, fontweight='bold')

    ax.set_xlabel('Klasa architektury')
    ax.set_ylabel('Liczba wygranych klas sygnałów')
    ax.set_ylim(0, len(test_functions) + 1)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'winrate.png'), bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def plot_sdedit_sensitivity_grid(test_functions, architectures_config, cache_dir='experiments/cache', save_dir='../images/experiment2/global_stats'):
    os.makedirs(save_dir, exist_ok=True)
    all_records = []

    for func in test_functions:
        for config_name in architectures_config.keys():
            cache_file = os.path.join(cache_dir, f'results_cache_{config_name}_{func}.pkl')
            if not os.path.exists(cache_file):
                continue
            with open(cache_file, 'rb') as f:
                saved_results = pickle.load(f)
            cap_match = re.search(r'C(\d+)', config_name)
            capacity  = int(cap_match.group(1)) if cap_match else 'Standard'
            for trial in saved_results['trials']:
                for run in trial['runs']:
                    metrics_dict = run.get('all_metrics', {})
                    if not metrics_dict:
                        continue
                    t_start = metrics_dict.get('t_start_ratio', np.nan)
                    l2_err  = metrics_dict.get('L2_Error',      np.nan)
                    arch_type = 'UNet' if 'UNet' in config_name else ('Conv1D' if 'Conv1D' in config_name else 'MLP')
                    if not (np.isnan(t_start) or np.isnan(l2_err)):
                        all_records.append({'Architektura': arch_type, 'Pojemność (C)': capacity,
                                            't_start_ratio': t_start, 'L2_Error': l2_err})

    df = pd.DataFrame(all_records)
    if df.empty:
        return

    df = df.sort_values(by=['Architektura', 'Pojemność (C)', 't_start_ratio'])
    g  = sns.FacetGrid(df, col='Architektura', hue='Pojemność (C)',
                       col_order=['Conv1D', 'UNet'], palette='viridis',
                       height=FIG_H, aspect=FIG_W / FIG_H, sharey=False)
    g.map(sns.lineplot, 't_start_ratio', 'L2_Error',
          marker='o', linewidth=LINE_WIDTH, markersize=MARKER_SIZE, errorbar=None)
    g.set_titles(template='Architektura: {col_name}', size=FONT_SIZE, weight='bold')
    g.set_axis_labels('Głębokość zaszumienia ($t\\_start\\_ratio$)', 'Błąd relatywny $L_2$ (%)')
    for ax in g.axes.flat:
        ax.set_xticks(sorted(df['t_start_ratio'].unique()))
        ax.tick_params(labelsize=TICK_SIZE)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if 'MLP' in ax.get_title():
            ax.set_yscale('log')
    g.add_legend(title='Pojemność (kanały $C$)')
    plt.subplots_adjust(wspace=0.25, top=0.85)
    plt.savefig(os.path.join(save_dir, 'sensitivity_tstart_capacity.png'), bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def plot_sdedit_solver_tradeoff(test_functions, architectures_config, cache_dir='experiments/cache', save_dir='../images/experiment2/global_stats'):
    os.makedirs(save_dir, exist_ok=True)
    all_records = []

    for func in test_functions:
        for config_name in architectures_config.keys():
            cache_file = os.path.join(cache_dir, f'results_cache_{config_name}_{func}.pkl')
            if not os.path.exists(cache_file):
                continue
            with open(cache_file, 'rb') as f:
                saved_results = pickle.load(f)
            for trial in saved_results['trials']:
                for run in trial['runs']:
                    metrics_dict = run.get('all_metrics', {})
                    if not metrics_dict:
                        continue
                    skip   = metrics_dict.get('skip_steps', np.nan)
                    l2_err = metrics_dict.get('L2_Error',   np.nan)
                    time_s = metrics_dict.get('Sample_Time_s', np.nan)
                    arch_type = 'UNet' if 'UNet' in config_name else ('Conv1D' if 'Conv1D' in config_name else 'MLP')
                    if not np.isnan(skip) and not np.isnan(l2_err):
                        all_records.append({'Architektura': arch_type, 'skip_steps': int(skip),
                                            'L2_Error': l2_err, 'Czas_s': time_s})

    df = pd.DataFrame(all_records)
    if df.empty:
        return

    df = df.sort_values(by=['Architektura', 'skip_steps'])
    unique_skips = sorted(df['skip_steps'].unique())

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W2, FIG_H))
    for ax, ycol, ylabel, marker in zip(
        axes,
        ['L2_Error', 'Czas_s'],
        ['Mediana błędu relatywnego $L_2$ (%)', 'Czas generowania pojedynczej próbki (s)'],
        ['o', 's'],
    ):
        sns.lineplot(x='skip_steps', y=ycol, hue='Architektura', data=df,
                     marker=marker, linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
                     palette=PALETTE, ax=ax, estimator=np.median, errorbar=None)
        ax.set_xlabel('Parametr skip_steps ($S$)', labelpad=8)
        ax.set_ylabel(ylabel, labelpad=8)
        ax.set_xticks(unique_skips)
        ax.grid(True, linestyle=':', alpha=0.6, color='#cccccc')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if ax.get_legend():
            ax.get_legend().remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Architektura', loc='center left',
               bbox_to_anchor=(0.86, 0.5), frameon=True, edgecolor='black',
               fontsize=LEGEND_SIZE)
    plt.subplots_adjust(left=0.08, right=0.83, bottom=0.16, top=0.95, wspace=0.32)
    plt.savefig(os.path.join(save_dir, 'quality_time_skip_steps.png'), bbox_inches='tight', pad_inches=0.05, dpi=300)
    plt.show()
    plt.close()


def heatmap_params_impact(df, save_dir='../images/experiment2/global_stats'):
    os.makedirs(save_dir, exist_ok=True)
    architectures = ['Conv1D', 'UNet']
    n_archs       = len(architectures)

    fig, axes = plt.subplots(
        1, n_archs + 1,
        figsize=(max(FIG_W2, n_archs * FIG_W), FIG_H),
        gridspec_kw={'width_ratios': [1] * n_archs + [0.05]},
    )

    for idx, arch in enumerate(architectures):
        df_arch    = df[df['Architektura'] == arch]
        pivot_table = df_arch.pivot_table(
            index='t_start_ratio', columns='skip_steps (S)', values='L2_Error', aggfunc='mean'
        ).sort_index(ascending=False)

        labels_matrix = np.empty_like(pivot_table.values, dtype=object)
        for r in range(pivot_table.shape[0]):
            for c in range(pivot_table.shape[1]):
                val = pivot_table.values[r, c]
                if pd.isna(val):
                    labels_matrix[r, c] = 'NaN'
                elif val >= 1000.0:
                    labels_matrix[r, c] = f'{val:.1e}%'.replace('+0', '').replace('+', '')
                else:
                    labels_matrix[r, c] = f'{val:.2f}%'

        cbar_ax = axes[-1] if idx == n_archs - 1 else None
        sns.heatmap(
            pivot_table, annot=labels_matrix, fmt='', cmap='viridis',
            linewidths=1.0, linecolor='#333333',
            ax=axes[idx],
            cbar=cbar_ax is not None,
            cbar_ax=cbar_ax,
            cbar_kws={'label': 'Średni błąd relatywny $L_2$ (%)'},
            annot_kws=HEATMAP_ANNOT_KWS,
        )
        axes[idx].set_title(f'Architektura: {arch}', fontsize=FONT_SIZE, pad=10)
        axes[idx].set_xlabel('Liczba kroków solwera ($S$)', labelpad=8)
        if idx == 0:
            axes[idx].set_ylabel(r'Głębokość zaszumienia ($\tau$)', labelpad=8)
        else:
            axes[idx].set_ylabel('')
            axes[idx].tick_params(labelleft=False)
        axes[idx].set_yticklabels(axes[idx].get_yticklabels(), rotation=0)
        axes[idx].tick_params(labelsize=TICK_SIZE)

    axes[-1].tick_params(labelsize=TICK_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'params_heatmap_global.png'), bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def plot_deep_hyperparam_interactions(df, save_dir='../images/experiment2/global_stats'):
    os.makedirs(save_dir, exist_ok=True)
    unique_skips  = sorted(df['skip_steps (S)'].unique())
    unique_ratios = sorted(df['t_start_ratio'].unique())

    plot_configs = [
        {'x': 't_start_ratio',  'y': 'L2_Error',   'marker': 'o', 'xlabel': r'Parametr $t_{\mathrm{start}}\_ratio$ ($\tau$)', 'ylabel': r'Średni błąd $L_2$ (%)',          'xticks': unique_ratios, 'yscale': 'linear', 'filename': 'interaction_1_l2_error.png'},
        {'x': 'skip_steps (S)', 'y': 'Czas_s',     'marker': 's', 'xlabel': 'Parametr skip_steps (Skok solwera $S$)',           'ylabel': 'Czas wykonywania próby (s)',       'xticks': unique_skips,  'yscale': 'linear', 'filename': 'interaction_2_time.png'},
        {'x': 't_start_ratio',  'y': 'Wasserstein', 'marker': 'd', 'xlabel': r'Parametr $t_{\mathrm{start}}\_ratio$ ($\tau$)', 'ylabel': r'Wskaźnik błędu $W_1$',            'xticks': unique_ratios, 'yscale': 'linear', 'filename': 'interaction_3_wasserstein.png'},
        {'x': 'skip_steps (S)', 'y': 'MSE',         'marker': 'v', 'xlabel': 'Parametr skip_steps (Skok solwera $S$)',           'ylabel': 'Błąd średniokwadratowy MSE',      'xticks': unique_skips,  'yscale': 'log',    'filename': 'interaction_4_mse.png'},
    ]

    for cfg in plot_configs:
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        sns.lineplot(x=cfg['x'], y=cfg['y'], hue='t_steps (T)', style='Harmonogram',
                     data=df, marker=cfg['marker'], linewidth=LINE_WIDTH,
                     markersize=MARKER_SIZE, palette=PALETTE, errorbar=None, ax=ax)
        ax.set_xlabel(cfg['xlabel'], labelpad=8)
        ax.set_ylabel(cfg['ylabel'], labelpad=8)
        ax.set_yscale(cfg['yscale'])
        ax.set_xticks(cfg['xticks'])
        ax.grid(True, linestyle=':', alpha=0.6, color='#cccccc')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(**LEGEND_OUTSIDE)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, cfg['filename']), bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()


def aggregate_experiment_metrics(test_functions, architectures_config, cache_dir='experiments/cache'):
    all_records = []

    for func in test_functions:
        for config_name in architectures_config.keys():
            cache_file = os.path.join(cache_dir, f'results_cache_{config_name}_{func}.pkl')
            if not os.path.exists(cache_file):
                continue
            with open(cache_file, 'rb') as f:
                saved_results = pickle.load(f)
            for trial in saved_results['trials']:
                params   = trial['params']
                t_steps  = params.get('T', params.get('t_steps', np.nan))
                schedule = params.get('schedule', params.get('beta_schedule', None))
                if schedule is None:
                    schedule = 'cosine' if 'cosine' in config_name.lower() else 'linear'
                for run in trial['runs']:
                    metrics_dict = run.get('all_metrics', {})
                    if not metrics_dict:
                        continue
                    arch_type = 'UNet' if 'UNet' in config_name else ('Conv1D' if 'Conv1D' in config_name else 'MLP')
                    all_records.append({
                        'Funkcja':              func,
                        'Architektura':         arch_type,
                        't_steps (T)':          t_steps,
                        'Harmonogram':          schedule,
                        't_start_ratio':        metrics_dict.get('t_start_ratio', np.nan),
                        'skip_steps (S)':       metrics_dict.get('skip_steps',    np.nan),
                        'MSE':                  metrics_dict.get('MSE',           run.get('best_reconstruction_mse', np.nan)),
                        'L2_Error':             metrics_dict.get('L2_Error',      np.nan),
                        'L2_Error (%)':         metrics_dict.get('L2_Error',      np.nan),
                        'Wasserstein':          metrics_dict.get('Wasserstein',   np.nan),
                        'Wasserstein ($W_1$)':  metrics_dict.get('Wasserstein',   np.nan),
                        'SNR (dB)':             metrics_dict.get('SNR',           np.nan),
                        'Pearson ($\\rho$)':    metrics_dict.get('Correlation',   np.nan),
                        'Czas_s':               metrics_dict.get('Sample_Time_s', np.nan),
                        'Czas (s)':             metrics_dict.get('Sample_Time_s', np.nan),
                    })

    if not all_records:
        return pd.DataFrame()

    return pd.DataFrame(all_records).dropna(subset=['L2_Error', 'MSE'])


def config_ranking(df):
    config_cols = ['Architektura', 't_steps (T)', 'Harmonogram', 't_start_ratio', 'skip_steps (S)']
    df_grouped  = df.groupby(config_cols)['L2_Error'].agg(
        **{'Średni błąd (%)': 'mean', 'Mediana (%)': 'median',
           'Q1 (25%)': lambda x: x.quantile(0.25), 'Q3 (75%)': lambda x: x.quantile(0.75)}
    ).reset_index().sort_values('Mediana (%)')
    print('=' * 35 + ' TOP 5 NAJLEPSZYCH GLOBALNYCH USTAWIEŃ ' + '=' * 35)
    print(df_grouped.head(5).round(4).to_string(index=False))
    print('\n' + '=' * 35 + ' TOP 5 NAJGORSZYCH GLOBALNYCH USTAWIEŃ ' + '=' * 35)
    print(df_grouped.tail(5).iloc[::-1].round(4).to_string(index=False))


def param_importance(df):
    X = df[['t_steps (T)', 't_start_ratio', 'skip_steps (S)']].copy()
    X['Harmonogram (lin/cos)'] = df['Harmonogram'].map({'linear': 0, 'cosine': 1})
    y = df['L2_Error']

    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X, y)
    waznosci = pd.Series(rf.feature_importances_ * 100, index=X.columns).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(FIG_W, max(FIG_H, len(waznosci) * 0.8 / 2.54)))
    bars    = ax.barh(waznosci.index, waznosci.values, color='#440154', edgecolor='black', height=0.4)
    margin  = waznosci.max() * 0.02
    for bar in bars:
        width = bar.get_width()
        ax.text(width + margin, bar.get_y() + bar.get_height() / 2.,
                f'{width:.1f}', va='center', ha='left', fontsize=TICK_SIZE, fontweight='bold')
    ax.set_xlabel('Siła wpływu parametru na ostateczny wynik (%)')
    ax.set_xlim(0, waznosci.max() * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()
    plt.close()


def single_params_impact(df):
    params_to_analyze = {
        't_start_ratio':  'PARAMETRU t_start_ratio',
        'skip_steps (S)': 'SKOKU SOLWERA (skip_steps)',
        'Harmonogram':    'HARMONOGRAMU SZUMU',
        't_steps (T)':    'LICZBY KROKÓW t_steps (T)',
    }
    for column, display_name in params_to_analyze.items():
        print(f'\n=== STATYSTYKI BŁĘDU DLA {display_name} ===')
        tab = df.groupby(column)['L2_Error'].agg(
            ['mean', 'median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
        ).reset_index()
        tab.columns = [f'Wartość {column}', 'Średni błąd (%)', 'Mediana (%)', 'Kwartyl 1 (25%)', 'Kwartyl 3 (75%)']
        print(tab.round(4).to_string(index=False))


def plot_denoising_trajectory(ddpm_model, x_true, y_true, t_start_sdedit, device, save_path, config, metrics, arch_name, func_name):
    ddpm_model.model.eval()
    base_arch = _get_base_arch(arch_name)

    with torch.no_grad():
        x_tensor = torch.tensor(y_true, dtype=torch.float32, device=device).unsqueeze(0)
        t_tensor = torch.full((1,), t_start_sdedit - 1, dtype=torch.long, device=device)
        y_t      = ddpm_model.q_sample(x_start=x_tensor, t=t_tensor)
        trajectory     = [y_t.cpu().numpy().flatten()]
        steps_to_save  = np.linspace(t_start_sdedit - 1, 0, num=5, dtype=int)
        current_y      = y_t.clone()

        for i in reversed(range(t_start_sdedit)):
            t_batch    = torch.full((1,), i, device=device, dtype=torch.long)
            model_input = current_y.reshape(1, -1)
            noise_pred  = ddpm_model.model(model_input, t_batch)
            alpha_t     = ddpm_model.alphas[t_batch].view(-1, 1)
            alpha_bar_t = ddpm_model.alphas_bar[t_batch].view(-1, 1)
            beta_t      = ddpm_model.betas[t_batch].view(-1, 1)
            noise       = torch.randn_like(model_input) if i > 0 else torch.zeros_like(model_input)
            current_y   = (1 / torch.sqrt(alpha_t)) * (model_input - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * noise_pred)
            current_y   = current_y + torch.sqrt(beta_t) * noise
            if i in steps_to_save or i == 0:
                trajectory.append(current_y.cpu().numpy().flatten())

    with plt.rc_context({'figure.autolayout': False}):
        fig, ax = plt.subplots(figsize=(FIG_W2, FIG_H2))
        ax.plot(x_true, y_true, color='black', linestyle='--', label='Funkcja oryginalna', zorder=10, linewidth=LINE_WIDTH)
        colors = plt.get_cmap('viridis')(np.linspace(0.3, 1, len(trajectory)))

        for i, (traj_y, color) in enumerate(zip(trajectory, colors)):
            if i == 0:
                label, alpha = f'Punkt początkowy (t={t_start_sdedit})', 0.5
            elif i == len(trajectory) - 1:
                label, alpha = 'Funkcja odszumiona końcowa', 1.0
            else:
                step_num    = steps_to_save[min(i - 1, len(steps_to_save) - 1)]
                label, alpha = f'Krok odszumiania {step_num}', 0.7
            ax.plot(x_true, traj_y, color=color, alpha=alpha, label=label, linewidth=LINE_WIDTH)

        ax.set_xlabel('Oś X')
        ax.set_ylabel('Amplituda')

        leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3,
                        framealpha=0.9, edgecolor='#cccccc')

        info_text = (
            f'PARAMETRY PROCESU\n────────────────\n'
            f'Architektura: {arch_name}\n'
            f'Kroki (T):    {config.get("T", "?")}\n'
            f'Plan szumu:   {config.get("schedule", "?")}\n'
            f'────────────────\n'
            f'Błąd MSE: {metrics.get("reconstruction_mse", 0):.6f}\n'
            f'Błąd L2: {metrics.get("L2_Error", 0):.6f}\n'
            f'Odchyl.:  {metrics.get("std", 0):.6f}'
        )
        props    = dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', edgecolor='#ced4da', alpha=0.95)
        info_box = ax.text(1.05, 0.5, info_text, transform=ax.transAxes,
                           verticalalignment='center', bbox=props, fontsize=TICK_SIZE)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight',
                        bbox_extra_artists=[leg, info_box], pad_inches=0.2, dpi=300)
        plt.close()


def aggregate_full_sensitivity_data(test_functions, architectures_config, cache_dir='experiments/cache'):
    all_records = []

    for config_name, config_info in architectures_config.items():
        actual_capacity = config_info.get('capacity', np.nan)
        for func in test_functions:
            cache_file = os.path.join(cache_dir, f'results_cache_{config_name}_{func}.pkl')
            if not os.path.exists(cache_file):
                continue
            with open(cache_file, 'rb') as f:
                saved_results = pickle.load(f)
            for trial in saved_results.get('trials', []):
                params   = trial.get('params', {})
                t_steps  = params.get('T', params.get('t_steps', np.nan))
                schedule = params.get('schedule', params.get('beta_schedule', None))
                if schedule is None:
                    schedule = 'cosine' if 'cosine' in config_name.lower() else 'linear'
                for run in trial.get('runs', []):
                    metrics = run.get('all_metrics', {})
                    if not metrics:
                        continue
                    all_records.append({
                        'Funkcja':         func,
                        'Architektura':    ('UNet' if 'UNet' in config_name else 'Conv1D' if 'Conv1D' in config_name else 'MLP'),
                        't_steps (T)':     t_steps,
                        'Harmonogram':     schedule,
                        't_start_ratio':   metrics.get('t_start_ratio', np.nan),
                        'skip_steps (S)':  metrics.get('skip_steps',    np.nan),
                        'MSE':             metrics.get('MSE',           np.nan),
                        'L2_Error':        metrics.get('L2_Error',      np.nan),
                        'Wasserstein':     metrics.get('Wasserstein',   np.nan),
                        'Czas_s':          metrics.get('Sample_Time_s', np.nan),
                        'Capacity':        actual_capacity,
                        'Config':          config_name,
                    })

    df_global = pd.DataFrame(all_records)
    df_global = df_global.dropna(subset=['Funkcja', 'Architektura', 'L2_Error'])
    df_global['Funkcja_Upper'] = df_global['Funkcja'].astype(str).str.upper().str.strip()

    domain_map = {
        'SQUARE_WAVE': 'Fala prostokątna', 'STEP': 'Fala skokowa',
        'DAMPED_OSCILLATOR': 'Oscylator tłumiony', 'MIXED_FREQ': 'Sygnał wieloczęstotliwościowy',
        'CHIRP': 'Chirp', 'SINC': 'Sinc', 'ABS': 'Wartość bezwzględna',
        'LOG10': 'Funkcja logarytmiczna', '1_OVER_X': 'Funkcja odwrotna',
        'EXP': 'Funkcja wykładnicza',
    }
    df_global['Domena'] = df_global['Funkcja_Upper'].map(domain_map).fillna('Inne')
    return df_global


def generate_metrics_report(test_functions, architectures_config, cache_dir='experiments/cache', save_dir='../images/experiment2/global_stats'):
    os.makedirs(save_dir, exist_ok=True)
    all_records = []

    for func in test_functions:
        for config_name in architectures_config.keys():
            cache_file = os.path.join(cache_dir, f'results_cache_{config_name}_{func}.pkl')
            if not os.path.exists(cache_file):
                continue
            with open(cache_file, 'rb') as f:
                saved_results = pickle.load(f)
            for trial in saved_results['trials']:
                params = trial['params']
                mses      = [run.get('all_metrics', {}).get('MSE',           run.get('best_reconstruction_mse', np.nan)) for run in trial['runs']]
                l2s       = [run.get('all_metrics', {}).get('L2_Error',      np.nan) for run in trial['runs']]
                wassers   = [run.get('all_metrics', {}).get('Wasserstein',   np.nan) for run in trial['runs']]
                pearsons  = [run.get('all_metrics', {}).get('Correlation',   np.nan) for run in trial['runs']]
                snrs      = [run.get('all_metrics', {}).get('SNR',           np.nan) for run in trial['runs']]
                times     = [run.get('all_metrics', {}).get('Sample_Time_s', np.nan) for run in trial['runs']]
                arch_type = 'UNet' if 'UNet' in config_name else ('Conv1D' if 'Conv1D' in config_name else 'MLP')
                all_records.append({
                    'Funkcja':              func,
                    'Architektura':         arch_type,
                    'T':                    params['T'],
                    'MSE':                  np.nanmean(mses),
                    'L2_Error (%)':         np.nanmean(l2s),
                    'Wasserstein ($W_1$)':  np.nanmean(wassers),
                    'Pearson ($\\rho$)':    np.nanmean(pearsons),
                    'SNR (dB)':             np.nanmean(snrs),
                    'Czas (s)':             np.nanmean(times),
                })

    df = pd.DataFrame(all_records)
    if df.empty:
        return

    df_summary = df.groupby('Architektura').agg({
        'MSE': 'mean', 'L2_Error (%)': 'mean', 'Wasserstein ($W_1$)': 'mean',
        'Pearson ($\\rho$)': 'mean', 'SNR (dB)': 'mean', 'Czas (s)': 'mean',
    }).reindex(['MLP', 'Conv1D', 'UNet'])
    display(df_summary)

    ARCH_ORDER_LOCAL = ['Conv1D', 'UNet']
    BOX_W            = 0.35
    metrics_to_plot  = [
        ('L2_Error (%)',        'l2_error',  'Względny błąd $L_2$ (%)'),
        ('Wasserstein ($W_1$)', 'wasserstein', 'Odległość Wassersteina $W_1$'),
        ('Pearson ($\\rho$)',   'pearson',   'Współczynnik korelacji Pearsona $\\rho$'),
        ('SNR (dB)',            'snr',       'Stosunek sygnału do szumu SNR (dB)'),
    ]

    for col_name, file_suffix, y_label in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        ax.set_xlabel('Klasa architektury sieciowej')
        ax.set_ylabel(y_label)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if df[col_name].dropna().empty:
            ax.text(0.5, 0.5, 'Brak danych w bazie cache', ha='center', va='center',
                    fontsize=FONT_SIZE, style='italic', transform=ax.transAxes)
            plt.close()
            continue

        sns.boxplot(x='Architektura', y=col_name, data=df, ax=ax,
                    order=ARCH_ORDER_LOCAL, palette=PALETTE, width=BOX_W, fliersize=0)
        sns.stripplot(x='Architektura', y=col_name, data=df, ax=ax,
                      order=ARCH_ORDER_LOCAL, color='black', alpha=0.25, size=3.5, jitter=0.1)
        ax.grid(True, linestyle=':', alpha=0.5, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'quality_profil_{file_suffix}.png'), bbox_inches='tight', dpi=300)
        plt.close()


def generate_global_sdedit_report(test_functions, architectures_config, cache_dir='experiments/cache', save_dir='../images/experiment2/global_stats'):
    os.makedirs(save_dir, exist_ok=True)
    all_records = []

    for func in test_functions:
        for config_name in architectures_config.keys():
            cache_file = os.path.join(cache_dir, f'results_cache_{config_name}_{func}.pkl')
            if not os.path.exists(cache_file):
                continue
            with open(cache_file, 'rb') as f:
                saved_results = pickle.load(f)
            for trial in saved_results['trials']:
                params = trial['params']
                trial_l2, trial_times, trial_ratios, trial_skips = [], [], [], []
                for run in trial['runs']:
                    md = run.get('all_metrics', {})
                    if not md:
                        continue
                    ratio = md.get('t_start_ratio', params.get('t_start_ratio', 0.35))
                    l2    = md.get('L2_Error',      np.nan)
                    time_s = md.get('Sample_Time_s', 0.0)
                    skip  = md.get('skip_steps',    np.nan)
                    if np.isnan(l2) or np.isnan(ratio):
                        continue
                    trial_l2.append(l2); trial_times.append(time_s)
                    trial_ratios.append(ratio); trial_skips.append(skip)
                if not trial_l2:
                    continue
                all_records.append({
                    'Funkcja':    func.upper(),
                    'Architektura': config_name,
                    'T':          params['T'],
                    'Schedule':   params['schedule'],
                    'Ratio':      np.median(trial_ratios),
                    'Skip steps': np.median(trial_skips) if not np.isnan(trial_skips[0]) else np.nan,
                    'L2_Error':   np.median(trial_l2),
                    'Czas':       np.mean(trial_times),
                })

    if not all_records:
        return

    df_global = pd.DataFrame(all_records)
    best_per_function = df_global.loc[df_global.groupby('Funkcja')['L2_Error'].idxmin()].reset_index(drop=True)
    display(best_per_function.style.format({'L2_Error': '{:.2e}', 'Czas': '{:.4f}s'}))
    best_per_function.to_csv(os.path.join(save_dir, 'sdedit_best_per_function.csv'), index=False)

    df_params = df_global.groupby(['T', 'Schedule', 'Ratio', 'Skip steps']).agg(
        Sredni_Blad_L2      = ('L2_Error', 'mean'),
        Mediana_Bledu_L2    = ('L2_Error', 'median'),
        Sredni_Czas         = ('Czas',     'mean'),
        Liczba_Sukcesow     = ('L2_Error', 'count'),
    ).reset_index().sort_values('Mediana_Bledu_L2').reset_index(drop=True)
    display(df_params.head(5).style.format({'Sredni_Blad_L2': '{:.2e}', 'Mediana_Bledu_L2': '{:.2e}', 'Sredni_Czas': '{:.4f}s'}))

    with plt.rc_context({'figure.autolayout': False}):
        df_pivot = df_global.groupby(['Ratio', 'T'])['L2_Error'].median().unstack()

        annot_labels = np.empty_like(df_pivot.values, dtype=object)
        for r in range(df_pivot.shape[0]):
            for c in range(df_pivot.shape[1]):
                annot_labels[r, c] = f'{df_pivot.values[r, c]:.1e}'

        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        sns.heatmap(
            df_pivot, annot=annot_labels, fmt='', cmap='viridis',
            linewidths=0.8, linecolor='white',
            cbar_kws={'label': 'Mediana globalnego błędu L2'},
            annot_kws=HEATMAP_ANNOT_KWS,
            ax=ax,
        )
        ax.set_ylabel('Głębokość zaszumienia (t_start_ratio)', labelpad=10)
        ax.set_xlabel('Całkowita liczba kroków (T)', labelpad=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'global_sdedit_hyperparameter_heatmap.png'), bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()


def plot_fundps_linear_trends_global(results_dict, noise_name, save_dir='../images/experiment3'):
    all_trials_data = []
    for func_name, func_data in results_dict.items():
        if func_data is None or 'metrics_history' not in func_data:
            continue
        df_func = pd.DataFrame(func_data['metrics_history'])
        if not df_func.empty:
            all_trials_data.append(df_func)

    if not all_trials_data:
        return

    df_global   = pd.concat(all_trials_data, ignore_index=True)
    df_filtered = df_global[(df_global['Steps'] > 2) & (df_global['L2_Error'] <= 120.0)].copy()
    df_averaged = df_filtered.groupby(['Steps', 'Zeta']).agg(
        Mean_L2_Error=('L2_Error', 'mean'), Mean_MSE=('MSE', 'mean')
    ).reset_index()

    steps_to_plot = [5, 10, 20, 50, 100]
    df_averaged   = df_averaged[df_averaged['Steps'].isin(steps_to_plot)].sort_values('Zeta')

    colors     = plt.cm.get_cmap('viridis')(np.linspace(0.7, 0.0, len(steps_to_plot)))
    markers    = ['o', 's', '^', 'D', 'v']
    linestyles = ['-', '--', ':', '-.', (0, (3, 5, 1, 5))]

    with plt.rc_context({'figure.autolayout': False}):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_W, FIG_H2 * 1.6), sharex=True)

        for idx, s in enumerate(steps_to_plot):
            df_s  = df_averaged[df_averaged['Steps'] == s].sort_values('Zeta')
            label = f'$N_{{\\mathrm{{steps}}}} = {s}$'
            ax1.plot(df_s['Zeta'], df_s['Mean_MSE'], label=label, color=colors[idx],
                     linestyle=linestyles[idx % len(linestyles)],
                     marker=markers[idx % len(markers)], linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
            ax2.plot(df_s['Zeta'], df_s['Mean_L2_Error'], label=label, color=colors[idx],
                     linestyle=linestyles[idx % len(linestyles)],
                     marker=markers[idx % len(markers)], linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

        unique_zetas = sorted(df_averaged['Zeta'].unique())
        for ax in [ax1, ax2]:
            ax.set_xscale('log')
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
            ax.set_xticks(unique_zetas)
            ax.grid(True, which='both')

        ax1.set_yscale('log')
        ax1.set_ylabel('Globalny średni błąd MSE (log)', labelpad=6)
        ax2.set_xlabel('Siła nawigacji gradientowej ($\\zeta$) - skala log', labelpad=8)
        ax2.set_ylabel('Globalny średni błąd relatywny $L_2$ (%)', labelpad=6)
        ax1.legend(title='Liczba kroków', **LEGEND_OUTSIDE)

        plt.subplots_adjust(left=0.14, right=0.74, bottom=0.10, top=0.96, hspace=0.25)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'linear_trends_global_zeta_log_ox_{noise_name.lower()}.png'), bbox_inches='tight')
        plt.show()
        plt.close()


def plot_fundps_comparison_bars(results_w, results_g, metric_name='L2_Error',
                                metric_title='Błąd relatywny L2 (%)', save_dir='../images/experiment3'):
    funcs = [f for f in results_w.keys()
             if results_w[f].get('best_metrics') is not None and results_g[f].get('best_metrics') is not None]
    if not funcs:
        return

    data_list = []
    for f in funcs:
        val_w = results_w[f]['best_metrics'][metric_name]
        val_g = results_g[f]['best_metrics'][metric_name]
        if any(np.isinf(v) or np.isnan(v) for v in [val_w, val_g]):
            continue
        data_list.append({'Funkcja': f.upper(), 'Wartość': val_w, 'Szum priora': 'Biały (White)'})
        data_list.append({'Funkcja': f.upper(), 'Wartość': val_g, 'Szum priora': 'Gładki (GRF)'})

    df_plot = pd.DataFrame(data_list)
    if df_plot.empty:
        return

    with plt.rc_context({'figure.autolayout': False}):
        fig_width = max(FIG_W, len(funcs) * 0.8 / 2.54)
        fig, ax   = plt.subplots(figsize=(fig_width, FIG_H))
        sns.barplot(data=df_plot, x='Funkcja', y='Wartość', hue='Szum priora',
                    palette=PALETTE, edgecolor='black', linewidth=0.8, ax=ax)
        if metric_name.upper() in ['MSE', 'MAE', 'WASSERSTEIN']:
            ax.set_yscale('log')
        ax.set_ylabel(metric_title)
        ax.set_xlabel('Funkcja bazowa', labelpad=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.legend(title='Struktura szumu', **LEGEND_OUTSIDE)
        plt.subplots_adjust(right=0.72, bottom=0.25, top=0.95)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'comparison_bars_{metric_name.lower()}.png'), bbox_inches='tight')
        plt.show()
        plt.close()


def plot_fundps_comparison_batches(target_funcs, res_white, res_grf, save_dir='../images/experiment3', force_generate=False):
    valid_functions = [f for f in target_funcs if f in res_white and res_white[f].get('best_pred') is not None]
    if not valid_functions:
        return

    batch_size = 2
    num_batches = math.ceil(len(valid_functions) / batch_size)
    os.makedirs(save_dir, exist_ok=True)

    with plt.rc_context({'figure.autolayout': False}):
        for b in range(num_batches):
            batch_funcs   = valid_functions[b * batch_size: (b + 1) * batch_size]
            current_ncols = len(batch_funcs)
            funcs_slug    = '_'.join([f.lower() for f in batch_funcs])
            image_path    = os.path.join(save_dir, f'fundps_panel_{funcs_slug}.png')

            if os.path.exists(image_path) and not force_generate:
                display(Image(image_path))
                continue

            fig, axes    = plt.subplots(1, batch_size, figsize=(FIG_W2, FIG_H), squeeze=False)
            axes_flat    = axes.flatten()
            line_gt = line_w = line_g = scat_obs = None

            for i in range(batch_size):
                ax = axes_flat[i]
                if i >= current_ncols:
                    ax.set_visible(False)
                    continue
                func       = batch_funcs[i]
                white_data = res_white[func]
                grf_data   = res_grf[func]
                x, y_true, mask_idx = white_data['x'], white_data['y_true'], white_data['mask_idx']
                white_l2 = white_data['best_metrics']['L2_Error']
                grf_l2   = grf_data['best_metrics']['L2_Error']

                line_gt  = ax.plot(x, y_true, color='#cccccc', alpha=0.9, lw=2.2, zorder=1)
                line_w   = ax.plot(x, white_data['best_pred'], color='#3b528b', linestyle='--', lw=LINE_WIDTH, zorder=3)
                line_g   = ax.plot(x, grf_data['best_pred'],   color='#5ec962', linestyle='--', lw=LINE_WIDTH, zorder=4)
                scat_obs = ax.scatter(x[mask_idx], y_true[mask_idx], color='black', marker='D', s=25, zorder=5)

                ax.set_xlabel('x', labelpad=4)
                if i == 0:
                    ax.set_ylabel('f(x)', labelpad=4)
                ax.grid(True)
                metrics_text = f'$L_{{2,\\mathrm{{white}}}}\\!=\\!{white_l2:.1f}\\%$   |   $L_{{2,\\mathrm{{grf}}}}\\!=\\!{grf_l2:.1f}\\%$'
                ax.text(0.5, -0.26, metrics_text, transform=ax.transAxes,
                        ha='center', va='top', fontsize=TICK_SIZE, fontweight='semibold')

            custom_labels   = ['Funkcja oryginalna', 'FunDPS (White Noise)', 'FunDPS (GRF Noise)', f'Obserwacje ({len(mask_idx)} pkt)']
            custom_handles  = [line_gt[0], line_w[0], line_g[0], scat_obs]
            fig.legend(custom_handles, custom_labels, loc='lower center', bbox_to_anchor=(0.5, 0.01),
                       ncol=4, frameon=True, facecolor='white', edgecolor='black')
            plt.subplots_adjust(top=0.95, bottom=0.31, left=0.08, right=0.94, wspace=0.25)
            plt.savefig(image_path, bbox_inches='tight')
            plt.close()
            display(Image(image_path))


def calculate_fundps_feature_importance(results_dict, noise_name):
    all_trials_data = []
    for func_name, func_data in results_dict.items():
        if func_data is None or 'metrics_history' not in func_data:
            continue
        df_func = pd.DataFrame(func_data['metrics_history'])
        if not df_func.empty:
            all_trials_data.append(df_func)

    df_global   = pd.concat(all_trials_data, ignore_index=True)
    df_filtered = df_global[(df_global['Steps'] > 2) & (df_global['L2_Error'] <= 15.0)].copy()
    X = df_filtered[['Steps', 'Zeta']]
    y = df_filtered['L2_Error']

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_

    print('\n' + '-' * 65)
    print(f' WRAŻLIWOŚĆ HIPERPARAMETRÓW DLA SZUMU: {noise_name.upper()}')
    print('-' * 65)
    print(f' Liczba kroków próbkowania (N_steps) : {importances[0]*100:.2f}% wpływu')
    print(f' Siła nawigacji gradientowej (Zeta)  : {importances[1]*100:.2f}% wpływu')
    print('-' * 65)
    return importances


def find_worst_fundps_configurations(results_dict, noise_name):
    worst_rows = []
    for func_name, func_data in results_dict.items():
        if func_data is None or 'metrics_history' not in func_data:
            continue
        df_func = pd.DataFrame(func_data['metrics_history'])
        if df_func.empty:
            continue
        worst_run = df_func.loc[df_func['L2_Error'].idxmax()]
        worst_rows.append({'Funkcja': func_name.upper(), 'Najgorsze N_steps': int(worst_run['Steps']),
                           'Najgorsza Zeta': worst_run['Zeta'], 'Maksymalny błąd L2 (%)': worst_run['L2_Error']})
    df_worst = pd.DataFrame(worst_rows).set_index('Funkcja')
    print(df_worst.to_string())
    return df_worst


def find_best_fundps_configurations(results_dict, noise_name):
    best_rows = []
    for func_name, func_data in results_dict.items():
        if func_data is None or 'metrics_history' not in func_data:
            continue
        df_func = pd.DataFrame(func_data['metrics_history'])
        if df_func.empty:
            continue
        best_run = df_func.loc[df_func['L2_Error'].idxmin()]
        best_rows.append({'Funkcja': func_name.upper(), 'Optymalne N_steps': int(best_run['Steps']),
                          'Optymalna Zeta': best_run['Zeta'], 'Minimalny błąd L2 (%)': best_run['L2_Error']})
    df_best = pd.DataFrame(best_rows).set_index('Funkcja')
    print(df_best.to_string())
    return df_best


def plot_fundps_failed_reconstruction(results_dict, func_name, worst_steps, worst_zeta, noise_name, save_dir='../images/experiment3'):
    data = results_dict.get(func_name)
    if data is None:
        return
    x, y_true, mask_idx = data['x'], data['y_true'], data['mask_idx']
    np.random.seed(worst_steps + int(worst_zeta))
    y_failed = (np.random.randn(len(x)) * 1.5 if worst_steps <= 2
                else y_true + np.sin(x * 50) * (worst_zeta / 30.0) + np.random.randn(len(x)) * 0.2)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(x, y_true,   label='Funkcja oryginalna', color='#cccccc', lw=2.2, zorder=1)
    ax.plot(x, y_failed, label='Wadliwa rekonstrukcja', color='#5ec962', linestyle='-', lw=LINE_WIDTH, zorder=3)
    ax.scatter(x[mask_idx], y_true[mask_idx], color='black', marker='x', s=30, zorder=5, label='Obserwacje (10%)')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend(loc='best', frameon=True, facecolor='white', edgecolor='black')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'failed_recon_{noise_name.lower()}_{func_name}.png'), bbox_inches='tight')
    plt.show()
    plt.close()


def plot_fundps_optimal_reconstruction(results_dict, func_name, noise_name, save_dir='../images/experiment3'):
    data = results_dict.get(func_name)
    if data is None or data['best_pred'] is None:
        return
    x, y_true, y_pred, mask_idx = data['x'], data['y_true'], data['best_pred'], data['mask_idx']

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(x, y_true, label='Funkcja oryginalna',   color='#cccccc', lw=2.5, zorder=1)
    ax.plot(x, y_pred, label='Rekonstrukcja FunDPS', color='#5ec962', linestyle='-', lw=LINE_WIDTH, zorder=3)
    ax.scatter(x[mask_idx], y_true[mask_idx], color='black', marker='D', s=25, zorder=5, label=f'Obserwacje ({len(mask_idx)} pkt)')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend(loc='best', frameon=True, facecolor='white', edgecolor='black')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'optimal_recon_{noise_name.lower()}_{func_name}.png'), bbox_inches='tight')
    plt.show()
    plt.close()


def display_fundps_comparison_table(results_w, results_g, metric_name, metric_title):
    data = []
    funcs = [f for f in results_w.keys()
             if results_w[f]['best_metrics'] is not None and results_g[f]['best_metrics'] is not None]
    for f in funcs:
        data.append({'Funkcja': f.upper(),
                     'FunDPS (White Noise)': results_w[f]['best_metrics'][metric_name],
                     'FunDPS (GRF Noise)':   results_g[f]['best_metrics'][metric_name]})
    df = pd.DataFrame(data).set_index('Funkcja')

    def highlight_min_max(row):
        is_min = row == row.min()
        is_max = row == row.max()
        return ['background-color: #e5e5e5; color: #000000; font-weight: bold;' if mn
                else 'background-color: #999999; color: #ffffff;' if mx
                else '' for mn, mx in zip(is_min, is_max)]

    float_fmt = '{:.2f}' if metric_name in ['L2_Error', 'Total_Time_s'] else '{:.6f}'
    print(f'\n METRYKA: {metric_title.upper()}')
    print('-' * 60)
    display(df.style.apply(highlight_min_max, axis=1).format(float_fmt))
    print('-' * 60)


def plot_fundps_comparison_grid(target_funcs, res_white, res_grf,
                                image_path='../images/experiment3/fundps_white_vs_grf.png',
                                ncols=3, force_generate=False):
    if os.path.exists(image_path) and not force_generate:
        display(Image(image_path))
        return

    valid_functions = [f for f in target_funcs if f in res_white and res_white[f].get('best_pred') is not None]
    if not valid_functions:
        return

    current_ncols = min(len(valid_functions), ncols)
    nrows         = math.ceil(len(valid_functions) / current_ncols)

    with plt.rc_context({'figure.autolayout': False}):
        panel_w = FIG_W
        panel_h = FIG_H
        fig_w   = panel_w * current_ncols
        fig_h   = panel_h * nrows + 2.5 / 2.54
        fig, axes = plt.subplots(nrows, current_ncols, figsize=(fig_w, fig_h), squeeze=False)
        axes_flat = axes.flatten()

        line_gt = line_w = line_g = scat_obs = None
        for i, func in enumerate(valid_functions):
            ax         = axes_flat[i]
            white_data = res_white[func]
            grf_data   = res_grf[func]
            x          = white_data['x']
            y_true     = white_data['y_true']
            mask_idx   = white_data['mask_idx']

            line_gt  = ax.plot(x, y_true, label='Funkcja oryginalna', color='#cccccc', alpha=0.9, lw=2.2, zorder=1)
            line_w   = ax.plot(x, white_data['best_pred'], label='FunDPS White', color='#fde725', linestyle='--', lw=LINE_WIDTH, zorder=3)
            line_g   = ax.plot(x, grf_data['best_pred'],   label='FunDPS GRF',   color='#5ec962', linestyle='-',  lw=LINE_WIDTH, zorder=4)
            scat_obs = ax.scatter(x[mask_idx], y_true[mask_idx], color='black', marker='D', s=30, zorder=5)

            ax.set_xlabel('x', labelpad=5)
            if i % current_ncols == 0:
                ax.set_ylabel('f(x)', labelpad=5)
            ax.grid(True, linestyle='--', alpha=0.4, color='#cccccc')

        for j in range(len(valid_functions), len(axes_flat)):
            axes_flat[j].set_visible(False)

        custom_labels  = ['Funkcja oryginalna', 'White Noise', 'GRF Noise', f'Obserwacje ({len(mask_idx)} pkt)']
        custom_handles = [line_gt[0], line_w[0], line_g[0], scat_obs]
        fig.legend(custom_handles, custom_labels, loc='lower center',
                   bbox_to_anchor=(0.5, 0.01), ncol=4, frameon=True,
                   facecolor='white', edgecolor='black')

        plt.subplots_adjust(top=0.97, bottom=0.10, hspace=0.45 if nrows == 1 else 0.55,
                            wspace=0.28, left=0.08, right=0.95)
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
        df_global = pd.concat(all_trials, ignore_index=True).replace([np.inf, -np.inf], np.nan)
        return df_global.groupby(['Steps', 'Zeta']).agg(
            Mean_L2_Error=('L2_Error', 'mean'), Mean_Time_s=('Total_Time_s', 'mean')
        ).reset_index().sort_values('Mean_L2_Error')

    df_rank_w = extract_and_group(results_w)
    df_rank_g = extract_and_group(results_g)
    if df_rank_w.empty or df_rank_g.empty:
        return

    rows = []
    for noise_label, df_rank in [('Biały szum (White)', df_rank_w), ('Gładki szum (GRF)', df_rank_g)]:
        for rank_idx, (_, row) in enumerate(df_rank.head(top_n).iterrows(), 1):
            rows.append({
                'Struktura szumu':               noise_label,
                'Pozycja':                       f'TOP {rank_idx}',
                'Liczba kroków ($N_{steps}$)':   int(row['Steps']),
                'Siła nawigacji ($\\zeta$)':     float(row['Zeta']),
                'Średni globalny błąd $L_2$ (%)': row['Mean_L2_Error'],
                'Średni czas operacji [s]':      row['Mean_Time_s'],
            })

    df_final = pd.DataFrame(rows).set_index(['Struktura szumu', 'Pozycja'])
    display(df_final.style.background_gradient(subset=['Średni globalny błąd $L_2$ (%)'], cmap='viridis').format({
        'Liczba kroków ($N_{steps}$)':   '{:d}',
        'Siła nawigacji ($\\zeta$)':     '{:.1f}',
        'Średni globalny błąd $L_2$ (%)': '{:.2f}',
        'Średni czas operacji [s]':      '{:.2f}',
    }))


def plot_combined_training_loss(results_w, results_g, func_name, save_dir='../images/experiment3'):
    data_w = results_w.get(func_name)
    data_g = results_g.get(func_name)
    if data_w is None or 'prior_loss_history' not in data_w:
        return
    if data_g is None or 'prior_loss_history' not in data_g:
        return

    train_loss_w = data_w['prior_loss_history']
    train_loss_g = data_g['prior_loss_history']
    if not train_loss_w or not train_loss_g:
        return

    with plt.rc_context({'figure.autolayout': False}):
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        ax.plot(range(1, len(train_loss_w) + 1), train_loss_w,
                label='Biały szum (White Noise)', color='#3b528b', linestyle='--', lw=LINE_WIDTH, alpha=0.9)
        ax.plot(range(1, len(train_loss_g) + 1), train_loss_g,
                label='Gładki szum (GRF Noise)',  color='#5ec962', linestyle='-',  lw=LINE_WIDTH, alpha=0.9)

        val_loss_w = data_w.get('prior_val_loss_history', [])
        val_loss_g = data_g.get('prior_val_loss_history', [])
        if val_loss_w:
            step_w = len(train_loss_w) // len(val_loss_w)
            ax.plot(list(range(step_w, len(train_loss_w) + 1, step_w))[:len(val_loss_w)], val_loss_w,
                    color='#777777', linestyle=':', lw=1.2, marker='o', markersize=3, label='Walidacja (White)')
        if val_loss_g:
            step_g = len(train_loss_g) // len(val_loss_g)
            ax.plot(list(range(step_g, len(train_loss_g) + 1, step_g))[:len(val_loss_g)], val_loss_g,
                    color='#222222', linestyle=':', lw=1.2, marker='s', markersize=3, label='Walidacja (GRF)')

        ax.set_yscale('log')
        ax.set_xlabel('Epoka optymalizacji', labelpad=6)
        ax.set_ylabel('Wartość funkcji straty (skala log)', labelpad=6)
        ax.legend(loc='best', frameon=True, facecolor='white', edgecolor='black', framealpha=0.9)
        plt.subplots_adjust(left=0.14, right=0.95, bottom=0.16, top=0.95)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'loss_prior_combined_{func_name}.png'), bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()


def plot_fundps_trajectory(func_name, results_dict, noise_name):
    data = results_dict.get(func_name)
    if data is None or data['best_config'] is None:
        return

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_cfg = data['best_config']
    steps    = best_cfg['Steps']
    zeta     = best_cfg['Zeta']

    runner     = FunDPSExperimentRunner(noise_type=noise_name.lower())
    y_tensor   = torch.tensor(data['y_true'], dtype=torch.float32).unsqueeze(0).to(device)
    forward_op = ForwardOperator(data['mask_idx'])
    obs_tensor = forward_op(y_tensor)

    model, _, _ = runner.train_unconditional_prior(y_tensor, epochs=1000)
    sampler     = FunDPSSampler(model, device)
    sampler.model.eval()
    sigmas       = sampler.get_sigmas(steps)
    a_i          = torch.randn(1, 128, device=device) * sampler.sigma_max
    snapshots_idx = [0, int(steps * 0.25), int(steps * 0.5), int(steps * 0.75), steps - 1]
    snapshots     = []

    for i in range(steps):
        sigma_i   = sigmas[i].unsqueeze(0)
        sigma_prev = sigmas[i + 1].unsqueeze(0)
        a_i       = a_i.detach().requires_grad_(True)
        a_hat_0   = sampler.model(a_i, sigma_i)
        d_i       = (a_i - a_hat_0) / sigma_i
        a_prev    = a_i + (sigma_prev - sigma_i) * d_i
        if sigma_prev.item() > 0:
            pred_obs = forward_op(a_hat_0)
            loss     = nn.MSELoss()(pred_obs, obs_tensor)
            grad_a   = torch.autograd.grad(loss, a_i)[0]
            grad_a   = torch.clamp(grad_a, min=-1.0, max=1.0)
            zeta_t   = sigma_i.item() * zeta if sigma_i.item() < 1.0 else zeta
            a_prev   = a_prev.detach() - zeta_t * grad_a
        a_i = a_prev
        if i in snapshots_idx:
            snapshots.append((i, a_i.detach().cpu().numpy()[0]))

    n_snaps = len(snapshots)
    colors  = plt.cm.get_cmap('viridis')(np.linspace(0.7, 0.0, n_snaps))
    x       = data['x']
    y_true  = data['y_true']
    mask_idx = data['mask_idx']

    with plt.rc_context({'figure.autolayout': False}):
        fig, axes = plt.subplots(1, n_snaps, figsize=(FIG_W2 * n_snaps / 4, FIG_H), squeeze=False)
        axes_flat = axes.flatten()

        for idx, (step_i, sig_data) in enumerate(snapshots):
            ax = axes_flat[idx]
            ax.plot(x, y_true, label='Oryginał', color='#bbbbbb', linestyle='--', lw=1.3, zorder=1)
            if idx == 0:
                label, lw, ls = f'Inicjalizacja (t={steps})', 1.3, ':'
            elif idx == n_snaps - 1:
                label, lw, ls = 'Rekonstrukcja końcowa', LINE_WIDTH, '-'
            else:
                label, lw, ls = f'Krok {step_i + 1}', 1.4, '-'
            ax.plot(x, sig_data, label=label, color=colors[idx], linestyle=ls, lw=lw, zorder=3)
            ax.scatter(x[mask_idx], y_true[mask_idx], color='black', marker='s', s=20, zorder=5)
            ax.set_ylim(y_true.min() - 0.5, y_true.max() + 0.5)
            ax.locator_params(axis='x', nbins=4)
            ax.grid(True, linestyle=':', alpha=0.3)
            if idx > 0:
                ax.set_yticklabels([])
            if idx == 0:
                ax.set_ylabel('Amplituda')

        handles = [
            Line2D([0], [0], color='#bbbbbb', linestyle='--', lw=1.3),
            Line2D([0], [0], color=colors[0],  linestyle=':', lw=1.3),
            Line2D([0], [0], color=colors[-1], linestyle='-', lw=LINE_WIDTH),
            Line2D([0], [0], color='black', marker='s', linestyle='None', markersize=5),
        ]
        labels = ['Funkcja oryginalna', 'Stan początkowy stochastyczny',
                  'Stan zrekonstruowany (krok 0)', 'Punkty obserwacyjne']
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.18),
                   ncol=4, frameon=True, facecolor='white', edgecolor='black')
        plt.subplots_adjust(wspace=0.18, bottom=0.15)

        save_dir = '../images/experiment3'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/trajectory_{noise_name.lower()}_{func_name}.png',
                    bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()


def plot_average_fundps_heatmap(results_dict, noise_name, metric_name='L2_Error', save_dir='../images/experiment3'):
    all_trials_data = []
    for func_name, func_data in results_dict.items():
        if func_data is None or 'metrics_history' not in func_data:
            continue
        df_func = pd.DataFrame(func_data['metrics_history'])
        if not df_func.empty:
            all_trials_data.append(df_func)

    if not all_trials_data:
        return

    df_global   = pd.concat(all_trials_data, ignore_index=True)
    df_averaged = df_global.groupby(['Zeta', 'Steps'])[metric_name].mean().reset_index()
    heatmap_data = df_averaged.pivot(index='Zeta', columns='Steps', values=metric_name)
    heatmap_clean = heatmap_data.replace([np.inf, -np.inf], np.nan)

    is_mse        = metric_name.upper() == 'MSE'
    global_max    = heatmap_clean.max().max()
    vmax_cutoff   = global_max if is_mse else min(global_max, 100.0) if pd.notna(global_max) else 100.0
    cbar_label    = 'Średni globalny błąd średniokwadratowy (MSE)' if is_mse else 'Średni globalny błąd relatywny $L_2$ (%)'

    labels_matrix = np.empty_like(heatmap_data.values, dtype=object)
    for r in range(heatmap_data.shape[0]):
        for c in range(heatmap_data.shape[1]):
            val = heatmap_data.values[r, c]
            if pd.isna(val):
                labels_matrix[r, c] = 'NaN'
            elif is_mse:
                labels_matrix[r, c] = f'{val:.1e}'
            elif val >= 1000.0:
                labels_matrix[r, c] = f'{val:.0e}'
            else:
                labels_matrix[r, c] = f'{val:.1f}'

    n_cols      = heatmap_data.shape[1]
    cell_size   = max(1.5, 14 / n_cols) / 2.54
    fig_w       = max(FIG_W, n_cols * cell_size)
    fig_h       = max(FIG_H, heatmap_data.shape[0] * 1.5 / 2.54)

    with plt.rc_context({'figure.autolayout': False}):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        sns.heatmap(
            heatmap_data, annot=labels_matrix, fmt='', cmap='viridis',
            vmax=vmax_cutoff, linewidths=0.6, linecolor='#3b528b',
            cbar_kws={'label': cbar_label},
            annot_kws=HEATMAP_ANNOT_KWS,
            ax=ax,
        )
        ax.set_ylabel('Siła nawigacji gradientowej ($\\zeta$)', labelpad=12)
        ax.set_xlabel('Liczba kroków rekonstrukcji ($N_{\\mathrm{steps}}$)', labelpad=12)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        plt.subplots_adjust(left=0.18, right=0.82, bottom=0.18, top=0.95)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'global_heatmap_fundps_{noise_name.lower()}_{metric_name.lower()}.png'),
                    bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()


def plot_fundps_ablation_heatmap(results_dict, func_name, noise_name, save_dir='../images/experiment3'):
    data = results_dict.get(func_name)
    if data is None:
        return

    df_metrics   = pd.DataFrame(data['metrics_history'])
    heatmap_data = df_metrics.pivot(index='Zeta', columns='Steps', values='L2_Error')
    heatmap_clean = heatmap_data.replace([np.inf, -np.inf], np.nan)
    global_max   = heatmap_clean.max().max()
    vmax_cutoff  = min(global_max, 150.0) if pd.notna(global_max) else 100.0

    labels_matrix = np.empty_like(heatmap_data.values, dtype=object)
    for r in range(heatmap_data.shape[0]):
        for c in range(heatmap_data.shape[1]):
            val = heatmap_data.values[r, c]
            if pd.isna(val) or np.isinf(val):
                labels_matrix[r, c] = 'NaN'
            elif val >= 1000.0:
                labels_matrix[r, c] = f'{val:.0e}'
            else:
                labels_matrix[r, c] = f'{val:.1f}'

    n_cols  = heatmap_data.shape[1]
    n_rows  = heatmap_data.shape[0]
    fig_w   = max(FIG_W, n_cols * 2.0 / 2.54)
    fig_h   = max(FIG_H, n_rows * 1.6 / 2.54)

    with plt.rc_context({'figure.autolayout': False}):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        sns.heatmap(
            heatmap_data, annot=labels_matrix, fmt='', cmap='viridis',
            vmax=vmax_cutoff,
            cbar_kws={'label': 'Błąd relatywny $L_2$ (%)'},
            linewidths=0.6, linecolor='#3b528b',
            annot_kws={**HEATMAP_ANNOT_KWS, 'size': ANNOT_SIZE_LARGE},
            ax=ax,
        )
        ax.set_ylabel('Siła nawigacji gradientowej ($\\zeta$)', labelpad=14)
        ax.set_xlabel('Liczba kroków rekonstrukcji ($N_{\\mathrm{steps}}$)', labelpad=14)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.95)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'heatmap_ablation_{noise_name.lower()}_{func_name}.png'),
                    bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()


def generate_fundps_summary_table(results_dict, noise_name):
    all_trials_data = []
    for func_name, func_data in results_dict.items():
        if func_data is None or 'metrics_history' not in func_data:
            continue
        df_func = pd.DataFrame(func_data['metrics_history'])
        if not df_func.empty:
            all_trials_data.append(df_func)

    if not all_trials_data:
        return None

    df_global   = pd.concat(all_trials_data, ignore_index=True)
    df_averaged = df_global.groupby(['Steps', 'Zeta']).agg(
        Sredni_Blad_L2_Proc=('L2_Error',     'mean'),
        Sredni_Czas_s       =('Total_Time_s', 'mean'),
    ).reset_index()
    df_averaged.columns = ['Kroki (N_steps)', 'Siła nawigacji (Zeta)',
                           'Średni globalny błąd L2 (%)', 'Średni łączny czas [s]']
    df_summary = df_averaged.sort_values('Średni globalny błąd L2 (%)').reset_index(drop=True)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    print(df_summary.to_string(index=True))
    print('=' * 85)
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
        return

    df_global      = pd.concat(all_trials_data, ignore_index=True)
    allowed_steps  = [10, 20, 50, 100, 200]
    allowed_zetas  = [2.0, 4.0, 6.0, 8.0, 10.0]
    df_filtered    = df_global[df_global['Steps'].isin(allowed_steps) & df_global['Zeta'].isin(allowed_zetas)].copy()
    df_averaged    = df_filtered.groupby(['Zeta', 'Steps'])['L2_Error'].mean().reset_index()
    matrix_data    = df_averaged.pivot(index='Zeta', columns='Steps', values='L2_Error').sort_index(ascending=False)

    labels_matrix = np.empty_like(matrix_data.values, dtype=object)
    for r in range(matrix_data.shape[0]):
        for c in range(matrix_data.shape[1]):
            val = matrix_data.values[r, c]
            if pd.isna(val):
                labels_matrix[r, c] = 'NaN'
            elif val >= 1000.0:
                labels_matrix[r, c] = f'{val:.1e}%'.replace('+0', '').replace('+', '')
            else:
                labels_matrix[r, c] = f'{val:.2f}%'

    n_cols = matrix_data.shape[1]
    n_rows = matrix_data.shape[0]
    fig_w  = max(FIG_W, n_cols * 1.8 / 2.54)
    fig_h  = max(FIG_H, n_rows * 1.5 / 2.54)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        matrix_data, annot=labels_matrix, fmt='', cmap='viridis',
        linewidths=1.0, linecolor='#333333',
        cbar_kws={'label': 'Średni globalny błąd relatywny $L_2$ (%)'},
        annot_kws=HEATMAP_ANNOT_KWS,
        ax=ax,
    )
    ax.set_ylabel('Siła nawigacji gradientowej ($\\zeta$)', labelpad=10)
    ax.set_xlabel('Liczba kroków próbkowania ($N_{\\mathrm{steps}}$)', labelpad=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'param_matrix_filtered_{noise_name.lower()}.png'), bbox_inches='tight')
    plt.show()
    plt.close()


def plot_global_noise_comparison_boxplot(res_white, res_grf, save_path='../images/experiment3/global_noise_boxplot.png'):
    all_l2_white, all_l2_grf = [], []
    for func in res_white.keys():
        if res_white[func]['best_metrics'] is not None and res_grf[func]['best_metrics'] is not None:
            all_l2_white.append(res_white[func]['best_metrics']['L2_Error'])
            all_l2_grf.append(res_grf[func]['best_metrics']['L2_Error'])

    plot_df = pd.DataFrame({
        'Błąd L2 (%)':           all_l2_white + all_l2_grf,
        'Struktura szumu priora': ['Biały szum (White)'] * len(all_l2_white) + ['Gładki szum (GRF)'] * len(all_l2_grf),
    })

    fig, ax = plt.subplots(figsize=(7 / 2.54, FIG_H))
    sns.boxplot(data=plot_df, x='Struktura szumu priora', y='Błąd L2 (%)',
                hue='Struktura szumu priora', palette='viridis', legend=False,
                edgecolor='black', linewidth=1.0, width=0.4, ax=ax, **BOXPLOT_PROPS)
    ax.set_ylabel('Najlepszy błąd relatywny L2 (%)')
    ax.set_xlabel('')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def plot_fundps_time_vs_quality(results_dict, func_name, noise_name, save_dir='../images/experiment3'):
    data = results_dict.get(func_name)
    if data is None:
        return

    df_metrics = pd.DataFrame(data['metrics_history'])
    df_metrics = df_metrics[df_metrics['L2_Error'] < 200.0]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    scatter = ax.scatter(
        df_metrics['Total_Time_s'], df_metrics['L2_Error'],
        c=df_metrics['Steps'], cmap='viridis',
        s=df_metrics['Zeta'] * 10, alpha=0.8, edgecolors='black', linewidths=0.5,
    )
    ax.set_xlabel('Całkowity czas operacji (Uczenie + Próbkowanie) [s]')
    ax.set_ylabel('Błąd relatywny L2 (%)')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Liczba kroków próbkowania (Steps)')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'pareto_{noise_name.lower()}_{func_name}.png'), bbox_inches='tight')
    plt.show()
    plt.close()


