"""Generate clean, simple charts for the experimental report."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.dpi': 150,
})


def plot_rmse_comparison():
    models = [
        'Core Model\nLGB+RF Blend',
        'Baseline\nRandom Forest',
        'Baseline\nLightGBM',
        'Baseline\nSimple Linear',
        'Baseline\nXGBoost',
    ]
    rmse = [1.9596, 2.0778, 2.0798, 2.6077, 3.9751]
    colors = ['#1a73e8', '#9e9e9e', '#9e9e9e', '#9e9e9e', '#9e9e9e']

    fig, ax = plt.subplots(figsize=(10, 4.5))
    y_pos = range(len(models))
    bars = ax.barh(y_pos, rmse, color=colors, edgecolor='white', height=0.6)

    for bar, v in zip(bars, rmse):
        ax.text(v + 0.05, bar.get_y() + bar.get_height()/2, f'{v:.4f}', va='center', fontsize=11)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel('RMSE')
    ax.set_xlim(0, 4.6)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    fig.savefig(OUT_DIR / 'fig1_rmse_comparison.png')
    plt.close(fig)
    print("Saved: fig1_rmse_comparison.png")


def plot_r2_comparison():
    models = [
        'Core Model LGB+RF Blend',
        'Baseline Random Forest',
        'Baseline LightGBM',
        'Baseline Simple Linear',
        'Baseline XGBoost',
    ]
    r2 = [0.9942, 0.9935, 0.9935, 0.9897, 0.9762]
    colors = ['#1a73e8', '#9e9e9e', '#9e9e9e', '#9e9e9e', '#9e9e9e']

    fig, ax = plt.subplots(figsize=(10, 4.5))
    y_pos = range(len(models))
    bars = ax.barh(y_pos, r2, color=colors, edgecolor='white', height=0.6)

    for bar, v in zip(bars, r2):
        ax.text(v + 0.0005, bar.get_y() + bar.get_height()/2, f'{v:.4f}', va='center', fontsize=11)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel('R²')
    ax.set_xlim(0.97, 0.996)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    fig.savefig(OUT_DIR / 'fig2_r2_comparison.png')
    plt.close(fig)
    print("Saved: fig2_r2_comparison.png")


def plot_ablation_waterfall():
    steps = [
        'Full Model\n(opt LGB + RF)',
        '− Hyperparam\nTuning',
        '− Blending\n(LGB alone)',
        '− Both\n(baseline LGB)',
    ]
    rmse_vals = [1.9596, 1.9788, 2.0229, 2.0798]
    colors = ['#1a73e8', '#6db3f2', '#f9ab00', '#ea4335']

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(steps))
    bars = ax.bar(x, rmse_vals, color=colors, edgecolor='white', width=0.55)

    for bar, v in zip(bars, rmse_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.015, f'{v:.4f}',
                ha='center', fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(steps, fontsize=10)
    ax.set_ylabel('RMSE')
    ax.set_ylim(1.80, 2.20)
    ax.grid(axis='y', alpha=0.3)

    fig.savefig(OUT_DIR / 'fig3_ablation_waterfall.png')
    plt.close(fig)
    print("Saved: fig3_ablation_waterfall.png")


def plot_feature_comparison():
    feature_sets = ['Baseline\n(61 dims)', 'Combo\n(73 dims)',
                    'Enhanced\nno-route (69 dims)', 'Enhanced\nfull (20,115 dims)']
    rmse_vals = [2.0229, 2.0745, 2.0962, 2.1054]
    colors = ['#34a853', '#f9ab00', '#f9ab00', '#ea4335']

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(range(len(feature_sets)), rmse_vals, color=colors, edgecolor='white', width=0.5)
    for bar, v in zip(bars, rmse_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f'{v:.4f}', ha='center', fontsize=11)
    ax.set_xticks(range(len(feature_sets)))
    ax.set_xticklabels(feature_sets)
    ax.set_ylabel('RMSE')
    ax.set_ylim(1.98, 2.14)
    ax.grid(axis='y', alpha=0.3)

    fig.savefig(OUT_DIR / 'fig4_feature_comparison.png')
    plt.close(fig)
    print("Saved: fig4_feature_comparison.png")


def plot_training_strategy():
    strategies = ['Full Training\n(160 rounds)', 'Validation Split\n+ Early Stopping', '5-Fold CV\nAverage']
    rmse_vals = [2.0229, 2.0311, 2.1346]
    colors = ['#34a853', '#f9ab00', '#ea4335']

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(range(len(strategies)), rmse_vals, color=colors, edgecolor='white', width=0.45)
    for bar, v in zip(bars, rmse_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f'{v:.4f}', ha='center', fontsize=12)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies)
    ax.set_ylabel('RMSE')
    ax.set_ylim(1.95, 2.20)
    ax.grid(axis='y', alpha=0.3)

    fig.savefig(OUT_DIR / 'fig5_training_strategy.png')
    plt.close(fig)
    print("Saved: fig5_training_strategy.png")


def plot_blending_comparison():
    blends = [
        'LGB×0.30\n+RF×0.70',
        'LGB×0.40\n+RF×0.60',
        'LGB×0.50\n+RF×0.50',
        'LGB×0.55\n+RF×0.45',
        'LGB×0.60\n+RF×0.40',
        'LGB×0.65\n+RF×0.35',
    ]
    rmse_vals = [1.9907, 1.9729, 1.9625, 1.9601, 1.9596, 1.9610]
    colors = ['#6db3f2', '#6db3f2', '#6db3f2', '#6db3f2', '#1a73e8', '#6db3f2']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(blends)), rmse_vals, color=colors, edgecolor='white', width=0.5)
    for bar, v in zip(bars, rmse_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.002, f'{v:.4f}', ha='center', fontsize=11)
    ax.set_xticks(range(len(blends)))
    ax.set_xticklabels(blends, fontsize=9)
    ax.set_ylabel('RMSE')
    ax.set_ylim(1.93, 2.01)
    ax.grid(axis='y', alpha=0.3)

    fig.savefig(OUT_DIR / 'fig6_blending_comparison.png')
    plt.close(fig)
    print("Saved: fig6_blending_comparison.png")


def plot_overview():
    fig = plt.figure(figsize=(14, 8))

    # Subplot 1: RMSE ranking
    ax1 = fig.add_subplot(2, 3, (1, 2))
    models_short = ['LGB+RF\nBlend', 'RF\nBaseline', 'LGB\nBaseline',
                    'Linear\nBaseline', 'XGBoost\nBaseline']
    rmse = [1.9596, 2.0778, 2.0798, 2.6077, 3.9751]
    colors1 = ['#1a73e8', '#9e9e9e', '#9e9e9e', '#9e9e9e', '#9e9e9e']
    bars = ax1.bar(range(len(models_short)), rmse, color=colors1, edgecolor='white')
    for bar, v in zip(bars, rmse):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.05, f'{v:.3f}', ha='center', fontsize=10)
    ax1.set_xticks(range(len(models_short)))
    ax1.set_xticklabels(models_short, fontsize=8)
    ax1.set_ylabel('RMSE')
    ax1.set_title('RMSE Ranking')
    ax1.set_ylim(1.7, 4.3)
    ax1.grid(axis='y', alpha=0.3)

    # Subplot 2: Leave-one-out ablation
    ax2 = fig.add_subplot(2, 3, 3)
    steps = ['Full\nModel', '− HP\nTuning', '− Blending', '− Both']
    vals = [1.9596, 1.9788, 2.0229, 2.0798]
    colors2 = ['#1a73e8', '#6db3f2', '#f9ab00', '#ea4335']
    bars2 = ax2.bar(range(len(steps)), vals, color=colors2, edgecolor='white', width=0.5)
    for bar, v in zip(bars2, vals):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.015, f'{v:.4f}', ha='center', fontsize=10)
    ax2.set_xticks(range(len(steps)))
    ax2.set_xticklabels(steps, fontsize=9)
    ax2.set_ylabel('RMSE')
    ax2.set_title('Leave-One-Out Ablation')
    ax2.set_ylim(1.80, 2.20)
    ax2.grid(axis='y', alpha=0.3)

    # Subplot 3: Feature set
    ax3 = fig.add_subplot(2, 3, 4)
    feat_names = ['61', '69', '73', '20,115']
    feat_rmse = [2.0229, 2.0962, 2.0745, 2.1054]
    colors3 = ['#34a853', '#f9ab00', '#f9ab00', '#ea4335']
    bars3 = ax3.bar(range(len(feat_names)), feat_rmse, color=colors3, edgecolor='white', width=0.45)
    for bar, v in zip(bars3, feat_rmse):
        ax3.text(bar.get_x() + bar.get_width()/2, v + 0.005, f'{v:.3f}', ha='center', fontsize=10)
    ax3.set_xticks(range(len(feat_names)))
    ax3.set_xticklabels(feat_names)
    ax3.set_xlabel('Feature Dimension')
    ax3.set_ylabel('RMSE')
    ax3.set_title('Feature Set Comparison')
    ax3.set_ylim(1.98, 2.14)
    ax3.grid(axis='y', alpha=0.3)

    # Subplot 4: Training strategy
    ax4 = fig.add_subplot(2, 3, 5)
    strat_names = ['Full\n160r', 'Valid\n+ES', '5-Fold\nCV']
    strat_rmse = [2.0229, 2.0311, 2.1346]
    colors4 = ['#34a853', '#f9ab00', '#ea4335']
    bars4 = ax4.bar(range(len(strat_names)), strat_rmse, color=colors4, edgecolor='white', width=0.4)
    for bar, v in zip(bars4, strat_rmse):
        ax4.text(bar.get_x() + bar.get_width()/2, v + 0.005, f'{v:.3f}', ha='center', fontsize=10)
    ax4.set_xticks(range(len(strat_names)))
    ax4.set_xticklabels(strat_names, fontsize=9)
    ax4.set_ylabel('RMSE')
    ax4.set_title('Training Strategy')
    ax4.set_ylim(1.98, 2.18)
    ax4.grid(axis='y', alpha=0.3)

    # Subplot 5: Blending
    ax5 = fig.add_subplot(2, 3, 6)
    blend_names = ['30/70', '40/60', '50/50', '55/45', '60/40', '65/35']
    blend_rmse = [1.9907, 1.9729, 1.9625, 1.9601, 1.9596, 1.9610]
    colors5 = ['#6db3f2', '#6db3f2', '#6db3f2', '#6db3f2', '#1a73e8', '#6db3f2']
    bars5 = ax5.bar(range(len(blend_names)), blend_rmse, color=colors5, edgecolor='white', width=0.4)
    for bar, v in zip(bars5, blend_rmse):
        ax5.text(bar.get_x() + bar.get_width()/2, v + 0.003, f'{v:.3f}', ha='center', fontsize=9)
    ax5.set_xticks(range(len(blend_names)))
    ax5.set_xticklabels(blend_names, fontsize=8)
    ax5.set_ylabel('RMSE')
    ax5.set_title('Blending Strategy')
    ax5.set_ylim(1.93, 2.05)
    ax5.grid(axis='y', alpha=0.3)

    fig.savefig(OUT_DIR / 'fig7_overview.png')
    plt.close(fig)
    print("Saved: fig7_overview.png")


def plot_improvement():
    models = [
        'Core Model\nLGB+RF Blend',
        'Baseline\nRandom Forest',
        'Baseline\nLightGBM',
        'Baseline\nSimple Linear',
        'Baseline\nXGBoost',
    ]
    rmse = [1.9596, 2.0778, 2.0798, 2.6077, 3.9751]
    best_baseline = 2.0778
    improvements = [(best_baseline - v) / best_baseline * 100 for v in rmse]
    colors = ['#1a73e8', '#9e9e9e', '#9e9e9e', '#9e9e9e', '#9e9e9e']

    fig, ax = plt.subplots(figsize=(9, 4.5))
    y_pos = range(len(models))
    bars = ax.barh(y_pos, improvements, color=colors, edgecolor='white', height=0.55)

    for bar, imp in zip(bars, improvements):
        label = f' {imp:+.1f}%'
        x_pos = imp + 0.8 if imp >= 0 else imp - 8
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, label, va='center', fontsize=12)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel('Improvement over Best Baseline (%)')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    fig.savefig(OUT_DIR / 'fig8_improvement.png')
    plt.close(fig)
    print("Saved: fig8_improvement.png")


if __name__ == '__main__':
    plot_rmse_comparison()
    plot_r2_comparison()
    plot_ablation_waterfall()
    plot_feature_comparison()
    plot_training_strategy()
    plot_blending_comparison()
    plot_overview()
    plot_improvement()
    print(f"\nAll figures saved to: {OUT_DIR}")
