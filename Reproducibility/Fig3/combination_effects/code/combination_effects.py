import os

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

try:
    import colorcet as cc
    GLASBEY = cc.glasbey_light
except ImportError:
    GLASBEY = 'tab20'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.path.dirname(SCRIPT_DIR)


def find_repo_root():
    current = SCRIPT_DIR
    while True:
        if os.path.exists(os.path.join(current, 'README.md')) and os.path.exists(os.path.join(current, 'model.py')):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return os.path.abspath(os.path.join(MODULE_DIR, '..', '..', '..'))
        current = parent


REPO_ROOT = find_repo_root()
RAW_DATA_DIR = os.path.join(REPO_ROOT, 'raw_data', 'fig3', 'combination_effects')
OUTPUT_DIR = os.path.join(MODULE_DIR, 'output_plot')
os.makedirs(OUTPUT_DIR, exist_ok=True)

NODE_SIZE = 28
CELL_TYPE_COLORS = {'A549': '#46C3DB', 'MCF7': '#FF7517'}
PRED_COORD_FILE = 'sciplex4_predict_tsne_result.csv'
REAL_COORD_FILE = 'sciplex4_real_tsne_result.csv'
COLOR_TASKS = [
    ('cell_type', 'Cell type', 'categorical'),
    ('drug1_name', 'First drug identity', 'categorical'),
    ('drug1_dose', 'First drug dose', 'dose'),
    ('drug2_name', 'Second drug identity', 'categorical'),
    ('drug2_dose', 'Second drug dose', 'dose'),
]
SCENARIOS = [(0, '0/2 Unseen'), (1, '1/2 Unseen'), (2, '2/2 Unseen')]
MODEL_PALETTE = {'UniCure': '#E64B35', 'Additive Baseline': '#4DBBD5'}

# Upstream coordinate-generation workflow:
#   1. Use predicted and observed sci-Plex4 dual-drug expression matrices with
#      cell type, first/second drug identity, and first/second dose metadata.
#   2. Traverse PCA/t-SNE parameter grids to generate candidate two-dimensional
#      layouts for predicted and observed combination-response profiles.
#   3. Save candidate coordinate tables and select representative layouts for
#      Fig. 3G and Fig. S11A-E.
# The selected coordinate tables are released as the inputs read below.
#
# Example upstream code:
# for source_name, source_df in [("predict", predictions_df), ("real", real_df)]:
#     meta_info = source_df[["cell_type", "drug1_name", "drug1_dose", "drug2_name", "drug2_dose"]].copy()
#     expression_values = source_df.iloc[:, 5:]
#     for pca_config in PCA_PARAMETER_GRID:
#         pca_input = run_pca_preprocessing(expression_values, pca_config)
#         for tsne_config in TSNE_PARAMETER_GRID:
#             coords = run_tsne_projection(pca_input, tsne_config)
#             out = meta_info.copy()
#             out[["TSNE1", "TSNE2"]] = coords
#             out.to_csv(
#                 f"sciplex4_{source_name}_tsne_candidate_{pca_config}_{tsne_config}.csv",
#                 index=False,
#             )
#
# Fig. S11F statistical workflow:
#   - Read scenario-specific UniCure and additive-baseline correlation tables.
#   - Compare model distributions within each unseen-combination scenario using
#     a one-sided Mann-Whitney U test with UniCure expected to be greater.


def read_coordinates(filename):
    path = os.path.join(RAW_DATA_DIR, filename)
    df = pd.read_csv(path)
    required = {'cell_type', 'drug1_name', 'drug1_dose', 'drug2_name', 'drug2_dose', 'TSNE1', 'TSNE2'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'{filename} is missing required columns: {sorted(missing)}')
    return df


def categorical_palette(series, preset=None):
    categories = sorted(series.dropna().astype(str).unique())
    if preset:
        return {cat: preset.get(cat, '#808080') for cat in categories}
    return dict(zip(categories, sns.color_palette(GLASBEY, n_colors=len(categories))))


def plot_categorical(ax, df, column, title, palette=None, legend=True):
    plot_df = df.copy()
    plot_df[column] = plot_df[column].astype(str)
    if palette is None:
        palette = categorical_palette(plot_df[column])
    sns.scatterplot(
        data=plot_df,
        x='TSNE1',
        y='TSNE2',
        hue=column,
        palette=palette,
        s=NODE_SIZE,
        alpha=0.95,
        linewidth=0,
        legend=legend,
        ax=ax,
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('tSNE 1')
    ax.set_ylabel('tSNE 2')
    if legend and ax.legend_ is not None:
        ax.legend(title='', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, fontsize=8)


def positive_log_norm(values):
    positive = values[values > 0]
    if positive.empty:
        return None
    return matplotlib.colors.LogNorm(vmin=positive.min(), vmax=positive.max())


def plot_dose(ax, df, column, title, add_colorbar=True):
    values = pd.to_numeric(df[column], errors='coerce')
    norm = positive_log_norm(values)
    color_values = values.copy()
    if norm is not None:
        positive_min = values[values > 0].min()
        color_values = color_values.mask(color_values <= 0, positive_min)

    scatter = ax.scatter(
        df['TSNE1'],
        df['TSNE2'],
        c=color_values,
        cmap='viridis',
        norm=norm,
        s=NODE_SIZE,
        alpha=0.95,
        linewidth=0,
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('tSNE 1')
    ax.set_ylabel('tSNE 2')
    if add_colorbar:
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Dose')


def plot_single_panel(df, column, title, output_stem):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    task_type = 'dose' if 'dose' in column else 'categorical'
    if task_type == 'dose':
        plot_dose(ax, df, column, title)
    else:
        preset = CELL_TYPE_COLORS if column == 'cell_type' else None
        plot_categorical(ax, df, column, title, categorical_palette(df[column], preset=preset))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{output_stem}.png'), dpi=300, transparent=True)
    plt.savefig(os.path.join(OUTPUT_DIR, f'{output_stem}.pdf'), transparent=True)
    plt.close()
    print(f'  Saved {output_stem}')


def plot_fig3g_prediction(df_pred):
    fig, axes = plt.subplots(1, 5, figsize=(28, 5.5))
    for ax, (column, title, task_type) in zip(axes, COLOR_TASKS):
        if task_type == 'dose':
            plot_dose(ax, df_pred, column, title, add_colorbar=True)
        else:
            preset = CELL_TYPE_COLORS if column == 'cell_type' else None
            palette = categorical_palette(df_pred[column], preset=preset)
            plot_categorical(ax, df_pred, column, title, palette=palette, legend=False)
    plt.tight_layout()
    output_stem = 'fig3g_sciplex4_prediction_combination_effects'
    plt.savefig(os.path.join(OUTPUT_DIR, f'{output_stem}.png'), dpi=300, transparent=True)
    plt.savefig(os.path.join(OUTPUT_DIR, f'{output_stem}.pdf'), transparent=True)
    plt.close()
    print(f'  Saved {output_stem}')


def plot_figs8_real(df_real):
    panel_map = {
        'cell_type': ('Fig. S8A real sci-Plex4 cells by cell type', 'figs8a_sciplex4_real_cell_type'),
        'drug1_name': ('Fig. S8B real sci-Plex4 cells by first drug identity', 'figs8b_sciplex4_real_drug1_name'),
        'drug1_dose': ('Fig. S8C real sci-Plex4 cells by first drug dose', 'figs8c_sciplex4_real_drug1_dose'),
        'drug2_name': ('Fig. S8D real sci-Plex4 cells by second drug identity', 'figs8d_sciplex4_real_drug2_name'),
        'drug2_dose': ('Fig. S8E real sci-Plex4 cells by second drug dose', 'figs8e_sciplex4_real_drug2_dose'),
    }
    for column, (title, output_stem) in panel_map.items():
        plot_single_panel(df_real, column, title, output_stem)


def read_correlation_table(path, model, scenario):
    df = pd.read_csv(path)
    required = {'cell_type', 'drug1_name', 'drug1_dose', 'drug2_name', 'drug2_dose', 'correlation'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'{os.path.basename(path)} is missing required columns: {sorted(missing)}')
    df = df.copy()
    df['correlation'] = pd.to_numeric(df['correlation'], errors='coerce')
    df = df.dropna(subset=['correlation'])
    df['Model'] = model
    df['Scenario'] = scenario
    return df


def load_s8f_data():
    frames = []
    for index, scenario in SCENARIOS:
        unicure_path = os.path.join(RAW_DATA_DIR, f'sciplex4_unseen_correlations_{index}.csv')
        baseline_path = os.path.join(RAW_DATA_DIR, f'sciplex4_unseen_correlations_additive_baseline_{index}.csv')
        frames.append(read_correlation_table(unicure_path, 'UniCure', scenario))
        frames.append(read_correlation_table(baseline_path, 'Additive Baseline', scenario))
    return pd.concat(frames, ignore_index=True)


def p_to_stars(p_value):
    if p_value < 0.0001:
        return '****'
    if p_value < 0.001:
        return '***'
    if p_value < 0.01:
        return '**'
    if p_value < 0.05:
        return '*'
    return 'ns'


def add_stat_annotations(ax, df):
    y_range = df['correlation'].max() - df['correlation'].min()
    offset = max(0.025, y_range * 0.08)
    line_h = max(0.008, y_range * 0.025)
    for i, (_, scenario) in enumerate(SCENARIOS):
        unicure = df[(df['Scenario'] == scenario) & (df['Model'] == 'UniCure')]['correlation'].values
        baseline = df[(df['Scenario'] == scenario) & (df['Model'] == 'Additive Baseline')]['correlation'].values
        if len(unicure) == 0 or len(baseline) == 0:
            continue
        stat = mannwhitneyu(unicure, baseline, alternative='greater')
        star = p_to_stars(stat.pvalue)
        current_max = max(unicure.max(), baseline.max())
        y_line = current_max + offset
        x1 = i - 0.2
        x2 = i + 0.2
        ax.plot([x1, x1, x2, x2], [y_line, y_line + line_h, y_line + line_h, y_line], lw=1.2, color='black')
        ax.text((x1 + x2) / 2, y_line + line_h, star, ha='center', va='bottom', color='black', fontsize=13, fontweight='bold')


def plot_s8f(df):
    sns.set_theme(style='ticks', font_scale=1.1)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    sns.boxplot(
        data=df,
        x='Scenario',
        y='correlation',
        hue='Model',
        palette=MODEL_PALETTE,
        order=[scenario for _, scenario in SCENARIOS],
        hue_order=['UniCure', 'Additive Baseline'],
        width=0.62,
        showfliers=False,
        linewidth=1.4,
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x='Scenario',
        y='correlation',
        hue='Model',
        order=[scenario for _, scenario in SCENARIOS],
        hue_order=['UniCure', 'Additive Baseline'],
        dodge=True,
        palette={'UniCure': 'black', 'Additive Baseline': 'black'},
        alpha=0.32,
        size=3,
        ax=ax,
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title='', loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    add_stat_annotations(ax, df)
    y_min = df['correlation'].min()
    y_max = df['correlation'].max()
    ax.set_ylim(max(0, y_min - 0.08), min(1.05, y_max + 0.12))
    ax.set_xlabel('')
    ax.set_ylabel('Pearson Correlation')
    ax.set_title('Predictive Performance on Unseen Drug Combinations', pad=36, fontsize=14)
    sns.despine(trim=False)
    plt.tight_layout()
    output_stem = 'figs8f_sciplex4_unseen_evaluation'
    plt.savefig(os.path.join(OUTPUT_DIR, f'{output_stem}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, f'{output_stem}.pdf'), bbox_inches='tight')
    plt.close()
    print(f'  Saved {output_stem}')


def main():
    print('=' * 60)
    print('Fig. 3G / Fig. S8 sci-Plex4 combination effects')
    print('=' * 60)
    df_pred = read_coordinates(PRED_COORD_FILE)
    df_real = read_coordinates(REAL_COORD_FILE)
    print(f'Predicted coordinates: {df_pred.shape[0]} rows')
    print(f'Real coordinates: {df_real.shape[0]} rows')
    plot_fig3g_prediction(df_pred)
    plot_figs8_real(df_real)
    s8f_df = load_s8f_data()
    plot_s8f(s8f_df)
    print(f'Output directory: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
