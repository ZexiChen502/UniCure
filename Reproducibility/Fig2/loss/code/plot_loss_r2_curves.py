from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_DIR = SCRIPT_DIR.parent


def find_repo_root() -> Path:
    for parent in SCRIPT_DIR.parents:
        if (parent / "README.md").exists() and (parent / "model.py").exists():
            return parent
    return MODULE_DIR.parents[2]


REPO_ROOT = find_repo_root()
RAW = REPO_ROOT / 'raw_data' / 'fig2' / 'loss'
OUT = MODULE_DIR / 'output_plot'
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    'lincs_training_results.csv': 'LINCS 2020',
    'ucelora_training_results.csv': 'UCE LoRA',
    'sciplex3_training_results.csv': 'SciPlex 3',
    'sciplex4_training_results.csv': 'SciPlex 4',
}
PLOT_START_EPOCH = {
    'sciplex3_training_results.csv': 10,
}

EXPECTED = ['train_loss', 'train_r2', 'val_loss', 'val_r2']
COLORS = {
    'train_loss': '#E66101',
    'val_loss': '#5DA5DA',
    'train_r2': '#E66101',
    'val_r2': '#5DA5DA',
}

plt.style.use('default')


def style_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.1)
    ax.spines['bottom'].set_linewidth(1.1)
    ax.tick_params(axis='both', labelsize=10)


def prepare_plot_df(csv_path: Path):
    df = pd.read_csv(csv_path)
    missing = [c for c in EXPECTED if c not in df.columns]
    if missing:
        raise ValueError(f'{csv_path.name} missing columns: {missing}')

    df = df[EXPECTED].copy()
    start_epoch = PLOT_START_EPOCH.get(csv_path.name, 0)
    if start_epoch > 0:
        df = df.iloc[start_epoch:].reset_index(drop=True)
    df['Epoch'] = range(len(df))
    return df, start_epoch


def plot_single(csv_path: Path, title: str):
    df, start_epoch = prepare_plot_df(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.3), dpi=160)
    fig.suptitle(title, fontsize=16, y=0.98)

    ax = axes[0]
    ax.plot(df['Epoch'], df['train_loss'], color=COLORS['train_loss'], linewidth=2.6, label='train loss')
    ax.plot(df['Epoch'], df['val_loss'], color=COLORS['val_loss'], linewidth=2.6, label='val loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training & Validation Loss', fontsize=12)
    style_axis(ax)
    ax.legend(title='Type', frameon=False, fontsize=10, title_fontsize=10, loc='best')

    ax = axes[1]
    ax.plot(df['Epoch'], df['train_r2'], color=COLORS['train_r2'], linewidth=2.6, label='train_r2')
    ax.plot(df['Epoch'], df['val_r2'], color=COLORS['val_r2'], linewidth=2.6, label='val_r2')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training & Validation R$^2$', fontsize=12)
    style_axis(ax)
    ax.legend(title='Type', frameon=False, fontsize=10, title_fontsize=10, loc='best')

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    stem = csv_path.stem.replace('_training_results', '')
    png_path = OUT / f'{stem}_loss_r2.png'
    pdf_path = OUT / f'{stem}_loss_r2.pdf'
    fig.savefig(png_path, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    return png_path, pdf_path, len(df), start_epoch


def plot_combined(summary):
    fig, axes = plt.subplots(len(summary), 2, figsize=(9.2, 3.4 * len(summary)), dpi=160)
    if len(summary) == 1:
        axes = [axes]

    for idx, (csv_name, title) in enumerate(summary):
        df, _ = prepare_plot_df(RAW / csv_name)

        ax = axes[idx][0]
        ax.plot(df['Epoch'], df['train_loss'], color=COLORS['train_loss'], linewidth=2.1, label='train loss')
        ax.plot(df['Epoch'], df['val_loss'], color=COLORS['val_loss'], linewidth=2.1, label='val loss')
        ax.set_title(title, fontsize=13, pad=4)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        style_axis(ax)
        ax.legend(frameon=False, fontsize=8, loc='best')

        ax = axes[idx][1]
        ax.plot(df['Epoch'], df['train_r2'], color=COLORS['train_r2'], linewidth=2.1, label='train_r2')
        ax.plot(df['Epoch'], df['val_r2'], color=COLORS['val_r2'], linewidth=2.1, label='val_r2')
        ax.set_title(title, fontsize=13, pad=4)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('R$^2$', fontsize=10)
        style_axis(ax)
        ax.legend(frameon=False, fontsize=8, loc='best')

    fig.suptitle('Fig2A / FigS4 Training Curves Overview', fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(OUT / 'all_loss_r2_overview.png', bbox_inches='tight')
    fig.savefig(OUT / 'all_loss_r2_overview.pdf', bbox_inches='tight')
    plt.close(fig)


results = []
for csv_name, title in DATASETS.items():
    png_path, pdf_path, n_epoch, start_epoch = plot_single(RAW / csv_name, title)
    results.append((csv_name, title, png_path.name, pdf_path.name, n_epoch, start_epoch))

plot_combined([(k, v) for k, v in DATASETS.items()])

readme = OUT / 'README_loss_plots.txt'
with readme.open('w', encoding='utf-8') as f:
    f.write('Generated loss/R2 plots for Fig2A / FigS4-like training curves.\n')
    f.write('Each figure contains two panels: left = training/validation loss; right = training/validation R^2.\n\n')
    for csv_name, title, png_name, pdf_name, n_epoch, start_epoch in results:
        if start_epoch > 0:
            f.write(f'- {title}: source={csv_name}; plotted_epochs={n_epoch}; plot_start_epoch={start_epoch}; outputs={png_name}, {pdf_name}\n')
        else:
            f.write(f'- {title}: source={csv_name}; plotted_epochs={n_epoch}; outputs={png_name}, {pdf_name}\n')
    f.write('\nAdditional combined overview figure: all_loss_r2_overview.png / .pdf\n')

print('Generated files:')
for _, title, png_name, pdf_name, _, _ in results:
    print(f'  {title}: {png_name}, {pdf_name}')
print('  Combined: all_loss_r2_overview.png, all_loss_r2_overview.pdf')
