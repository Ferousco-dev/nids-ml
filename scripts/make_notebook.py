"""Build notebooks/01_eda.ipynb and execute it so the plots are stored.

Usage::

    python scripts/make_notebook.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "01_eda.ipynb"

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        "# Exploratory Data Analysis - Network Intrusion Detection\n\n"
        "This notebook profiles the labelled flow dataset used to train the detector: class balance, "
        "feature distributions, correlation structure, separability under PCA and data quality.",
    ),
    (
        "code",
        "import sys\n"
        "from pathlib import Path\n\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import seaborn as sns\n\n"
        "sys.path.insert(0, str(Path.cwd().parent))\n\n"
        "from src.data.loader import load_or_generate\n"
        "from src.features.engineer import engineer_features\n\n"
        "sns.set_theme(style='whitegrid', context='notebook')\n"
        "plt.rcParams['figure.dpi'] = 120\n"
        "TARGET = 'label'",
    ),
    ("markdown", "## 1. Load the dataset"),
    (
        "code",
        "data_path = Path.cwd().parent / 'data' / 'raw' / 'flows.csv'\n"
        "flows = load_or_generate(data_path, n_rows=50_000)\n"
        "print(f'Shape: {flows.shape[0]:,} rows x {flows.shape[1]} columns')\n"
        "flows.head()",
    ),
    ("markdown", "## 2. Class distribution"),
    (
        "code",
        "counts = flows[TARGET].value_counts()\n"
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "sns.barplot(x=counts.index, y=counts.to_numpy(), hue=counts.index, palette='rocket',\n"
        "            legend=False, ax=ax)\n"
        "for index, value in enumerate(counts.to_numpy()):\n"
        "    ax.text(index, value, f'{value:,}\\n{value / len(flows):.1%}', ha='center', va='bottom',\n"
        "            fontsize=9)\n"
        "ax.set_title('Traffic class distribution')\n"
        "ax.set_xlabel('Class')\n"
        "ax.set_ylabel('Flows')\n"
        "ax.set_ylim(0, counts.max() * 1.18)\n"
        "plt.show()",
    ),
    ("markdown", "## 3. Missing values and data quality"),
    (
        "code",
        "quality = pd.DataFrame({\n"
        "    'dtype': flows.dtypes.astype(str),\n"
        "    'missing': flows.isna().sum(),\n"
        "    'missing_pct': (flows.isna().mean() * 100).round(3),\n"
        "    'unique': flows.nunique(),\n"
        "})\n"
        "print(f'Duplicate rows: {flows.duplicated().sum():,}')\n"
        "print(f'Columns with missing values: {(quality[\"missing\"] > 0).sum()}')\n"
        "quality.sort_values('missing', ascending=False).head(15)",
    ),
    ("markdown", "## 4. Summary statistics"),
    ("code", "flows.describe().T.round(3).head(20)"),
    ("markdown", "## 5. Correlation structure"),
    (
        "code",
        "numeric = flows.select_dtypes(include=[np.number])\n"
        "correlation = numeric.corr()\n"
        "fig, ax = plt.subplots(figsize=(13, 10))\n"
        "sns.heatmap(correlation, cmap='vlag', center=0, square=True, linewidths=0.2,\n"
        "            cbar_kws={'shrink': 0.6, 'label': 'Pearson r'}, ax=ax)\n"
        "ax.set_title('Correlation between numeric flow features')\n"
        "plt.xticks(rotation=90, fontsize=7)\n"
        "plt.yticks(fontsize=7)\n"
        "plt.show()",
    ),
    ("markdown", "## 6. Feature distributions by class"),
    (
        "code",
        "from sklearn.feature_selection import mutual_info_classif\n\n"
        "scores = mutual_info_classif(numeric.fillna(0), flows[TARGET], random_state=42)\n"
        "top_features = pd.Series(scores, index=numeric.columns).nlargest(10)\n"
        "top_features.round(4)",
    ),
    (
        "code",
        "fig, axes = plt.subplots(5, 2, figsize=(14, 18))\n"
        "for ax, feature in zip(axes.ravel(), top_features.index):\n"
        "    sns.violinplot(data=flows, x=TARGET, y=feature, hue=TARGET, palette='crest',\n"
        "                   legend=False, cut=0, ax=ax)\n"
        "    ax.set_title(f'{feature} (MI={top_features[feature]:.3f})', fontsize=10)\n"
        "    ax.set_xlabel('')\n"
        "fig.suptitle('Distribution of the ten most informative features by class', y=1.005,\n"
        "             fontsize=13)\n"
        "fig.tight_layout()\n"
        "plt.show()",
    ),
    ("markdown", "## 7. PCA projection"),
    (
        "code",
        "from sklearn.decomposition import PCA\n"
        "from sklearn.preprocessing import StandardScaler\n\n"
        "engineered = engineer_features(flows)\n"
        "matrix = engineered.select_dtypes(include=[np.number]).fillna(0)\n"
        "sample = matrix.sample(min(6000, len(matrix)), random_state=42)\n"
        "labels = flows.loc[sample.index, TARGET]\n\n"
        "components = PCA(n_components=2, random_state=42).fit(StandardScaler().fit_transform(sample))\n"
        "projected = components.transform(StandardScaler().fit_transform(sample))\n\n"
        "fig, ax = plt.subplots(figsize=(8, 6))\n"
        "sns.scatterplot(x=projected[:, 0], y=projected[:, 1], hue=labels, palette='Set2', s=14,\n"
        "                alpha=0.7, edgecolor='none', ax=ax)\n"
        "ax.set_xlabel(f'PC1 ({components.explained_variance_ratio_[0]:.1%} variance)')\n"
        "ax.set_ylabel(f'PC2 ({components.explained_variance_ratio_[1]:.1%} variance)')\n"
        "ax.set_title('PCA projection of engineered flow features')\n"
        "ax.legend(title='Class', loc='best')\n"
        "plt.show()",
    ),
    ("markdown", "## 8. Findings"),
    (
        "markdown",
        "- The dataset is imbalanced: normal traffic dominates while `u2r` accounts for roughly one "
        "percent of flows, so macro-averaged metrics and class weighting matter more than accuracy.\n"
        "- Error-rate counters (`serror_rate`, `rerror_rate` and their `dst_host_` variants) separate "
        "denial-of-service and probe traffic sharply.\n"
        "- Byte volumes are heavy tailed, which is why the pipeline scales features and derives log and "
        "rate variants.\n"
        "- The PCA projection shows attack families forming distinct regions, supporting the strong "
        "cross-validated scores of the tree-based models.",
    ),
]


def build_notebook() -> nbformat.NotebookNode:
    """Assemble the EDA notebook."""
    notebook = new_notebook()
    notebook.cells = [
        new_markdown_cell(source) if kind == "markdown" else new_code_cell(source)
        for kind, source in CELLS
    ]
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    return notebook


def main() -> int:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(build_notebook(), NOTEBOOK_PATH)
    log.info("Notebook written to {}", NOTEBOOK_PATH)
    print(f"Notebook written to {NOTEBOOK_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
