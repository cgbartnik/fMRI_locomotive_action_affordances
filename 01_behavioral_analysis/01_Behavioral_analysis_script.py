"""
This script:
- loads behavioral data (actions, materials, categories, attributes, objects)
- computes RDMs with different distance metrics
- visualizes RDMs with clustermaps (full set + fMRI subset)
- computes correlations between different behavioral spaces
- computes an MDS embedding of the action RDM
- visualizes embeddings with both dots and image thumbnails
- runs 2D PCA scree plots and 3D PCA visualizations (plotly)
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import plotly.express as px
from PIL import Image
from rsatoolbox import vis

from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler as SS
from sklearn.decomposition import PCA
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from typing import Union, Optional, Dict, List, Tuple


# ---------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------

# Paths for data (adapt to your environment as needed)

DATA_BASE = "/home/cle/projects/PhD/fMRI_locomotive_action_affordances/fMRI_OSF_data/"
IMG_FULL_SET = os.path.join(
    DATA_BASE, "full_image_set"
)
IMG_FMRI_SET = os.path.join(
    DATA_BASE, "fmri_image_set"
)
BEHAVIOR_MEAN_DFS = os.path.join(
    DATA_BASE, "mean_behavior_dfs"
)

# Figures directory (relative to this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "Figures")

# Ordering lists (unchanged from your script)
fMRI_stim_ordering = [
    'indoor_0021', 'indoor_0025', 'indoor_0033', 'indoor_0055',
    'indoor_0058', 'indoor_0066', 'indoor_0080', 'indoor_0100',
    'indoor_0103', 'indoor_0130', 'indoor_0136', 'indoor_0145',
    'indoor_0146', 'indoor_0156', 'indoor_0163', 'indoor_0212',
    'indoor_0214', 'indoor_0215', 'indoor_0216', 'indoor_0221',
    'indoor_0235', 'indoor_0249', 'indoor_0266', 'indoor_0270',
    'indoor_0271', 'indoor_0272', 'indoor_0279', 'indoor_0281',
    'indoor_0282', 'indoor_0283', 'outdoor_manmade_0015',
    'outdoor_manmade_0030', 'outdoor_manmade_0032', 'outdoor_manmade_0040',
    'outdoor_manmade_0063', 'outdoor_manmade_0064', 'outdoor_manmade_0068',
    'outdoor_manmade_0089', 'outdoor_manmade_0110', 'outdoor_manmade_0117',
    'outdoor_manmade_0119', 'outdoor_manmade_0131', 'outdoor_manmade_0133',
    'outdoor_manmade_0147', 'outdoor_manmade_0148', 'outdoor_manmade_0149',
    'outdoor_manmade_0152', 'outdoor_manmade_0154', 'outdoor_manmade_0155',
    'outdoor_manmade_0157', 'outdoor_manmade_0161', 'outdoor_manmade_0165',
    'outdoor_manmade_0167', 'outdoor_manmade_0173', 'outdoor_manmade_0175',
    'outdoor_manmade_0220', 'outdoor_manmade_0256', 'outdoor_manmade_0257',
    'outdoor_manmade_0258', 'outdoor_manmade_0276', 'outdoor_natural_0004',
    'outdoor_natural_0008', 'outdoor_natural_0009', 'outdoor_natural_0010',
    'outdoor_natural_0011', 'outdoor_natural_0017', 'outdoor_natural_0023',
    'outdoor_natural_0034', 'outdoor_natural_0042', 'outdoor_natural_0049',
    'outdoor_natural_0050', 'outdoor_natural_0052', 'outdoor_natural_0053',
    'outdoor_natural_0062', 'outdoor_natural_0079', 'outdoor_natural_0091',
    'outdoor_natural_0097', 'outdoor_natural_0104', 'outdoor_natural_0128',
    'outdoor_natural_0132', 'outdoor_natural_0160', 'outdoor_natural_0198',
    'outdoor_natural_0200', 'outdoor_natural_0207', 'outdoor_natural_0246',
    'outdoor_natural_0250', 'outdoor_natural_0252', 'outdoor_natural_0255',
    'outdoor_natural_0261', 'outdoor_natural_0273'
]

VISACT_full_ordering = [
    'outdoor_natural_0001', 'outdoor_natural_0002', 'outdoor_natural_0003',
    'outdoor_natural_0004', 'outdoor_natural_0005', 'outdoor_natural_0007',
    'outdoor_natural_0008', 'outdoor_natural_0009', 'outdoor_natural_0010',
    'outdoor_natural_0011', 'outdoor_natural_0013', 'outdoor_manmade_0015',
    'outdoor_natural_0017', 'outdoor_manmade_0018', 'outdoor_manmade_0019',
    'indoor_0021', 'outdoor_natural_0023', 'outdoor_manmade_0024',
    'indoor_0025', 'outdoor_manmade_0026', 'outdoor_natural_0027',
    'outdoor_manmade_0028', 'outdoor_manmade_0030', 'outdoor_natural_0031',
    'outdoor_manmade_0032', 'indoor_0033', 'outdoor_natural_0034',
    'indoor_0035', 'indoor_0036', 'indoor_0038', 'outdoor_manmade_0040',
    'outdoor_natural_0042', 'outdoor_natural_0043', 'outdoor_manmade_0045',
    'outdoor_manmade_0046', 'outdoor_natural_0049', 'outdoor_natural_0050',
    'outdoor_natural_0052', 'outdoor_natural_0053', 'indoor_0054',
    'indoor_0055', 'indoor_0056', 'indoor_0057', 'indoor_0058',
    'outdoor_manmade_0059', 'outdoor_manmade_0060', 'outdoor_natural_0062',
    'outdoor_manmade_0063', 'outdoor_manmade_0064', 'outdoor_manmade_0065',
    'indoor_0066', 'outdoor_manmade_0067', 'outdoor_manmade_0068',
    'outdoor_manmade_0070', 'indoor_0072', 'outdoor_manmade_0074',
    'outdoor_natural_0075', 'outdoor_manmade_0076', 'outdoor_natural_0078',
    'outdoor_natural_0079', 'indoor_0080', 'outdoor_natural_0083',
    'outdoor_natural_0084', 'outdoor_natural_0085', 'outdoor_manmade_0087',
    'outdoor_natural_0088', 'outdoor_manmade_0089', 'outdoor_natural_0090',
    'outdoor_natural_0091', 'outdoor_manmade_0092', 'outdoor_manmade_0093',
    'outdoor_natural_0095', 'outdoor_natural_0096', 'outdoor_natural_0097',
    'outdoor_natural_0098', 'outdoor_natural_0099', 'indoor_0100',
    'outdoor_natural_0101', 'indoor_0102', 'indoor_0103',
    'outdoor_natural_0104', 'outdoor_natural_0105', 'outdoor_natural_0106',
    'outdoor_natural_0108', 'outdoor_manmade_0109', 'outdoor_manmade_0110',
    'outdoor_natural_0111', 'outdoor_natural_0113', 'outdoor_manmade_0115',
    'outdoor_manmade_0117', 'outdoor_manmade_0119', 'outdoor_natural_0120',
    'outdoor_natural_0121', 'outdoor_manmade_0123', 'outdoor_manmade_0124',
    'outdoor_manmade_0125', 'outdoor_natural_0126', 'outdoor_manmade_0127',
    'outdoor_natural_0128', 'outdoor_manmade_0129', 'indoor_0130',
    'outdoor_manmade_0131', 'outdoor_natural_0132', 'outdoor_manmade_0133',
    'indoor_0135', 'indoor_0136', 'indoor_0137', 'indoor_0139',
    'indoor_0140', 'outdoor_manmade_0142', 'indoor_0143',
    'outdoor_natural_0144', 'indoor_0145', 'indoor_0146',
    'outdoor_manmade_0147', 'outdoor_manmade_0148', 'outdoor_manmade_0149',
    'outdoor_manmade_0150', 'outdoor_manmade_0151', 'outdoor_manmade_0152',
    'outdoor_natural_0153', 'outdoor_manmade_0154', 'outdoor_manmade_0155',
    'indoor_0156', 'outdoor_manmade_0157', 'outdoor_natural_0158',
    'indoor_0159', 'outdoor_natural_0160', 'outdoor_manmade_0161',
    'outdoor_manmade_0162', 'indoor_0163', 'indoor_0164',
    'outdoor_manmade_0165', 'outdoor_natural_0166', 'outdoor_manmade_0167',
    'outdoor_manmade_0168', 'outdoor_manmade_0170', 'outdoor_manmade_0171',
    'outdoor_manmade_0172', 'outdoor_manmade_0173', 'outdoor_manmade_0175',
    'outdoor_manmade_0176', 'outdoor_manmade_0177', 'outdoor_natural_0178',
    'outdoor_manmade_0179', 'outdoor_natural_0181', 'outdoor_manmade_0182',
    'outdoor_manmade_0183', 'indoor_0184', 'indoor_0185',
    'outdoor_manmade_0186', 'outdoor_manmade_0188', 'outdoor_manmade_0189',
    'outdoor_manmade_0191', 'outdoor_manmade_0193', 'outdoor_manmade_0195',
    'outdoor_natural_0198', 'outdoor_natural_0199', 'outdoor_natural_0200',
    'outdoor_manmade_0202', 'outdoor_natural_0203', 'outdoor_manmade_0204',
    'outdoor_natural_0205', 'outdoor_natural_0206', 'outdoor_natural_0207',
    'indoor_0208', 'indoor_0209', 'outdoor_natural_0210', 'indoor_0211',
    'indoor_0212', 'indoor_0214', 'indoor_0215', 'indoor_0216',
    'outdoor_manmade_0217', 'outdoor_natural_0218', 'outdoor_manmade_0220',
    'indoor_0221', 'indoor_0224', 'indoor_0226', 'outdoor_natural_0227',
    'indoor_0229', 'outdoor_natural_0230', 'indoor_0231', 'indoor_0232',
    'indoor_0233', 'indoor_0234', 'indoor_0235', 'indoor_0236',
    'indoor_0237', 'indoor_0238', 'indoor_0239', 'outdoor_manmade_0240',
    'indoor_0241', 'indoor_0242', 'indoor_0243', 'indoor_0244',
    'indoor_0245', 'outdoor_natural_0246', 'outdoor_manmade_0247',
    'indoor_0249', 'outdoor_natural_0250', 'outdoor_natural_0251',
    'outdoor_natural_0252', 'outdoor_manmade_0253', 'outdoor_natural_0254',
    'outdoor_natural_0255', 'outdoor_manmade_0256', 'outdoor_manmade_0257',
    'outdoor_manmade_0258', 'indoor_0259', 'outdoor_natural_0261',
    'indoor_0262', 'indoor_0263', 'indoor_0264', 'indoor_0265',
    'indoor_0266', 'indoor_0268', 'indoor_0270', 'indoor_0271',
    'indoor_0272', 'outdoor_natural_0273', 'indoor_0274',
    'outdoor_natural_0275', 'outdoor_manmade_0276', 'indoor_0277',
    'indoor_0278', 'indoor_0279', 'indoor_0280', 'indoor_0281',
    'indoor_0282', 'indoor_0283'
]

# Long action-sorted list (unchanged)
action_sorted_list = [
    'outdoor_natural_0113', 'outdoor_manmade_0147', 'outdoor_manmade_0148',
    'outdoor_natural_0246', 'outdoor_natural_0062', 'outdoor_natural_0160',
    'outdoor_natural_0002', 'outdoor_natural_0255', 'outdoor_natural_0251',
    'outdoor_natural_0128', 'indoor_0156', 'outdoor_natural_0013',
    'outdoor_natural_0199', 'outdoor_natural_0105', 'outdoor_manmade_0173',
    'outdoor_manmade_0089', 'outdoor_natural_0104', 'outdoor_natural_0273',
    'outdoor_natural_0079', 'outdoor_manmade_0175', 'outdoor_natural_0078',
    'outdoor_natural_0218', 'outdoor_natural_0042', 'outdoor_natural_0198',
    'outdoor_natural_0153', 'outdoor_natural_0178', 'outdoor_natural_0090',
    'outdoor_manmade_0131', 'outdoor_natural_0091', 'outdoor_manmade_0152',
    'outdoor_natural_0200', 'outdoor_natural_0166', 'outdoor_manmade_0189',
    'outdoor_manmade_0253', 'outdoor_natural_0275', 'outdoor_natural_0205',
    'outdoor_natural_0031', 'outdoor_natural_0254', 'outdoor_manmade_0157',
    'indoor_0268', 'outdoor_natural_0181', 'outdoor_manmade_0155',
    'indoor_0282', 'outdoor_manmade_0256', 'outdoor_manmade_0257',
    'outdoor_natural_0011', 'indoor_0066', 'outdoor_manmade_0119',
    'outdoor_manmade_0220', 'outdoor_manmade_0068', 'outdoor_manmade_0133',
    'outdoor_manmade_0258', 'outdoor_manmade_0040', 'outdoor_natural_0132',
    'outdoor_manmade_0064', 'outdoor_manmade_0032', 'outdoor_manmade_0063',
    'outdoor_manmade_0204', 'outdoor_manmade_0015', 'outdoor_manmade_0129',
    'outdoor_manmade_0110', 'outdoor_manmade_0167', 'outdoor_manmade_0059',
    'outdoor_manmade_0124', 'outdoor_manmade_0123', 'outdoor_manmade_0117',
    'outdoor_manmade_0162', 'outdoor_manmade_0150', 'outdoor_manmade_0046',
    'outdoor_manmade_0087', 'outdoor_manmade_0109', 'outdoor_manmade_0030',
    'outdoor_natural_0111', 'outdoor_natural_0207', 'outdoor_natural_0027',
    'outdoor_natural_0053', 'outdoor_manmade_0217', 'outdoor_natural_0227',
    'indoor_0277', 'outdoor_manmade_0193', 'outdoor_natural_0083',
    'outdoor_manmade_0070', 'outdoor_manmade_0168', 'outdoor_natural_0206',
    'outdoor_natural_0261', 'outdoor_natural_0003', 'outdoor_natural_0097',
    'outdoor_manmade_0179', 'outdoor_manmade_0202', 'outdoor_manmade_0019',
    'outdoor_manmade_0024', 'outdoor_manmade_0176', 'outdoor_manmade_0186',
    'outdoor_manmade_0240', 'outdoor_natural_0004', 'outdoor_natural_0144',
    'outdoor_natural_0096', 'outdoor_manmade_0125', 'outdoor_manmade_0247',
    'outdoor_manmade_0045', 'outdoor_natural_0108', 'outdoor_natural_0007',
    'outdoor_manmade_0065', 'outdoor_natural_0001', 'outdoor_natural_0120',
    'outdoor_manmade_0182', 'outdoor_manmade_0028', 'outdoor_natural_0106',
    'outdoor_natural_0121', 'outdoor_manmade_0149', 'outdoor_natural_0005',
    'outdoor_manmade_0177', 'outdoor_natural_0101', 'outdoor_natural_0126',
    'outdoor_natural_0034', 'outdoor_natural_0098', 'outdoor_manmade_0195',
    'outdoor_manmade_0161', 'outdoor_natural_0203', 'outdoor_manmade_0115',
    'indoor_0033', 'indoor_0163', 'indoor_0235', 'indoor_0100', 'indoor_0058',
    'indoor_0145', 'indoor_0271', 'indoor_0266', 'outdoor_manmade_0127',
    'indoor_0130', 'outdoor_manmade_0151', 'outdoor_natural_0230',
    'outdoor_manmade_0067', 'outdoor_manmade_0276', 'outdoor_manmade_0188',
    'outdoor_manmade_0191', 'outdoor_manmade_0018', 'indoor_0025',
    'outdoor_manmade_0142', 'outdoor_manmade_0172', 'indoor_0021',
    'outdoor_manmade_0026', 'outdoor_manmade_0092', 'outdoor_manmade_0171',
    'outdoor_manmade_0165', 'indoor_0283', 'outdoor_manmade_0183',
    'indoor_0136', 'indoor_0249', 'indoor_0279', 'indoor_0215',
    'indoor_0280', 'indoor_0278', 'indoor_0274', 'indoor_0265',
    'indoor_0263', 'indoor_0262', 'indoor_0245', 'indoor_0244',
    'indoor_0243', 'indoor_0242', 'indoor_0238', 'indoor_0237',
    'indoor_0236', 'indoor_0234', 'indoor_0233', 'indoor_0232',
    'indoor_0231', 'indoor_0229', 'indoor_0226', 'indoor_0224',
    'indoor_0221', 'indoor_0216', 'indoor_0214', 'indoor_0211',
    'indoor_0209', 'indoor_0208', 'indoor_0185', 'indoor_0184',
    'indoor_0164', 'indoor_0159', 'indoor_0143', 'indoor_0140',
    'indoor_0139', 'indoor_0137', 'indoor_0135', 'indoor_0102',
    'indoor_0080', 'indoor_0072', 'indoor_0057', 'indoor_0056',
    'indoor_0035', 'indoor_0038', 'indoor_0103', 'indoor_0146',
    'indoor_0259', 'indoor_0055', 'indoor_0241', 'outdoor_manmade_0074',
    'indoor_0212', 'indoor_0054', 'outdoor_manmade_0076', 'indoor_0281',
    'indoor_0264', 'indoor_0036', 'indoor_0239', 'outdoor_manmade_0154',
    'outdoor_natural_0158', 'indoor_0270', 'outdoor_natural_0049',
    'outdoor_natural_0009', 'outdoor_natural_0010', 'indoor_0272',
    'outdoor_natural_0008', 'outdoor_natural_0052', 'outdoor_natural_0043',
    'outdoor_natural_0084', 'outdoor_natural_0210', 'outdoor_natural_0023',
    'outdoor_natural_0095', 'outdoor_natural_0099', 'outdoor_manmade_0060',
    'outdoor_natural_0088', 'outdoor_manmade_0170', 'outdoor_natural_0250',
    'outdoor_natural_0050', 'outdoor_natural_0075', 'outdoor_natural_0017',
    'outdoor_natural_0252', 'outdoor_natural_0085', 'outdoor_manmade_0093'
]

# Derived list: fMRI subset
action_sorted_list_fMRI = [x for x in action_sorted_list if x in fMRI_stim_ordering]

# Colors for actions / environments
ACTION_COLORS = {
    "walking": "#BBBBBB",
    "biking": "#BBCC33",
    "driving": "#EEDD88",
    "climbing": "#EE8866",
    "swimming": "#77AADD",
    "boating": "#99DDFF",
}

ENV_COLORS = {
    "indoor": "cornflowerblue",
    "natural": "seagreen",
    "manmade": "black",
}


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------

def compute_rdm_corr(features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """
    Compute correlation distance RDM using the Net2Brain-like procedure:
    1. z-score features across images
    2. Pearson correlation
    3. convert to distance: 1 - r
    """
    r_scaled = SS().fit_transform(np.asarray(features))
    rdm = 1 - np.corrcoef(r_scaled)
    return np.asarray(rdm)


def load_behavior_dfs(mean_df_path: str) -> Dict[str, pd.DataFrame]:
    """Load all behavioral mean dataframes from a given directory."""
    paths = {
        "action": "mean_action_df.csv",
        "material": "mean_material_df.csv",
        "categories": "mean_categories_df.csv",
        "attributes": "mean_attributes_df.csv",
        "objects": "mean_objects_df.csv",
    }
    dfs: Dict[str, pd.DataFrame] = {}
    for key, fname in paths.items():
        dfs[key] = pd.read_csv(os.path.join(mean_df_path, fname), index_col=0)
    return dfs


def normalize_image_index(df: Union[pd.DataFrame, pd.Series]) -> pd.Index:
    """
    Convert original image ids to 'indoor_XXXX', 'outdoor_manmade_XXXX', etc.
    Works for both DataFrames and Series (uses its index).
    """
    new_index: List[str] = []
    for idx in df.index:
        if "indoor" in idx:
            num = idx.split("_")[0]
            new_index.append(f"indoor_{num}")
        elif "outdoor_manmade" in idx:
            num = idx.split("_")[0]
            new_index.append(f"outdoor_manmade_{num}")
        elif "outdoor_natural" in idx:
            num = idx.split("_")[0]
            new_index.append(f"outdoor_natural_{num}")
        else:
            new_index.append(idx)
    return pd.Index(new_index)


def apply_normalized_index(dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Apply normalized index (indoor_XXXX / outdoor_...) consistently to all dfs.
    Returns a new dict of DataFrames.
    """
    ref_df = dfs["action"]
    new_index = normalize_image_index(ref_df)

    out: Dict[str, pd.DataFrame] = {}
    for name, df in dfs.items():
        df = df.copy()
        df.index = new_index
        out[name] = df
    return out


def balance_action_labels(action_df: pd.DataFrame) -> pd.Series:
    """
    Given mean action probabilities per image, extract max label per image
    and rebalance: if 'walking' is max but 'biking' or 'driving' > 0.5
    and is the 2nd largest, relabel accordingly.
    """
    labels = action_df.idxmax(axis=1).copy()

    for idx, row in action_df.iterrows():
        if (
            row.get("walking", 0) > row.get("biking", 0)
            and row.get("biking", 0) > row.get("driving", 0)
            and row.get("biking", 0) > 0.5
        ):
            labels[idx] = "biking"
        if (
            row.get("walking", 0) > row.get("driving", 0)
            and row.get("driving", 0) > row.get("biking", 0)
            and row.get("driving", 0) > 0.5
        ):
            labels[idx] = "driving"

    return labels


def get_env_labels(ordering: List[str]) -> Tuple[List[str], List[str]]:
    """
    For a list of image IDs, return:
    - a list of environment type (indoor / natural / manmade)
    - a list of corresponding colors
    """
    env_types: List[str] = []
    env_colors: List[str] = []

    for stim in ordering:
        if "indoor" in stim:
            env_types.append("indoor")
            env_colors.append(ENV_COLORS["indoor"])
        elif "outdoor_natural" in stim:
            env_types.append("natural")
            env_colors.append(ENV_COLORS["natural"])
        elif "outdoor_manmade" in stim:
            env_types.append("manmade")
            env_colors.append(ENV_COLORS["manmade"])
        else:
            env_types.append("unknown")
            env_colors.append("grey")

    return env_types, env_colors


def plot_action_rdm_clustermap(
    rdm: np.ndarray,
    ordering: List[str],
    action_labels: pd.Series,
    env_colors: List[str],
    save_path: Optional[str] = None,
    row_cluster: bool = True,
    col_cluster: bool = True,
    title: Optional[str] = None,
) -> None:
    """
    Plot a clustermap of the action RDM with row color by action
    and column color by environment.
    """
    row_colors = list(action_labels.loc[ordering].map(ACTION_COLORS))

    cg = sns.clustermap(
        data=rdm,
        row_colors=row_colors,
        col_colors=env_colors,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        cmap="mako",
        cbar_pos=(1, .2, .02, .4),
    )

    cg.ax_col_dendrogram.set_visible(False)
    cg.ax_row_dendrogram.set_visible(row_cluster)

    cg.ax_heatmap.set_xticklabels([])
    cg.ax_heatmap.set_yticklabels([])
    cg.ax_heatmap.tick_params(right=False, bottom=False)

    if title is not None:
        cg.fig.suptitle(title)

    if save_path is not None:
        cg.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)

    plt.show(block=False)


def compute_correlation_matrix(RDM_list: List[np.ndarray]) -> np.ndarray:
    """
    Given a list of square RDMs, compute the Spearman correlation between
    their vectorized (upper-triangle) forms.
    """
    n = len(RDM_list)
    corrs = np.zeros((n, n))

    for i, RDM1 in enumerate(RDM_list):
        rdv1 = squareform(RDM1.round(5))
        for j, RDM2 in enumerate(RDM_list):
            rdv2 = squareform(RDM2.round(5))
            corrs[i, j] = spearmanr(rdv1, rdv2)[0]

    return corrs


def plot_correlation_matrix(
    corrs: np.ndarray,
    ax: plt.Axes,
    title: str,
    names: bool,
    rdm_names: List[str],
) -> None:
    """Helper to plot the correlation matrix between RDMs."""
    ax.imshow(corrs, cmap="coolwarm", vmin=corrs.min(), vmax=corrs.max())
    if names:
        ax.set_xticklabels([])
        ax.set_yticks(np.arange(len(rdm_names)))
        ax.set_yticklabels(rdm_names, fontsize=15)
        ax.tick_params(bottom=False)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(right=False, bottom=False, left=False, top=False)

    # grid lines
    for pos in np.arange(0.5, len(rdm_names)):
        ax.hlines(pos, *ax.get_xlim(), color="w", linewidth=1.5)
        ax.vlines(pos, *ax.get_ylim(), color="w", linewidth=1.5)

    ax.set_title(title, fontsize=20)

    # off-diagonal values
    for i in range(corrs.shape[0]):
        for j in range(corrs.shape[1]):
            if i != j:
                ax.text(
                    j,
                    i,
                    f"{corrs[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10,
                )


def get_image_for_annotation(path: str) -> OffsetImage:
    image = Image.open(path)
    image = image.resize((50, 50))
    return OffsetImage(image)


def run_pca_with_scree(
    n_components: int,
    mean_df: pd.DataFrame,
    labels: pd.Series,
    label_column_name: str = "Cluster",
    save_path: Optional[str] = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Run PCA, return components and a dataframe; also plot scree and
    cumulative explained variance. Optionally save figure.
    """
    pca = PCA(n_components=n_components)
    pc = pca.fit_transform(mean_df)
    pc_labels = [f"PC{i+1}" for i in range(n_components)]

    pc_df = pd.DataFrame(pc, columns=pc_labels, index=mean_df.index)
    pc_df[label_column_name] = labels

    ev_df = pd.DataFrame({"var": pca.explained_variance_ratio_, "PC": pc_labels})
    ev_df["cumulative_var"] = ev_df["var"].cumsum()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scree plot
    ax1 = axes[0]
    ax1.plot(ev_df["PC"], ev_df["var"], marker="o", linestyle="-", color="black")
    ax1.set_ylim(0, 1)
    ax1.set_title("Scree Plot")
    ax1.set_xlabel("Principal Components")
    ax1.set_ylabel("Explained Variance")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Cumulative scree
    ax2 = axes[1]
    ax2.plot(ev_df["PC"], ev_df["cumulative_var"], marker="o", linestyle="-", color="black")
    ax2.set_ylim(0, 1)
    ax2.set_title("Cumulative Scree Plot")
    ax2.set_xlabel("Principal Components")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)

    plt.show()

    return pc, pc_df


def pca_only_3d(
    n_components: int,
    mean_df: pd.DataFrame,
    labels: pd.Series,
    df_name: str,
    label_name: str,
    stim_list: List[str],
    kind_plot: str = "not_fmri",
) -> pd.DataFrame:
    """
    Run PCA and visualize 3D embedding with plotly.

    kind_plot:
        - "not_fmri": label images not in stim_list as 'excluded'
        - "fmri":     mark images in stim_list as 'fmri set'
        - "fmri_only": keep only images in stim_list
        - anything else: no special treatment
    """
    mean_df = mean_df.copy()
    labels = labels.copy()
    mean_df.index = normalize_image_index(mean_df)
    labels.index = normalize_image_index(labels)

    stim_set = set(stim_list)

    if kind_plot == "not_fmri":
        for idx in labels.index:
            if idx not in stim_set:
                labels[idx] = "excluded"
    elif kind_plot == "fmri":
        for idx in labels.index:
            if idx in stim_set:
                labels[idx] = "fmri set"
    elif kind_plot == "fmri_only":
        keep_idx = [idx for idx in labels.index if idx in stim_set]
        labels = labels.loc[keep_idx]
        mean_df = mean_df.loc[keep_idx]

    # PCA
    pca = PCA(n_components=n_components)
    pc = pca.fit_transform(mean_df)
    pc_labels = [f"PC{i+1}" for i in range(n_components)]

    pc_df = pd.DataFrame(pc, columns=pc_labels, index=mean_df.index)
    pc_df["Action"] = labels

    order_map = {
        "walking": 1,
        "biking": 2,
        "driving": 3,
        "swimming": 4,
        "boating": 5,
        "climbing": 6,
        "excluded": 7,
    }
    pc_df["Order"] = pc_df["Action"].map(order_map)
    pc_df.sort_values(by="Order", inplace=True, ignore_index=True)

    if kind_plot == "fmri":
        color_seq = [
            "#BBBBBB",
            "#BBCC33",
            "#EEDD88",
            "#77AADD",
            "#99DDFF",
            "rgba(0,0,0,0)",
        ]
    else:
        color_seq = [
            "#BBBBBB",
            "#BBCC33",
            "#EEDD88",
            "#77AADD",
            "#99DDFF",
            "#EE8866",
            "rgba(0,0,0,0)",
        ]

    fig = px.scatter_3d(
        pc_df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Action",
        color_discrete_sequence=color_seq,
        custom_data=[pc_df.index],
        opacity=0.7,
    )

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.5, y=1.5, z=1),
    )

    fig.update_layout(
        scene_camera=camera,
        title_y=0.85,
        title_x=0.5,
        legend_x=0.8,
        legend_y=0.6,
        width=700,
        paper_bgcolor="white",
        scene=dict(
            xaxis=dict(
                backgroundcolor="white",
                gridcolor="lightgray",
                tick0=-0.5,
                dtick=0.5,
            ),
            yaxis=dict(
                backgroundcolor="white",
                gridcolor="lightgray",
                tick0=-0.5,
                dtick=0.5,
            ),
            zaxis=dict(
                backgroundcolor="white",
                gridcolor="lightgray",
                tick0=-0.5,
                dtick=0.5,
            ),
        ),
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(font=dict(size=15), itemsizing="constant"),
    )

    fig.update_traces(
        marker_size=5,
        hovertemplate="<br>".join(
            [
                "Image: %{customdata[0]}",
            ]
        ),
    )

    fig.show()
    return pc_df


def visact_icons(**kwargs) -> Dict[str, List[vis.Icon]]:
    """
    Create Icon objects (rsatoolbox.vis) for each stimulus in the fMRI image set.

    Returns a dict with keys:
        - 'image': image icons
        - 'string': label-based icons
        - 'marker': marker-style icons
    """
    stim_files = sorted(os.listdir(IMG_FMRI_SET))

    img_arrays = []
    for img_name in stim_files:
        img_path = os.path.join(IMG_FMRI_SET, img_name)
        image = Image.open(img_path)
        image = image.resize((150, 150))
        img_arrays.append(np.asarray(image))
    img_arrays = np.array(img_arrays)

    stim_list = [stim[:-4] for stim in stim_files]

    markers: List[str] = []
    colors: List[str] = []
    env_type: List[str] = []

    for stim in stim_list:
        parts = stim.split("_")
        if len(parts) == 2:
            env_type.append("indoor")
            colors.append("#D72C2C")
            markers.append("o")
        elif len(parts) == 3:
            env_label = f"outdoor_{parts[1]}"
            env_type.append(env_label)
            if parts[1] == "manmade":
                colors.append("#2C7CD7")
                markers.append("s")
            else:
                colors.append("#2CD7B1")
                markers.append("^")
        else:
            env_type.append("unknown")
            colors.append("grey")
            markers.append("o")

    icons: Dict[str, List[vis.Icon]] = defaultdict(list)
    for i, stim_name in enumerate(stim_list):
        icons["image"].append(
            vis.Icon(
                image=img_arrays[i],
                color=colors[i],
                circ_cut=None,
                border_type=None,
                border_width=5,
                **kwargs,
            )
        )
        icons["string"].append(
            vis.Icon(
                string=env_type[i],
                color=colors[i],
                font_color=colors[i],
                **kwargs,
            )
        )
        icons["marker"].append(
            vis.Icon(
                marker=markers[i],
                color=colors[i],
                **kwargs,
            )
        )
    return icons


# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------

def main() -> None:
    # Ensure Figures directory exists
    os.makedirs(FIG_DIR, exist_ok=True)

    # -----------------------------------------------------------------
    # 1. Load behavior data (from fs4 models/behavior)
    # -----------------------------------------------------------------
    dfs = load_behavior_dfs(BEHAVIOR_MEAN_DFS)
    dfs = apply_normalized_index(dfs)  # consistent indexing

    action_df = dfs["action"]
    material_df = dfs["material"]
    categories_df = dfs["categories"]
    attributes_df = dfs["attributes"]
    objects_df = dfs["objects"]

    # -----------------------------------------------------------------
    # 2. Reorder and compute RDMs
    # -----------------------------------------------------------------
    action_subset = action_df.loc[VISACT_full_ordering]
    material_subset = material_df.loc[VISACT_full_ordering]
    categories_subset = categories_df.loc[VISACT_full_ordering]
    attributes_subset = attributes_df.loc[VISACT_full_ordering]
    objects_subset = objects_df.loc[VISACT_full_ordering]

    action_rdm_euclidean = pairwise_distances(action_subset, metric="euclidean")
    action_rdm_cosine = pairwise_distances(action_subset, metric="cosine")
    action_rdm_corr = compute_rdm_corr(action_subset)

    material_rdm_euclidean = pairwise_distances(material_subset, metric="euclidean")
    material_rdm_cosine = pairwise_distances(material_subset, metric="cosine")
    material_rdm_corr = compute_rdm_corr(material_subset)

    categories_rdm_euclidean = pairwise_distances(categories_subset, metric="euclidean")
    categories_rdm_cosine = pairwise_distances(categories_subset, metric="cosine")
    categories_rdm_corr = compute_rdm_corr(categories_subset)

    attributes_rdm_euclidean = pairwise_distances(attributes_subset, metric="euclidean")
    attributes_rdm_cosine = pairwise_distances(attributes_subset, metric="cosine")
    attributes_rdm_corr = compute_rdm_corr(attributes_subset)

    objects_rdm_euclidean = pairwise_distances(objects_subset, metric="euclidean")
    objects_rdm_cosine = pairwise_distances(objects_subset, metric="cosine")
    objects_rdm_corr = compute_rdm_corr(objects_subset)

    # -----------------------------------------------------------------
    # 3. Action labels (balanced) & environment labels
    # -----------------------------------------------------------------
    mean_action_labels_balanced = balance_action_labels(action_df)
    full_labels = mean_action_labels_balanced.loc[VISACT_full_ordering]
    env_types_full, env_colors_full = get_env_labels(VISACT_full_ordering)

    # -----------------------------------------------------------------
    # 4. Clustermap: full RDM with clustering  (Figure 1)
    # -----------------------------------------------------------------
    fig_path_full_rdm = os.path.join(FIG_DIR, "Fig1_RDM_full_clustered.png")
    plot_action_rdm_clustermap(
        rdm=action_rdm_corr,
        ordering=VISACT_full_ordering,
        action_labels=mean_action_labels_balanced,
        env_colors=env_colors_full,
        save_path=fig_path_full_rdm,
        row_cluster=True,
        col_cluster=True,
        title="Action RDM (full, clustered)",
    )

    # -----------------------------------------------------------------
    # 5. Environment-sorted clustermap (no clustering)  (Figure 2)
    # -----------------------------------------------------------------
    action_rdm_df = pd.DataFrame(
        action_rdm_corr,
        index=VISACT_full_ordering,
        columns=VISACT_full_ordering,
    )
    action_rdm_df["env"] = env_types_full
    action_rdm_df["action"] = full_labels

    sorted_df = action_rdm_df.sort_values(by="env")
    visualization_df = sorted_df.iloc[:, :-2]
    visualization_df = visualization_df[visualization_df.index]
    _, env_colors_resorted = get_env_labels(list(sorted_df.index))

    plot_action_rdm_clustermap(
        rdm=visualization_df.to_numpy(),
        ordering=list(sorted_df.index),
        action_labels=sorted_df["action"],
        env_colors=env_colors_resorted,
        save_path=os.path.join(FIG_DIR, "Fig2_RDM_full_envsorted.png"),
        row_cluster=False,
        col_cluster=False,
        title="Action RDM (env-sorted)",
    )

    # -----------------------------------------------------------------
    # 6. fMRI subset (90 images) RDMs & clustermaps (Figures 3 & 4)
    # -----------------------------------------------------------------
    action_90_subset = action_df.loc[action_sorted_list_fMRI]
    action_90_rdm_euclidean = pairwise_distances(action_90_subset, metric="euclidean")
    action_90_rdm_corr = compute_rdm_corr(action_90_subset)

    labels_90 = balance_action_labels(action_df).loc[action_sorted_list_fMRI]
    _, env_colors_90 = get_env_labels(action_sorted_list_fMRI)

    # 90-image clustermap (Figure 3)
    plot_action_rdm_clustermap(
        rdm=action_90_rdm_corr,
        ordering=action_sorted_list_fMRI,
        action_labels=balance_action_labels(action_df),
        env_colors=env_colors_90,
        save_path=os.path.join(FIG_DIR, "Fig3_RDM_fmri_subset_clustered.png"),
        row_cluster=True,
        col_cluster=True,
        title="Action RDM (fMRI subset, clustered)",
    )

    # Environment-sorted version for 90 subset (Figure 4)
    action_rdm_90_df = pd.DataFrame(
        action_90_rdm_corr,
        index=action_sorted_list_fMRI,
        columns=action_sorted_list_fMRI,
    )
    env_types_90, _ = get_env_labels(action_sorted_list_fMRI)
    action_rdm_90_df["env"] = env_types_90
    action_rdm_90_df["action"] = labels_90

    sorted_90_df = action_rdm_90_df.sort_values(by="env")
    visualization_90_df = sorted_90_df.iloc[:, :-2]
    visualization_90_df = visualization_90_df[visualization_90_df.index]
    _, env_colors_90_sorted = get_env_labels(list(sorted_90_df.index))

    plot_action_rdm_clustermap(
        rdm=visualization_90_df.to_numpy(),
        ordering=list(sorted_90_df.index),
        action_labels=sorted_90_df["action"],
        env_colors=env_colors_90_sorted,
        save_path=os.path.join(FIG_DIR, "Fig4_RDM_fmri_subset_envsorted.png"),
        row_cluster=False,
        col_cluster=False,
        title="Action RDM (fMRI subset, env-sorted)",
    )

    # -----------------------------------------------------------------
    # 7. Correlation between distance spaces (Figure 5)
    # -----------------------------------------------------------------
    rdm_list_corr = [
        action_rdm_corr,
        objects_rdm_corr,
        material_rdm_corr,
        attributes_rdm_corr,
        categories_rdm_corr,
    ]
    rdm_list_euclidean = [
        action_rdm_euclidean,
        objects_rdm_euclidean,
        material_rdm_euclidean,
        attributes_rdm_euclidean,
        categories_rdm_euclidean,
    ]
    rdm_list_cosine = [
        action_rdm_cosine,
        objects_rdm_cosine,
        material_rdm_cosine,
        attributes_rdm_cosine,
        categories_rdm_cosine,
    ]

    rdm_names = [
        "Navigational\naffordances",
        "Objects",
        "Materials",
        "Global\nproperties",
        "Categories",
    ]

    corrs_corr = compute_correlation_matrix(rdm_list_corr)
    corrs_euclidean = compute_correlation_matrix(rdm_list_euclidean)
    corrs_cosine = compute_correlation_matrix(rdm_list_cosine)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plot_correlation_matrix(
        corrs_corr, axes[0], "Correlation Distance", names=True, rdm_names=rdm_names
    )
    plot_correlation_matrix(
        corrs_euclidean, axes[1], "Euclidean Distance", names=False, rdm_names=rdm_names
    )
    plot_correlation_matrix(
        corrs_cosine, axes[2], "Cosine Distance", names=False, rdm_names=rdm_names
    )
    plt.tight_layout()
    fig.savefig(
        os.path.join(FIG_DIR, "Fig5_RDMspace_correlations.png"),
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    plt.show(block=False)

    # -----------------------------------------------------------------
    # 8. MDS on action RDM (Euclidean) – dots only (Figure 6)
    # -----------------------------------------------------------------
    embedding = MDS(
        n_components=2,
        random_state=123,
        dissimilarity="precomputed",
    )
    embedding.fit_transform(action_rdm_euclidean)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(
        embedding.embedding_[:, 0],
        embedding.embedding_[:, 1],
        c=full_labels.map(ACTION_COLORS),
        edgecolor="black",
    )
    ax.set_xticks([])
    ax.set_yticks([])

    ax.legend(
        handles=[
            Patch(facecolor="#BBBBBB", label="walking"),
        Patch(facecolor="#EE8866", label="climbing"),
        Patch(facecolor="#BBCC33", label="biking"),
        Patch(facecolor="#77AADD", label="swimming"),
        Patch(facecolor="#EEDD88", label="driving"),
        Patch(facecolor="#99DDFF", label="boating"),
        ],
        fontsize=12,
        bbox_to_anchor=(0.5, -0.05),
        loc="upper center",
        ncol=3,
    )

    fig.tight_layout()
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.patch.set_facecolor("none")

    fig.savefig(
        os.path.join(FIG_DIR, "Fig6_MDS_actions.png"),
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    plt.show(block=False)

    # -----------------------------------------------------------------
    # 9. MDS scatter with images (Figure 7)
    # -----------------------------------------------------------------
    x = embedding.embedding_[:, 0]
    y = embedding.embedding_[:, 1]

    img_paths = [
        os.path.join(IMG_FULL_SET, f"{img_id}.jpg") for img_id in VISACT_full_ordering
    ]

    fig, ax = plt.subplots(figsize=(15, 20))
    ax.scatter(x, y)

    for x0, y0, path, color in zip(
        x, y, img_paths, full_labels.map(ACTION_COLORS)
    ):
        ab = AnnotationBbox(
            get_image_for_annotation(path),
            (x0, y0),
            bboxprops=dict(
                edgecolor=color,
                linewidth=5,
                facecolor="none",
                boxstyle="square,pad=0",
            ),
        )
        ax.add_artist(ab)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.patch.set_facecolor("none")
    plt.tight_layout()

    fig_path_tsne = os.path.join(FIG_DIR, "Fig7_MDS_actions_images.svg")
    plt.savefig(fig_path_tsne, dpi=300, bbox_inches="tight", transparent=True)
    plt.show(block=False)

    # -----------------------------------------------------------------
    # 10. PCA on action_df (scree + cumulative)  (Figure 8)
    # -----------------------------------------------------------------
    _pc, _pc_df = run_pca_with_scree(
        n_components=6,
        mean_df=action_df,
        labels=mean_action_labels_balanced,
        label_column_name="Action",
        save_path=os.path.join(FIG_DIR, "Fig8_PCA_scree.png"),
    )

    # -----------------------------------------------------------------
    # 11. 3D PCA plots (plotly) on BEHAVIOR mean_action_df
    #     (not saved as static figures here – still interactive)
    # -----------------------------------------------------------------
    mean_action_df_behavior = pd.read_csv(
        os.path.join(BEHAVIOR_MEAN_DFS, "mean_action_df.csv"),
        index_col=0,
    )
    mean_action_df_behavior.index = normalize_image_index(mean_action_df_behavior)
    behavior_labels = balance_action_labels(mean_action_df_behavior)

    # fMRI highlighting
    _ = pca_only_3d(
        n_components=3,
        mean_df=mean_action_df_behavior,
        labels=behavior_labels,
        df_name="action",
        label_name="balanced",
        stim_list=fMRI_stim_ordering,
        kind_plot="fmri",
    )

    # no_change
    _ = pca_only_3d(
        n_components=3,
        mean_df=mean_action_df_behavior,
        labels=behavior_labels,
        df_name="action",
        label_name="balanced",
        stim_list=fMRI_stim_ordering,
        kind_plot="no_change",
    )


if __name__ == "__main__":
    main()
