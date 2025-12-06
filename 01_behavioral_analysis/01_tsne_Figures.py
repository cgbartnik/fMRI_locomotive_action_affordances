import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_paths import get_data_root  

# ===========================
# Paths & basic config
# ===========================
DATA_ROOT = get_data_root()
STIMULUS_DIR = os.path.join(DATA_ROOT, "Stimulus_sets")
MEAN_BEHAVIOR_DIR = os.path.join(DATA_ROOT, "mean_behavior_dfs")

TSNE_DF_OSF = os.path.join(STIMULUS_DIR, "tsne_complete_df.csv")
TSNE_FEATURES_OSF = os.path.join(STIMULUS_DIR, "dataset_viz_tsne_features.npy")

# Where to save figures (next to this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "Figures")
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_palette("bright")


# ===========================
# Helpers
# ===========================

def load_osf_df_and_tsne():
    """Load the combined tsne dataframe and its TSNE features from OSF paths."""
    df = pd.read_csv(TSNE_DF_OSF, index_col=0).reset_index()
    tsne_features = np.load(TSNE_FEATURES_OSF)
    return df, tsne_features


def get_indices_by_dataset(df, dataset_name):
    """Return index positions for rows matching a given dataset name."""
    return df.index[df["dataset"] == dataset_name].tolist()


def scatter_group(ax, x, y, color, label, size=100, alpha=1.0, edgecolor="black"):
    """Convenience wrapper for a simple scatter group."""
    return sns.scatterplot(
        x=x,
        y=y,
        color=color,
        edgecolor=edgecolor,
        s=size,
        alpha=alpha,
        label=label,
        ax=ax,
    )


def label_highest_action_row(row, action_cols):
    """
    Return the name of the action column with the highest value > 0.
    If none are > 0, return 'no action'.
    """
    valid = row[action_cols][row[action_cols] > 0]
    return valid.idxmax() if not valid.empty else "no action"


def style_axes(ax):
    """Hide axes and spines for a clean t-SNE look."""
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(False)


# ===========================
# Plot 1: All datasets in t-SNE
# ===========================

def plot_all_datasets_tsne():
    complete_df, tsne_features = load_osf_df_and_tsne()

    # Index sets
    sun_idx    = get_indices_by_dataset(complete_df, "Patterson et al., (2012)")
    fmri_idx   = get_indices_by_dataset(complete_df, "used in fMRI")
    bonner_idx = get_indices_by_dataset(complete_df, "Bonner & Epstein, (2017)")
    elife_idx  = get_indices_by_dataset(complete_df, "elife")
    visact_idx = get_indices_by_dataset(complete_df, "Our stimuli")

    x, y = tsne_features[:, 0], tsne_features[:, 1]

    x_sun,    y_sun    = x[sun_idx],    y[sun_idx]
    x_fmri,   y_fmri   = x[fmri_idx],   y[fmri_idx]
    x_bonner, y_bonner = x[bonner_idx], y[bonner_idx]
    x_elife,  y_elife  = x[elife_idx],  y[elife_idx]
    x_visact, y_visact = x[visact_idx], y[visact_idx]

    fig, ax = plt.subplots(figsize=(15, 10))

    # SUN background
    scatter_group(
        ax, x_sun, y_sun,
        color="lightgray",
        label="SUN Attribute DB",
        size=35,
        alpha=0.5,
        edgecolor="none",
    )

    # Bonner & Epstein
    scatter_group(
        ax, x_bonner, y_bonner,
        color="darkcyan",
        label="Bonner & Epstein, (2017)",
        size=100,
    )

    # Groen et al. (eLife)
    scatter_group(
        ax, x_elife, y_elife,
        color="#a3b18a",
        label="Groen et al., (2018)",
        size=100,
    )

    # Our full stimulus set
    scatter_group(
        ax, x_visact, y_visact,
        color="peachpuff",
        label="Stimulus Set",
        size=100,
    )

    # fMRI subset
    scatter_group(
        ax, x_fmri, y_fmri,
        color="orange",
        label="Stimulus Set (fMRI)",
        size=100,
    )

    style_axes(ax)
    fig.tight_layout()
    plt.legend(
        fontsize=20,
        loc="upper left",
        title="Datasets",
        markerscale=1.5,
        title_fontsize=20,
        borderpad=0.2,
        labelspacing=0.4,
        handletextpad=0.1,
        bbox_to_anchor=(1.0, 1),
    )

    out_path = os.path.join(FIG_DIR, "tsne_all_sets_new_version.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)


# ===========================
# Plot 2: Action labels in t-SNE
# ===========================

def plot_action_labels_tsne():
    # Load data (OSF paths, same as above) 
    complete_df = pd.read_csv(TSNE_DF_OSF, index_col=0).reset_index()
    tsne_features = np.load(TSNE_FEATURES_OSF)

    # Remove Bonner & eLife, as in your original script
    complete_df = complete_df[complete_df["dataset"] != "Bonner & Epstein, (2017)"]
    complete_df = complete_df[complete_df["dataset"] != "elife"]

    # Indices for SUN vs "our" stimuli
    sun_idx = complete_df.index[complete_df["dataset"] == "Patterson et al., (2012)"].tolist()
    our_idx = complete_df.index[complete_df["dataset"] != "Patterson et al., (2012)"].tolist()

    sun_subset = complete_df[complete_df["dataset"] == "Patterson et al., (2012)"]
    our_subset = complete_df[complete_df["dataset"] != "Patterson et al., (2012)"]

    # Load action labels & balance them 
    mean_action_df = pd.read_csv(
        os.path.join(MEAN_BEHAVIOR_DIR, "mean_action_df.csv"),
        index_col=0,
    )
    mean_action_df_label = mean_action_df.idxmax(axis=1)
    mean_action_df_label_balanced = mean_action_df_label.copy()

    for index, row in mean_action_df.iterrows():
        if (row["walking"] > row["biking"]) & (row["biking"] > row["driving"]) & (row["biking"] > 0.5):
            mean_action_df_label_balanced[index] = "biking"
        if (row["walking"] > row["driving"]) & (row["driving"] > row["biking"]) & (row["driving"] > 0.5):
            mean_action_df_label_balanced[index] = "driving"

    
    action_list = list(mean_action_df_label_balanced.loc[our_subset["index"]])
    our_subset["VISACT_action"] = action_list

    # --- t-SNE coords ---
    x, y = tsne_features[:, 0], tsne_features[:, 1]

    x_sun, y_sun = x[sun_idx], y[sun_idx]
    x_our, y_our = x[our_idx], y[our_idx] 

    # --- Navigation-function labels over full df ---
    nav_functions = ["hiking", "climbing", "swimming", "sailing/ boating", "driving", "biking"]
    colors = ["#EEDD88", "#BBCC33", "#99DDFF", "#77AADD", "darkgrey", "#EE8866"]

    complete_df["label"] = complete_df[nav_functions].apply(
        lambda row: label_highest_action_row(row, nav_functions),
        axis=1,
    )

    sun_idx = complete_df.index[complete_df["dataset"] == "Patterson et al., (2012)"].tolist()
    no_action_idx = complete_df.index[complete_df["label"] == "no action"].tolist()
    action_idx = complete_df.index[complete_df["label"] != "no action"].tolist()

    action_subset = complete_df[complete_df["label"] != "no action"]

    x_no_action, y_no_action = x[no_action_idx], y[no_action_idx]
    x_action, y_action = x[action_idx], y[action_idx]

    # --- Plotting ---
    fig, ax = plt.subplots(1, figsize=(15, 10))

    # SUN background points
    sns.scatterplot(
        x=x_sun, y=y_sun,
        color="lightgray",
        edgecolor="none",
        s=35,
        alpha=0.5,
        label="SUN Attribute DB",
        ax=ax,
    )

    # Our stimuli colored by action label
    sns.scatterplot(
        x=x_action, y=y_action,
        hue=action_subset["label"],
        palette=colors,
        edgecolor="none",
        s=35,
        alpha=1,
        ax=ax,
    )

    style_axes(ax)
    fig.tight_layout()
    plt.legend(
        fontsize=20,
        loc="upper left",
        title="Actions",
        markerscale=1.5,
        title_fontsize=20,
        borderpad=0.2,
        labelspacing=0.4,
        handletextpad=0.1,
        bbox_to_anchor=(1.0, 1),
    )

    out_path = os.path.join(FIG_DIR, "tsne_with_action_labels_new_version.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)


# ===========================
# Main
# ===========================

if __name__ == "__main__":
    plot_all_datasets_tsne()
    plot_action_labels_tsne()
