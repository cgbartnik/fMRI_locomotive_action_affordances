import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, sem
from scipy.stats import ttest_1samp
from .data_paths import (
    get_dnn_rdm_root,
    behavior_path,
    brain_path,
    rdm_collection_path,
)

# ------------------------------------------------------------------
# Set up paths relative to this script & make Figures directory
# ------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
figures_dir = os.path.join(script_dir, "Figures")
os.makedirs(figures_dir, exist_ok=True)

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
behavior_action_rdms = np.load(
    behavior_path("VISACT_fmri_behavior", "fmri_behavior_action_rdms.npz")
)["arr_0"]
behavior_object_rdms = np.load(
    behavior_path("VISACT_fmri_behavior", "fmri_behavior_object_rdms.npz")
)["arr_0"]

# subject mean rdvs
mean_action_rdm = np.mean(behavior_action_rdms, axis=0)
mean_object_rdm = np.mean(behavior_object_rdms, axis=0)

V1_mean_RDM = np.load(
    brain_path("Brain_RDMs", "mean_task_all_subs_V1_Juelich_maxprob-thr50-2mm_RDMs.npy")
)

PPA_mean_RDM = np.load(
    brain_path("average", "fmri_PPA_mean.npz")
)["arr_0"]
OPA_mean_RDM = np.load(
    brain_path("average", "fmri_OPA_mean.npz")
)["arr_0"]
RSC_mean_RDM = np.load(
    brain_path("average", "fmri_RSC_mean.npz")
)["arr_0"]

GIST_RDM = np.load(
    rdm_collection_path("GIST", "VISACT_fMRI", "GIST_1024_RDM_fMRI.npy")
)


def plot_model_correlation(model_name, layer_order, bar_color, behavioral_rdm, save_path, num_comparisons, xtick_labels=None):
    """
    Plots the correlation between model layers and behavioral/brain RDMs and saves the figure.

    Parameters:
    - model_name (str): The name of the model.
    - layer_order (list): Ordered list of layer filenames.
    - bar_color (str): Color code for the bar plot.
    - behavioral_rdm (np.ndarray): RDM array of shape (subjects, n, n) or (n, n) for a single mean RDM.
    - save_path (str): Filename for saving (will be prefixed with 'Supplementary_').
    - num_comparisons (int): Number of comparisons for Bonferroni correction.
    - xtick_labels (list, optional): Custom x-tick labels for the layers.
    """
    # If a single RDM (e.g., mean ROI RDM) is passed, wrap into a "single subject" dimension
    if behavioral_rdm.ndim == 2:
        behavioral_rdm = np.expand_dims(behavioral_rdm, axis=0)

    # Convert RDMs to RDVs
    beh_rdvs = [squareform(behavioral_rdm[subj].round(5)) for subj in range(behavioral_rdm.shape[0])]
    
    # Initialize DataFrame
    corr_df = pd.DataFrame(columns=["layer", "subject", "correlation", "p-value"])
    
    # Define model path
    model_root = get_dnn_rdm_root()
    model_path = os.path.join(model_root, model_name)
    
    # Compute correlations
    for layer_file in layer_order:
        layer_array = np.load(os.path.join(model_path, layer_file))["arr_0"]
        layer_rdv = squareform(layer_array.round(5))
        
        for subj_idx, beh_rdv in enumerate(beh_rdvs):
            spearman = spearmanr(beh_rdv, layer_rdv)
            if not np.isnan(spearman.correlation):
                corr_df = pd.concat([
                    corr_df,
                    pd.DataFrame({
                        "layer": [layer_file],
                        "subject": [subj_idx],
                        "correlation": [spearman.correlation],
                        "p-value": [spearman.pvalue],
                    })
                ], ignore_index=True)
    
    # Compute mean, SEM, and test if correlations are different from zero
    sem_df = corr_df.groupby("layer").agg({"correlation": ["mean", sem]}).reset_index()
    sem_df.columns = ["layer", "mean_correlation", "sem"]

    # Add t-test results
    t_test_results = corr_df.groupby("layer").apply(lambda x: ttest_1samp(x["correlation"], 0))
    sem_df["t_stat"] = [result.statistic for result in t_test_results]
    sem_df["p_value"] = [result.pvalue for result in t_test_results]

    sem_df = sem_df.set_index("layer").loc[layer_order].reset_index()

    print(sem_df)
    
    # Bonferroni correction threshold
    bonferroni_threshold = 0.05 / num_comparisons

    # Plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(
        sem_df["layer"],
        sem_df["mean_correlation"],
        yerr=sem_df["sem"],
        capsize=5,
        alpha=1,
        color=bar_color,
        label="Mean Correlation"
    )
    
    for layer in layer_order:
        layer_individuals = corr_df[corr_df["layer"] == layer]["correlation"]
        ax.scatter(
            [layer] * len(layer_individuals),
            layer_individuals,
            alpha=0.6,
            color="gray",
            label="Individual Correlations" if layer == layer_order[0] else ""
        )
    
    # Add asterisks for significance
    for i, layer in enumerate(layer_order):
        layer_p_value = sem_df.loc[sem_df["layer"] == layer, "p_value"].values[0]
        if layer_p_value < bonferroni_threshold:
            ax.text(i, 0.5, "*", ha='center', va='bottom', fontsize=12, color='black')

    ax.set_ylabel("Spearman's rho Correlation", fontsize=15)
    if xtick_labels and len(xtick_labels) == len(layer_order):
        ax.set_xticks(range(len(layer_order)))
        ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
    else:
        ax.set_xticklabels(sem_df["layer"], rotation=45)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.axhline(y=0, xmin=0, xmax=len(layer_order), color='black', linestyle='-', linewidth=1)
    plt.ylim(-0.1, 0.60)
    plt.tight_layout()

    # Save figure
    filename = "Supplementary_" + save_path
    full_path = os.path.join(figures_dir, filename)
    plt.savefig(full_path, dpi=300, transparent=True)
    print(f"Saved figure to: {full_path}")
    plt.close(fig)


if __name__ == "__main__":
    # ---------------- AlexNet ----------------
    model_name = "AlexNet_VISACT_RDM"
    layer_order_example = [
        "features.0.npz", "features.3.npz", "features.6.npz",
        "features.8.npz", "features.10.npz"
    ]
    bar_color_example = "#7209b7"
    x_tick_labels = [
        "features0", "features3", "features6",
        "features8", "features10"
    ]

    save_path = "AlexNet_affordance.svg"
    plot_model_correlation(
        model_name, layer_order_example, bar_color_example,
        behavior_action_rdms, save_path, 20, x_tick_labels
    )

    save_path = "AlexNet_object.svg"
    plot_model_correlation(
        model_name, layer_order_example, bar_color_example,
        behavior_object_rdms, save_path, 20, x_tick_labels
    )

    save_path = "AlexNet_PPA.svg"
    plot_model_correlation(
        model_name, layer_order_example, bar_color_example,
        PPA_mean_RDM, save_path, 20, x_tick_labels
    )

    save_path = "AlexNet_V1.svg"
    plot_model_correlation(
        model_name, layer_order_example, bar_color_example,
        V1_mean_RDM, save_path, 20, x_tick_labels
    )

    # ---------------- SlowFast ----------------
    model_name = "slowfast_r101_VISACT_RDM"
    layer_order_example = [
        "blocks.1_slow.npz",
        "blocks.1_fast.npz",
        "blocks.2_slow.npz",
        "blocks.2_fast.npz",
        "blocks.3_slow.npz",
        "blocks.3_fast.npz",
        "blocks.4_slow.npz",
        "blocks.4_fast.npz",
        "blocks.5_slow.npz",
        "blocks.5.npz",
        "blocks.6_slow.npz",
        "blocks.6.npz",
    ]
    bar_color_example = "#4cc9f0"
    x_tick_labels = [
        "blocks.1_slow",
        "blocks.1_fast",
        "blocks.2_slow",
        "blocks.2_fast",
        "blocks.3_slow",
        "blocks.3_fast",
        "blocks.4_slow",
        "blocks.4_fast",
        "blocks.5_slow",
        "blocks.5",
        "blocks.6_slow",
        "blocks.6",
    ]

    save_path = "SlowFast_affordance.svg"
    plot_model_correlation(
        model_name, layer_order_example, bar_color_example,
        behavior_action_rdms, save_path, 48, x_tick_labels
    )

    save_path = "SlowFast_object.svg"
    plot_model_correlation(
        model_name, layer_order_example, bar_color_example,
        behavior_object_rdms, save_path, 48, x_tick_labels
    )

    save_path = "SlowFast_PPA.svg"
    plot_model_correlation(
        model_name, layer_order_example, bar_color_example,
        PPA_mean_RDM, save_path, 48, x_tick_labels
    )

    save_path = "SlowFast_V1.svg"
    plot_model_correlation(
        model_name, layer_order_example, bar_color_example,
        V1_mean_RDM, save_path, 48, x_tick_labels
    )

    # ---------------- ViT base ----------------
    model_name = "vit_base_patch16_384__VISACT_90_RDM"
    layer_order_example = [
        "blocks.0.npz", "blocks.1.npz", "blocks.2.npz",
        "blocks.3.npz", "blocks.4.npz", "blocks.5.npz",
        "blocks.6.npz", "blocks.7.npz", "blocks.8.npz",
        "blocks.9.npz", "blocks.10.npz", "blocks.11.npz"
    ]
    bar_color_example = "#f48c06"
    x_tick_labels = [
        "blocks.0", "blocks.1", "blocks.2",
        "blocks.3", "blocks.4", "blocks.5",
        "blocks.6", "blocks.7", "blocks.8",
        "blocks.9", "blocks.10", "blocks.11"
    ]

    save_path = "ViT_base_affordances.svg"
    plot_model_correlation(
        model_name, layer_order_example, bar_color_example,
        behavior_action_rdms, save_path, 48, x_tick_labels
    )

    save_path = "ViT_base_object.svg"
    plot_model_correlation(
        model_name, layer_order_example, bar_color_example,
        behavior_object_rdms, save_path, 48, x_tick_labels
    )

    save_path = "ViT_base_PPA.svg"
    plot_model_correlation(
        model_name, layer_order_example, bar_color_example,
        PPA_mean_RDM, save_path, 48, x_tick_labels
    )

    save_path = "ViT_base_V1.svg"
    plot_model_correlation(
        model_name, layer_order_example, bar_color_example,
        V1_mean_RDM, save_path, 48, x_tick_labels
    )
