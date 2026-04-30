import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, ttest_1samp

from data_paths import get_dnn_rdm_root, brain_path



# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def get_highest_layer_correlation(model_path, sub_rdv):
    """
    For a given model directory and subject RDV, find the layer
    whose RDM correlates most strongly with the subject RDV.
    """
    corrs = []
    layers = []
    p_values = []

    for layer in sorted(os.listdir(model_path)):
        if layer.endswith(".npz"):
            layer_file = os.path.join(model_path, layer)
            try:
                layer_array = np.load(layer_file)["arr_0"]
            except KeyError:
                layer_array = np.load(layer_file)["rdm"]

            layer_rdv = squareform(layer_array.round(5))
            spearman = spearmanr(sub_rdv, layer_rdv)

            corrs.append(spearman.correlation)
            layers.append(layer)
            p_values.append(spearman.pvalue)

    if not corrs:
        raise RuntimeError(f"No .npz layers found in {model_path}")

    highest_layer_idx = np.argmax(corrs)
    highest_layer_correlation = corrs[highest_layer_idx]
    highest_layer = layers[highest_layer_idx]
    p_value = p_values[highest_layer_idx]

    return highest_layer_correlation, highest_layer, p_value


def compute_noise_ceilings(subject_rdms):
    """
    Compute lower and upper noise ceilings for a set of subject RDMs.

    Parameters
    ----------
    subject_rdms : array-like
        Shape: (n_subjects, n_cond, n_cond)

    Returns
    -------
    lower_bound : float
    upper_bound : float
    """
    num_subjects = len(subject_rdms)

    lower_bound_correlations = np.zeros(num_subjects)
    upper_bound_correlations = np.zeros(num_subjects)

    # Group-average RDM
    group_rdm = np.mean(subject_rdms, axis=0)

    # Correlation of each subject with the group-average (upper bound)
    group_correlations = [
        spearmanr(squareform(group_rdm), squareform(rdm)).correlation
        for rdm in subject_rdms
    ]

    for i in range(num_subjects):
        # Leave-one-out: exclude current subject
        other_subjects_rdms = np.delete(subject_rdms, i, axis=0)
        other_group_rdm = np.mean(other_subjects_rdms, axis=0)

        # Correlate excluded subject with the others' group RDM (lower bound)
        lower_bound_correlation = spearmanr(
            squareform(other_group_rdm),
            squareform(subject_rdms[i])
        ).correlation

        lower_bound_correlations[i] = lower_bound_correlation
        upper_bound_correlations[i] = group_correlations[i]

    lower_bound = np.mean(lower_bound_correlations)
    upper_bound = np.mean(upper_bound_correlations)

    return lower_bound, upper_bound


def run_roi_analysis(roi_label, rdm_file, model_list, dnn_rdm_root):
    """
    Run full RSA for one ROI:
    - load ROI RDMs
    - compute noise ceilings
    - correlate all models & layers
    - aggregate across subjects

    Returns
    -------
    corr : pd.DataFrame
        Subject-level correlations.
    overview_df : pd.DataFrame
        Model-level mean correlations and summary stats.
    """
    # Load ROI RDMs (subjects x n_cond x n_cond)
    roi_rdms = np.load(rdm_file)["arr_0"]

    # Subject-level RDVs
    brain_rdvs = [squareform(roi_rdms[subj].round(5)) for subj in range(roi_rdms.shape[0])]

    # Noise ceilings
    lower_noise_ceiling, upper_noise_ceiling = compute_noise_ceilings(roi_rdms)

    # ---------------------------------------------
    # Subject-level correlations (corr)
    # ---------------------------------------------
    corr_rows = []
    for subj, rdv in enumerate(brain_rdvs):
        sub_name = f"sub_0{subj + 1}"
        for model in model_list:
            model_name = model[:-11]
            model_dir = os.path.join(dnn_rdm_root, model)

            highest_layer_correlation, highest_layer, p_value = \
                get_highest_layer_correlation(model_dir, rdv)

            corr_rows.append(
                {
                    "model": model_name,
                    "roi": roi_label,
                    "subject": sub_name,
                    "correlation": highest_layer_correlation,
                    "layer": highest_layer,
                    "p-value": p_value,
                    "lower_nc": lower_noise_ceiling,
                    "higher_nc": upper_noise_ceiling,
                }
            )

    corr = pd.DataFrame.from_records(corr_rows)

    # ---------------------------------------------
    # Model-level summary (overview_df)
    # ---------------------------------------------
    overview_rows = []

    for model in model_list:
        model_name = model[:-11]
        corr_subset = corr[corr["model"] == model_name]

        # Mean & SEM
        mean_corr = corr_subset["correlation"].mean()
        sem_corr = corr_subset["correlation"].std() / np.sqrt(len(corr_subset["correlation"]))

        # Test vs zero
        t, p = ttest_1samp(corr_subset["correlation"], 0.0)

        # Noise ceilings
        lower_nc = corr_subset["lower_nc"].iloc[0]
        upper_nc = corr_subset["higher_nc"].iloc[0]

        # Most frequent winning layer
        most_layer = corr_subset["layer"].value_counts().index[0]

        print(f"{roi_label} | {model_name}: t={t.round(3)} p={p.round(3)}")

        overview_rows.append(
            {
                "model": model_name,
                "roi": roi_label,
                "subject": "mean",
                "correlation": mean_corr,
                "sem": sem_corr,
                "p-value": p,
                "lower_nc": lower_nc,
                "upper_nc": upper_nc,
                "most_layer": most_layer,
            }
        )

    overview_df = pd.DataFrame.from_records(overview_rows)
    return corr, overview_df


def plot_roi_bar(overview_df, corr, roi_title, fig_path, alpha=0.05, bonferroni_factor=1):
    """
    Create barplot for one ROI with:
    - bars + SEM
    - individual subject scatter
    - noise ceilings
    - Bonferroni-corrected significance stars
    """
    df = overview_df.copy()

    # Color groups and indices (assuming same order across models)
    color_groups = [
        ("CNN object classification", "#7209b7", [0, 1, 2]),
        ("CNN scene classification", "#480ca8", [3]),
        ("CNN scene segmentation", "#3f37c9", [4]),
        ("CNN alternative training objective", "#4361ee", [5, 6]),
        ("CNN video", "#4cc9f0", [7, 8]),
        ("ViT object classification", "#f48c06", [9]),
        ("ViT alternative training objective", "#ffba08", [10, 11, 12]),
    ]

    x_positions = np.arange(len(df["model"]))
    alpha_bonferroni = alpha / (len(df["model"]) * bonferroni_factor)

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")

    for group_label, color, indices in color_groups:
        group_df = df.iloc[indices]

        # Bars + SEM
        ax.bar(
            x_positions[indices],
            group_df["correlation"],
            yerr=group_df["sem"],
            capsize=5,
            color=color,
            label=group_label,
        )

        # Subject-level scatter
        for idx in indices:
            model_name = df.iloc[idx]["model"]
            model_df = corr[corr["model"] == model_name]
            ax.scatter(
                [x_positions[idx]] * len(model_df["correlation"]),
                model_df["correlation"],
                color="gray",
                alpha=0.5,
                s=10,
                zorder=10,
            )

        # Significance stars
        for i, p_val in enumerate(group_df["p-value"]):
            if p_val < alpha_bonferroni:
                ax.text(
                    x_positions[indices[i]],
                    df["upper_nc"][0] + 0.02,
                    "*",
                    ha="center",
                    fontsize=12,
                )

    # Noise ceilings band
    ax.fill_between(
        [-0.8, len(x_positions)],
        df["lower_nc"][0],
        df["upper_nc"][0],
        color="gray",
        alpha=0.2,
    )
    ax.axhline(
        df["upper_nc"][0],
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Upper Noise Ceiling",
    )
    ax.axhline(
        df["lower_nc"][0],
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Lower Noise Ceiling",
    )

    # Axes cosmetics
    ax.set_ylabel("Spearman's rho Correlation", fontsize=15)
    plt.yticks(fontsize=12)
    ax.set_title(roi_title)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(df["model"], rotation=45, ha="right")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_xlim(-0.8, len(x_positions))
    ax.set_ylim(0, 0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------

def main():
    # Shared DNN model list (order must match color_groups indices)
    model_list = [
        "AlexNet_VISACT_RDM",
        "VGG16_VISACT_RDM",
        "ResNet50_VISACT_RDM",
        "Places365_VISACT_RDM",
        "SceneParsing_VISACT_RDM",
        "DINO_VISACT_RDM",
        "CLIP_RN101_VISACT_RDM",
        "x3d_m_VISACT_RDM",
        "slowfast_r101_VISACT_RDM",
        "DINO_VIT_BASE_P16",
        "vit_base_patch16_384__VISACT_90_RDM",
        "CLIP_ViT-B_-_16_VISACT_RDM",
        "CLIP_ViT-B_-_32_VISACT_90_RDM",
    ]

    # Path to DNN RDM folders
    dnn_rdm_root = get_dnn_rdm_root()

    # ROI RDM files
    base_brain_path = brain_path("average")
    roi_configs = [
        {
            "roi_label": "PPA_mean",
            "title": "PPA",
            "rdm_file": os.path.join(base_brain_path, "fmri_PPA_mean.npz"),
            "fig_name": "DNNs_PPA_correlation.png",
            "bonferroni_factor": 3,  # as in your original PPA code
        },
        {
            "roi_label": "TOS_mean",
            "title": "OPA",
            "rdm_file": os.path.join(base_brain_path, "fmri_OPA_mean.npz"),
            "fig_name": "DNNs_OPA_correlation.png",
            "bonferroni_factor": 1,
        },
        {
            "roi_label": "RSC_mean",
            "title": "RSC",
            "rdm_file": os.path.join(base_brain_path, "fmri_RSC_mean.npz"),
            "fig_name": "DNNs_RSC_correlation.png",
            "bonferroni_factor": 1,
        },
    ]

    # Create Figures directory in the same folder as this script
    script_root = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_root, "Figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Run analysis per ROI
    for cfg in roi_configs:
        print("\n=== ROI:", cfg["title"], "===")
        corr, overview_df = run_roi_analysis(
            roi_label=cfg["roi_label"],
            rdm_file=cfg["rdm_file"],
            model_list=model_list,
            dnn_rdm_root=dnn_rdm_root,
        )

        fig_path = os.path.join(figures_dir, cfg["fig_name"])
        plot_roi_bar(
            overview_df,
            corr,
            roi_title=cfg["title"],
            fig_path=fig_path,
            alpha=0.05,
            bonferroni_factor=cfg["bonferroni_factor"],
        )
        print(f"Saved figure for {cfg['title']} to: {fig_path}")


if __name__ == "__main__":
    main()
