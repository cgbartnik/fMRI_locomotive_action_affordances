#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, ttest_1samp, ttest_rel
from scipy.spatial.distance import squareform
import scipy.stats as stats
from .data_paths import get_dnn_rdm_root, behavior_path


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def corr_variability_whole(mean_rdm, feature_rdm, iterations=10):
    """Bootstrap correlation between two RDMs."""
    rdm_corr_boots = []

    for _ in range(iterations):
        # Create a random index that respects the structure of an RDM
        rdm_idx = np.random.randint(0, len(mean_rdm), size=len(mean_rdm))

        # Subsample from both the reference and the feature RDM
        mean_rdm_re = pd.DataFrame(mean_rdm).reindex(rdm_idx).reindex(rdm_idx, axis=1).to_numpy()
        feature_rdm_re = pd.DataFrame(feature_rdm).reindex(rdm_idx).reindex(rdm_idx, axis=1).to_numpy()

        # Replace zeros with NaNs
        mean_vec = squareform(mean_rdm_re.round(5))
        mean_vec = np.where(mean_vec == 0, np.nan, mean_vec)
        feature_vec = squareform(feature_rdm_re.round(5))
        feature_vec = np.where(feature_vec == 0, np.nan, feature_vec)

        # Spearman correlation
        rdm_corr_boots.append(spearmanr(mean_vec, feature_vec, nan_policy="omit")[0])

    return rdm_corr_boots


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


def compute_pairwise_comparisons(corr_action, base_height, x_pos_map):
    """
    Compute pairwise t-tests between model correlations with Bonferroni correction.

    Parameters
    ----------
    corr_action : pd.DataFrame
        Must have columns ['model', 'correlation'].
    base_height : float
        Base height for plotting significance bars.
    x_pos_map : dict
        Mapping from model name -> x position (float/int) for plotting.

    Returns
    -------
    test_result_df : pd.DataFrame
        Columns: model1, model2, t-statistic, p-value, adjusted p-value,
                 significant (Bonferroni), pos_model1, pos_model2, y_pos
    """
    test_result_df = pd.DataFrame(columns=[
        "model1", "model2", "t-statistic", "p-value"
    ])

    models = corr_action["model"].unique()

    # Pairwise dependent t-tests (within-subject models)
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model1 = models[i]
            model2 = models[j]

            corr1 = corr_action[corr_action["model"] == model1]["correlation"]
            corr2 = corr_action[corr_action["model"] == model2]["correlation"]

            if len(corr1) > 0 and len(corr2) > 0:
                t_stat, p_val = stats.ttest_rel(
                    corr1, corr2, nan_policy="omit"
                )
                test_result_df = test_result_df.append(
                    {
                        "model1": model1,
                        "model2": model2,
                        "t-statistic": t_stat,
                        "p-value": p_val,
                    },
                    ignore_index=True,
                )

    num_tests = len(test_result_df)
    alpha = 0.05
    bonferroni_alpha = alpha / num_tests if num_tests > 0 else np.nan
    print("Bonferroni alpha:", bonferroni_alpha)
    print("Number of tests:", num_tests)

    # Bonferroni-corrected p-values
    test_result_df["adjusted p-value"] = test_result_df["p-value"] * num_tests
    test_result_df["adjusted p-value"] = test_result_df["adjusted p-value"].clip(
        upper=1.0
    )
    test_result_df["significant (Bonferroni)"] = (
        test_result_df["adjusted p-value"] < bonferroni_alpha
    )

    # Positions for plotting
    test_result_df["pos_model1"] = test_result_df["model1"].map(x_pos_map)
    test_result_df["pos_model2"] = test_result_df["model2"].map(x_pos_map)

    # Stacked y positions per model
    test_result_df["y_pos"] = base_height + 0.05 * test_result_df.groupby("model1").cumcount()

    return test_result_df


def plot_behavior_bars(df, corr, title, output_path, use_bonferroni_for_stars=True):
    """
    Create the bar plot with individual subject scatter and noise ceilings.
    """
    alpha = 0.05
    if use_bonferroni_for_stars:
        star_alpha = alpha / len(df["model"])
    else:
        star_alpha = alpha

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")

    # Define color groups and labels
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

    for group_label, color, indices in color_groups:
        group_df = df.iloc[indices]

        # Bars with SEM error bars
        ax.bar(
            x_positions[indices],
            group_df["correlation"],
            yerr=group_df["sem"],
            capsize=5,
            color=color,
            label=group_label,
        )

        # Individual subject points
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

        # Asterisks for significant models
        for i, p_value in enumerate(group_df["p-value"]):
            if p_value < star_alpha:
                ax.text(
                    x_positions[indices[i]],
                    df["upper_nc"][0] + 0.02,
                    "*",
                    ha="center",
                    fontsize=12,
                )

    # Noise ceilings
    ax.fill_between(
        [-0.8, len(x_positions)],
        df["lower_nc"][0],
        df["upper_nc"][0],
        color="gray",
        alpha=0.2,
    )
    ax.axhline(
        y=df["upper_nc"][0],
        xmin=-0.8,
        xmax=len(x_positions),
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Upper Noise Ceiling",
    )
    ax.axhline(
        y=df["lower_nc"][0],
        xmin=-0.8,
        xmax=len(x_positions),
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Lower Noise Ceiling",
    )

    # Labels, ticks, limits
    ax.set_ylabel("Spearman's rho Correlation", fontsize=15)
    ax.set_title(title)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [
            "AlexNet (ImageNet;OC)",
            "VGG16 (ImageNet;OC)",
            "ResNet50 (ImageNet;OC)",
            "ResNet50 (Places365;SC)",
            "ResNet50 (ADEK20K,;SS)",
            "DINO ResNet50 (ImageNet;OC)",
            "CLIP ResNet101 (WIT)",
            "X3D-M (Kinetics400;AR)",
            "SlowFast (Kinetics400;AR)",
            "DINO ViT-16 (ImageNet;OC)",
            "ViT Base Path16",
            "CLIP ViT-B-16 (WIT)",
            "CLIP ViT-B-32 (WIT)",
        ],
        rotation=45,
        ha="right",
    )

    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_ylim(0, 0.8)
    ax.set_xlim(-0.8, len(x_positions))

    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, transparent=True)
    plt.close(fig)  # avoid popping up if run in some environments


def run_behavior_analysis(
    behavior_name,
    behavior_rdm_path,
    model_list,
    rdm_root,
    fig_output_path,
    fig_title,
    use_bonferroni_for_stars=True,
):
    """
    Generic function that runs the full behavior analysis:
    - loads RDMs
    - computes noise ceilings
    - correlates all models & layers with behavior
    - aggregates across subjects
    - creates plot
    """
    # Load behavior RDMs
    behavior_rdms = np.load(behavior_rdm_path)["arr_0"]

    # Prepare correlation dataframe
    corr = pd.DataFrame(
        columns=[
            "model",
            "behavior",
            "subject",
            "correlation",
            "layer",
            "p-value",
            "lower_nc",
            "higher_nc",
        ]
    )

    beh_rdvs = [
        squareform(behavior_rdms[subj].round(5))
        for subj in range(behavior_rdms.shape[0])
    ]

    lower_noise_ceiling, upper_noise_ceiling = compute_noise_ceilings(behavior_rdms)

    # Subject loop
    for subj, rdv in enumerate(beh_rdvs):
        sub_name = f"sub_0{subj + 1}"

        for model in model_list:
            model_name = model[:-11]  # keep your original naming convention
            model_dir = os.path.join(rdm_root, model)

            highest_layer_correlation, highest_layer, p_value = get_highest_layer_correlation(
                model_dir, rdv
            )

            corr = corr.append(
                {
                    "model": model_name,
                    "behavior": behavior_name,
                    "subject": sub_name,
                    "correlation": highest_layer_correlation,
                    "layer": highest_layer,
                    "p-value": p_value,
                    "lower_nc": lower_noise_ceiling,
                    "higher_nc": upper_noise_ceiling,
                },
                ignore_index=True,
            )

    # Overview across subjects
    overview_df = pd.DataFrame(
        columns=[
            "model",
            "behavior",
            "subject",
            "correlation",
            "sem",
            "p-value",
            "lower_nc",
            "upper_nc",
            "most_layer",
        ]
    )

    for model in model_list:
        model_name = model[:-11]
        corr_subset = corr[corr["model"] == model_name]

        # Mean & SEM
        mean_corr = np.mean(corr_subset["correlation"])
        sem_corr = np.std(corr_subset["correlation"]) / np.sqrt(
            len(corr_subset["correlation"])
        )

        # Test vs zero
        t, p = ttest_1samp(corr_subset["correlation"], 0)
        print(f"{behavior_name} | {model_name} t={t.round(2)} p={p.round(4)}")

        lower_nc = corr_subset["lower_nc"].iloc[0]
        upper_nc = corr_subset["higher_nc"].iloc[0]

        # Most frequent best layer
        most_layer = corr_subset["layer"].value_counts().index[0]

        overview_df = overview_df.append(
            {
                "model": model_name,
                "behavior": behavior_name,
                "subject": "mean",
                "correlation": mean_corr,
                "sem": sem_corr,
                "p-value": p,
                "lower_nc": lower_nc,
                "upper_nc": upper_nc,
                "most_layer": most_layer,
            },
            ignore_index=True,
        )

    # Plot & save
    plot_behavior_bars(
        overview_df,
        corr,
        fig_title,
        fig_output_path,
        use_bonferroni_for_stars=use_bonferroni_for_stars,
    )

    return corr, overview_df


# ---------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------

def main():
    # Common configuration
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
        "vit_base_patch16_384__VISACT_90_RDM",
        "DINO_VIT_BASE_P16",
        "CLIP_ViT-B_-_16_VISACT_RDM",
        "CLIP_ViT-B_-_32_VISACT_90_RDM",
    ]

    rdm_root = get_dnn_rdm_root()

    # -----------------------------------------------------------------
    # Figure output directory: <root of this script>/Figures
    # -----------------------------------------------------------------
    script_root = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_root, "Figures")
    os.makedirs(figures_dir, exist_ok=True)

    # -----------------------------------------------------------------
    # ACTION behavior
    # -----------------------------------------------------------------
    action_behavior_rdms_path = behavior_path(
        "VISACT_fmri_behavior",
        "fmri_behavior_action_rdms.npz",
    )
    action_fig_path = os.path.join(figures_dir, "behavior_action_fMRI.png")

    corr_action, overview_df_action = run_behavior_analysis(
        behavior_name="action",
        behavior_rdm_path=action_behavior_rdms_path,
        model_list=model_list,
        rdm_root=rdm_root,
        fig_output_path=action_fig_path,
        fig_title="Action Behavior in fMRI",
        use_bonferroni_for_stars=True,
    )

    all_action_corrs = overview_df_action["correlation"]
    print("ACTION mean corr:", round(overview_df_action["correlation"].mean(), 3))
    print("ACTION std corr:", round(overview_df_action["correlation"].std(), 2))

    # -----------------------------------------------------------------
    # OBJECT behavior
    # -----------------------------------------------------------------
    object_behavior_rdms_path = behavior_path(
        "VISACT_fmri_behavior",
        "fmri_behavior_object_rdms.npz",
    )
    object_fig_path = os.path.join(figures_dir, "behavior_object_fMRI.png")

    corr_object, overview_df_object = run_behavior_analysis(
        behavior_name="object",
        behavior_rdm_path=object_behavior_rdms_path,
        model_list=model_list,
        rdm_root=rdm_root,
        fig_output_path=object_fig_path,
        fig_title="Object Behavior in fMRI",
        use_bonferroni_for_stars=False,  # matches your original code
    )

    all_object_corrs = overview_df_object["correlation"]
    print("OBJECT mean corr:", round(overview_df_object["correlation"].mean(), 3))
    print("OBJECT std corr:", round(overview_df_object["correlation"].std(), 2))

    # -----------------------------------------------------------------
    # Extra stats: action vs object
    # -----------------------------------------------------------------

    # Difference object - action across models (overview)
    diff = overview_df_object["correlation"] - overview_df_action["correlation"]
    print("Mean (object - action):", round(diff.mean(), 2))
    print("Std  (object - action):", round(diff.std(), 2))

    # ViT Base Patch16: action vs object
    vit_name = "vit_base_patch16_384__VI"
    action_corr_vit = corr_action[corr_action["model"] == vit_name]["correlation"]
    object_corr_vit = corr_object[corr_object["model"] == vit_name]["correlation"]

    if len(action_corr_vit) > 0 and len(object_corr_vit) > 0:
        t_vit, p_vit = ttest_rel(object_corr_vit, action_corr_vit)
        print(
            f"ViT Base Patch16 Action vs Object: t={t_vit.round(2)} "
            f"p={p_vit.round(4)}"
        )
    else:
        print("Warning: No entries found for vit_base_patch16_384__VI.")

    # Average across all models: action vs object
    t_avg, p_avg = ttest_rel(all_object_corrs, all_action_corrs)
    print(
        f"AVG DNNs for action vs object: t={t_avg.round(2)} p={p_avg.round(4)}"
    )

    # Optional: show value counts if you still want that info
    print("\nCounts per model (ACTION):")
    print(corr_action["model"].value_counts())
    print("\nCounts per model (OBJECT):")
    print(corr_object["model"].value_counts())


if __name__ == "__main__":
    main()
