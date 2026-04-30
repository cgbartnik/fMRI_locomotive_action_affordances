#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, ttest_1samp, ttest_rel
from scipy.spatial.distance import squareform
import scipy.stats as stats
from data_paths import get_dnn_rdm_root, behavior_path


def corr_variability_whole(mean_rdm, feature_rdm, iterations=10):
    rdm_corr_boots = []

    for _ in range(iterations):
        rdm_idx = np.random.randint(0, len(mean_rdm), size=len(mean_rdm))

        mean_rdm_re = (
            pd.DataFrame(mean_rdm)
            .reindex(rdm_idx)
            .reindex(rdm_idx, axis=1)
            .to_numpy()
        )
        feature_rdm_re = (
            pd.DataFrame(feature_rdm)
            .reindex(rdm_idx)
            .reindex(rdm_idx, axis=1)
            .to_numpy()
        )

        mean_vec = squareform(mean_rdm_re.round(5))
        mean_vec = np.where(mean_vec == 0, np.nan, mean_vec)

        feature_vec = squareform(feature_rdm_re.round(5))
        feature_vec = np.where(feature_vec == 0, np.nan, feature_vec)

        rdm_corr_boots.append(
            spearmanr(mean_vec, feature_vec, nan_policy="omit")[0]
        )

    return rdm_corr_boots


def get_highest_layer_correlation(model_path, sub_rdv):
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

    if len(corrs) == 0:
        raise ValueError(f"No .npz layer files found in: {model_path}")

    highest_layer_idx = np.nanargmax(corrs)

    return (
        corrs[highest_layer_idx],
        layers[highest_layer_idx],
        p_values[highest_layer_idx],
    )


def compute_noise_ceilings(subject_rdms):
    num_subjects = len(subject_rdms)

    lower_bound_correlations = np.zeros(num_subjects)
    upper_bound_correlations = np.zeros(num_subjects)

    group_rdm = np.mean(subject_rdms, axis=0)

    group_correlations = [
        spearmanr(squareform(group_rdm), squareform(rdm)).correlation
        for rdm in subject_rdms
    ]

    for i in range(num_subjects):
        other_subjects_rdms = np.delete(subject_rdms, i, axis=0)
        other_group_rdm = np.mean(other_subjects_rdms, axis=0)

        lower_bound_correlations[i] = spearmanr(
            squareform(other_group_rdm),
            squareform(subject_rdms[i]),
        ).correlation

        upper_bound_correlations[i] = group_correlations[i]

    return np.mean(lower_bound_correlations), np.mean(upper_bound_correlations)


def compute_pairwise_comparisons(corr_action, base_height, x_pos_map):
    test_rows = []
    models = corr_action["model"].unique()

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model1 = models[i]
            model2 = models[j]

            corr1 = corr_action[corr_action["model"] == model1]["correlation"]
            corr2 = corr_action[corr_action["model"] == model2]["correlation"]

            if len(corr1) > 0 and len(corr2) > 0:
                t_stat, p_val = stats.ttest_rel(corr1, corr2, nan_policy="omit")

                test_rows.append(
                    {
                        "model1": model1,
                        "model2": model2,
                        "t-statistic": t_stat,
                        "p-value": p_val,
                    }
                )

    test_result_df = pd.DataFrame(test_rows)

    num_tests = len(test_result_df)
    alpha = 0.05
    bonferroni_alpha = alpha / num_tests if num_tests > 0 else np.nan

    print("Bonferroni alpha:", bonferroni_alpha)
    print("Number of tests:", num_tests)

    if num_tests > 0:
        test_result_df["adjusted p-value"] = (
            test_result_df["p-value"] * num_tests
        ).clip(upper=1.0)

        test_result_df["significant (Bonferroni)"] = (
            test_result_df["adjusted p-value"] < bonferroni_alpha
        )

        test_result_df["pos_model1"] = test_result_df["model1"].map(x_pos_map)
        test_result_df["pos_model2"] = test_result_df["model2"].map(x_pos_map)
        test_result_df["y_pos"] = (
            base_height
            + 0.05 * test_result_df.groupby("model1").cumcount()
        )

    return test_result_df


def plot_behavior_bars(df, corr, title, output_path, use_bonferroni_for_stars=True):
    alpha = 0.05

    if use_bonferroni_for_stars:
        star_alpha = alpha / len(df["model"])
    else:
        star_alpha = alpha

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")

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
        valid_indices = [idx for idx in indices if idx < len(df)]
        group_df = df.iloc[valid_indices]

        ax.bar(
            x_positions[valid_indices],
            group_df["correlation"],
            yerr=group_df["sem"],
            capsize=5,
            color=color,
            label=group_label,
        )

        for idx in valid_indices:
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

        for local_i, p_value in enumerate(group_df["p-value"]):
            if p_value < star_alpha:
                global_idx = valid_indices[local_i]
                ax.text(
                    x_positions[global_idx],
                    df["upper_nc"].iloc[0] + 0.02,
                    "*",
                    ha="center",
                    fontsize=12,
                )

    ax.fill_between(
        [-0.8, len(x_positions)],
        df["lower_nc"].iloc[0],
        df["upper_nc"].iloc[0],
        color="gray",
        alpha=0.2,
    )

    ax.axhline(
        y=df["upper_nc"].iloc[0],
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Upper Noise Ceiling",
    )

    ax.axhline(
        y=df["lower_nc"].iloc[0],
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Lower Noise Ceiling",
    )

    ax.set_ylabel("Spearman's rho Correlation", fontsize=15)
    ax.set_title(title)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [
            "AlexNet (ImageNet;OC)",
            "VGG16 (ImageNet;OC)",
            "ResNet50 (ImageNet;OC)",
            "ResNet50 (Places365;SC)",
            "ResNet50 (ADEK20K;SS)",
            "DINO ResNet50 (ImageNet;OC)",
            "CLIP ResNet101 (WIT)",
            "X3D-M (Kinetics400;AR)",
            "SlowFast (Kinetics400;AR)",
            "DINO ViT-16 (ImageNet;OC)",
            "ViT Base Patch16",
            "CLIP ViT-B-16 (WIT)",
            "CLIP ViT-B-32 (WIT)",
        ][: len(x_positions)],
        rotation=45,
        ha="right",
    )

    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_ylim(0, 0.8)
    ax.set_xlim(-0.8, len(x_positions))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, transparent=True)
    plt.close(fig)


def run_behavior_analysis(
    behavior_name,
    behavior_rdm_path,
    model_list,
    rdm_root,
    fig_output_path,
    fig_title,
    use_bonferroni_for_stars=True,
):
    behavior_rdms = np.load(behavior_rdm_path)["arr_0"]

    beh_rdvs = [
        squareform(behavior_rdms[subj].round(5))
        for subj in range(behavior_rdms.shape[0])
    ]

    lower_noise_ceiling, upper_noise_ceiling = compute_noise_ceilings(behavior_rdms)

    corr_rows = []

    for subj, rdv in enumerate(beh_rdvs):
        sub_name = f"sub_{subj + 1:02d}"

        for model in model_list:
            model_name = model[:-11]
            model_dir = os.path.join(rdm_root, model)

            highest_layer_correlation, highest_layer, p_value = (
                get_highest_layer_correlation(model_dir, rdv)
            )

            corr_rows.append(
                {
                    "model": model_name,
                    "behavior": behavior_name,
                    "subject": sub_name,
                    "correlation": highest_layer_correlation,
                    "layer": highest_layer,
                    "p-value": p_value,
                    "lower_nc": lower_noise_ceiling,
                    "higher_nc": upper_noise_ceiling,
                }
            )

    corr = pd.DataFrame(corr_rows)

    overview_rows = []

    for model in model_list:
        model_name = model[:-11]
        corr_subset = corr[corr["model"] == model_name]

        mean_corr = corr_subset["correlation"].mean()
        sem_corr = corr_subset["correlation"].std(ddof=1) / np.sqrt(len(corr_subset))

        t, p = ttest_1samp(corr_subset["correlation"], 0)

        print(f"{behavior_name} | {model_name} t={t.round(2)} p={p.round(4)}")

        lower_nc = corr_subset["lower_nc"].iloc[0]
        upper_nc = corr_subset["higher_nc"].iloc[0]
        most_layer = corr_subset["layer"].value_counts().index[0]

        overview_rows.append(
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
            }
        )

    overview_df = pd.DataFrame(overview_rows)

    plot_behavior_bars(
        overview_df,
        corr,
        fig_title,
        fig_output_path,
        use_bonferroni_for_stars=use_bonferroni_for_stars,
    )

    return corr, overview_df


def main():
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

    script_root = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_root, "Figures")
    os.makedirs(figures_dir, exist_ok=True)

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
        use_bonferroni_for_stars=False,
    )

    all_object_corrs = overview_df_object["correlation"]

    print("OBJECT mean corr:", round(overview_df_object["correlation"].mean(), 3))
    print("OBJECT std corr:", round(overview_df_object["correlation"].std(), 2))

    diff = overview_df_object["correlation"] - overview_df_action["correlation"]

    print("Mean (object - action):", round(diff.mean(), 2))
    print("Std  (object - action):", round(diff.std(), 2))

    vit_name = "vit_base_patch16_384__VI"

    action_corr_vit = corr_action[corr_action["model"] == vit_name]["correlation"]
    object_corr_vit = corr_object[corr_object["model"] == vit_name]["correlation"]

    if len(action_corr_vit) > 0 and len(object_corr_vit) > 0:
        t_vit, p_vit = ttest_rel(object_corr_vit, action_corr_vit)
        print(
            f"ViT Base Patch16 Action vs Object: "
            f"t={t_vit.round(2)} p={p_vit.round(4)}"
        )
    else:
        print("Warning: No entries found for vit_base_patch16_384__VI.")

    t_avg, p_avg = ttest_rel(all_object_corrs, all_action_corrs)

    print(f"AVG DNNs for action vs object: t={t_avg.round(2)} p={p_avg.round(4)}")

    print("\nCounts per model (ACTION):")
    print(corr_action["model"].value_counts())

    print("\nCounts per model (OBJECT):")
    print(corr_object["model"].value_counts())


if __name__ == "__main__":
    main()