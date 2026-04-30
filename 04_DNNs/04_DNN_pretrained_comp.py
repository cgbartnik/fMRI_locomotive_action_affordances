import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, ttest_1samp
import scipy.stats as stats

from data_paths import behavior_path, rdm_collection_path, dnn_rdm_path


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


def get_highest_layer_correlation(model_path, sub_rdv):
    corrs = []
    layers = []
    p_values = []

    for layer in sorted(os.listdir(model_path)):
        if not layer.endswith(".npz"):
            continue

        layer_file = os.path.join(model_path, layer)

        try:
            layer_array = np.load(layer_file)["arr_0"]
            layer_rdv = squareform(layer_array.round(5))
        except KeyError:
            layer_rdv = np.load(layer_file)["rdm"]

        spearman = spearmanr(sub_rdv, layer_rdv)

        corrs.append(spearman.correlation)
        layers.append(layer)
        p_values.append(spearman.pvalue)

    corrs = np.array(corrs)
    layers = np.array(layers)
    p_values = np.array(p_values)

    valid = ~np.isnan(corrs)

    corrs = corrs[valid]
    layers = layers[valid]
    p_values = p_values[valid]

    if len(corrs) == 0:
        raise ValueError(f"No valid layer correlations found in: {model_path}")

    highest_layer_idx = np.argmax(corrs)

    return (
        corrs[highest_layer_idx],
        layers[highest_layer_idx],
        p_values[highest_layer_idx],
    )


def compute_pairwise_comparisons(corr_df, base_height, model_order):
    test_rows = []

    for i in range(len(model_order)):
        for j in range(i + 1, len(model_order)):
            model1 = model_order[i]
            model2 = model_order[j]

            corr1 = corr_df[corr_df["model"] == model1]["correlation"]
            corr2 = corr_df[corr_df["model"] == model2]["correlation"]

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

        x_pos_map = {name: idx for idx, name in enumerate(model_order)}

        test_result_df["pos_model1"] = test_result_df["model1"].map(x_pos_map)
        test_result_df["pos_model2"] = test_result_df["model2"].map(x_pos_map)

        test_result_df["y_pos"] = (
            base_height
            + 0.05 * test_result_df.groupby("model1").cumcount()
        )
    else:
        test_result_df = pd.DataFrame(
            columns=[
                "model1",
                "model2",
                "t-statistic",
                "p-value",
                "adjusted p-value",
                "significant (Bonferroni)",
                "pos_model1",
                "pos_model2",
                "y_pos",
            ]
        )

    return test_result_df


def run_training_analysis(
    behavior_file,
    behavior_label,
    output_filename,
    colors,
    figures_dir,
):
    behavior_rdms = np.load(
        behavior_path("VISACT_fmri_behavior", behavior_file)
    )["arr_0"]

    original_path = dnn_rdm_path("Places365_VISACT_RDM")
    untrained_path = rdm_collection_path(
        "RdmFiles", "fmri_clemens", "affordances", "untrained_places"
    )
    pretrained_path = rdm_collection_path("vit_RDMs", "one_unfrozen_pretrained")
    scratch_path = rdm_collection_path(
        "RdmFiles", "fmri_clemens", "affordances", "scratch", "val_loss_relu"
    )

    fold_models = [pretrained_path, scratch_path]
    fold_model_names = ["finetuned", "scratch"]

    single_models = [original_path, untrained_path]
    single_model_names = ["original", "untrained"]

    model_order = ["original", "finetuned", "scratch", "untrained"]

    beh_rdvs = [
        squareform(behavior_rdms[subj].round(5))
        for subj in range(behavior_rdms.shape[0])
    ]

    lower_noise_ceiling, upper_noise_ceiling = compute_noise_ceilings(behavior_rdms)

    corr_rows = []

    for fold in ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"]:
        for subj, rdv in enumerate(beh_rdvs):
            sub_name = f"sub_{subj + 1:02d}"

            for idx, model in enumerate(fold_models):
                model_name = fold_model_names[idx]
                model_path = os.path.join(model, fold)

                highest_layer_correlation, highest_layer, p_value = (
                    get_highest_layer_correlation(model_path, rdv)
                )

                corr_rows.append(
                    {
                        "model": model_name,
                        "behavior": behavior_label,
                        "subject": sub_name,
                        "fold": fold,
                        "correlation": highest_layer_correlation,
                        "layer": highest_layer,
                        "p-value": p_value,
                        "lower_nc": lower_noise_ceiling,
                        "higher_nc": upper_noise_ceiling,
                    }
                )

    for subj, rdv in enumerate(beh_rdvs):
        sub_name = f"sub_{subj + 1:02d}"

        for idx, model in enumerate(single_models):
            model_name = single_model_names[idx]

            highest_layer_correlation, highest_layer, p_value = (
                get_highest_layer_correlation(model, rdv)
            )

            corr_rows.append(
                {
                    "model": model_name,
                    "behavior": behavior_label,
                    "subject": sub_name,
                    "fold": "fold_0",
                    "correlation": highest_layer_correlation,
                    "layer": highest_layer,
                    "p-value": p_value,
                    "lower_nc": lower_noise_ceiling,
                    "higher_nc": upper_noise_ceiling,
                }
            )

    corr = pd.DataFrame(corr_rows)

    corr_grouped = (
        corr.groupby(["model", "behavior", "subject"])
        .agg(
            {
                "correlation": "mean",
                "p-value": "mean",
                "lower_nc": "mean",
                "higher_nc": "mean",
            }
        )
        .reset_index()
    )

    overview_rows = []

    for model_name in model_order:
        corr_subset = corr[corr["model"] == model_name]

        mean_corr = corr_subset["correlation"].mean()
        sem_corr = corr_subset["correlation"].std(ddof=1) / np.sqrt(len(corr_subset))

        t, p = ttest_1samp(corr_subset["correlation"], 0)

        print(f"{behavior_label} | {model_name}: t={t.round(2)} p={p.round(4)}")

        overview_rows.append(
            {
                "model": model_name,
                "behavior": behavior_label,
                "subject": "mean",
                "correlation": mean_corr,
                "sem": sem_corr,
                "p-value": p,
                "lower_nc": corr_subset["lower_nc"].iloc[0],
                "upper_nc": corr_subset["higher_nc"].iloc[0],
                "most_layer": corr_subset["layer"].value_counts().index[0],
            }
        )

    overview_df = pd.DataFrame(overview_rows)
    overview_df.index = overview_df["model"]
    df = overview_df.loc[model_order].copy()

    pairwise_test_pos = overview_df["upper_nc"].max() + 0.09
    test_result_df = compute_pairwise_comparisons(
        corr_grouped,
        pairwise_test_pos,
        model_order,
    )

    print(test_result_df)
    print(df)

    alpha = 0.05
    alpha_bonferroni = alpha / len(df["model"])

    fig, ax = plt.subplots(figsize=(5, 5), facecolor="white")

    x_positions = np.arange(len(df["model"]))

    color_groups = [
        ("original", colors["original"], [0]),
        ("finetuned", colors["finetuned"], [1]),
        ("scratch", colors["scratch"], [2]),
        ("untrained", colors["untrained"], [3]),
    ]

    for group_label, color, indices in color_groups:
        group_df = df.iloc[indices]

        ax.bar(
            x_positions[indices],
            group_df["correlation"],
            yerr=group_df["sem"],
            capsize=5,
            color=color,
            label=group_label,
        )

        for index in indices:
            model_name = df.iloc[index]["model"]
            model_df = corr_grouped[corr_grouped["model"] == model_name]

            ax.scatter(
                [x_positions[index]] * len(model_df),
                model_df["correlation"],
                color="gray",
                alpha=0.5,
                s=10,
                zorder=10,
            )

        for i, p_value in enumerate(group_df["p-value"]):
            if p_value < alpha_bonferroni:
                ax.text(
                    x_positions[indices[i]],
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
        label="Upper Noise Ceiling",
        linewidth=1,
    )

    ax.axhline(
        y=df["lower_nc"].iloc[0],
        color="gray",
        linestyle="--",
        label="Lower Noise Ceiling",
        linewidth=1,
    )

    for _, row in test_result_df.iterrows():
        if row["significant (Bonferroni)"]:
            pos1 = row["pos_model1"]
            pos2 = row["pos_model2"]

            ax.plot(
                [pos1, pos2],
                [pairwise_test_pos, pairwise_test_pos],
                color="black",
                linestyle="-",
            )

            pairwise_test_pos += 0.03

    ax.set_ylabel("Spearman's rho Correlation", fontsize=15)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        ["original", "finetuned", "scratch", "untrained"],
        rotation=45,
        ha="right",
    )

    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    ax.set_ylim(0, 1)
    ax.set_xlim(-0.8, len(x_positions))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    output_path = os.path.join(figures_dir, output_filename)
    plt.savefig(output_path, dpi=300, transparent=True)
    plt.close(fig)

    print(f"Saved figure to: {output_path}")


script_root = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(script_root, "Figures")
os.makedirs(figures_dir, exist_ok=True)


run_training_analysis(
    behavior_file="fmri_behavior_action_rdms.npz",
    behavior_label="action",
    output_filename="ResNet_training_action.png",
    colors={
        "original": "#480ca8",
        "finetuned": "#e10b16",
        "scratch": "#ff7f51",
        "untrained": "#d4cbb3",
    },
    figures_dir=figures_dir,
)


run_training_analysis(
    behavior_file="fmri_behavior_object_rdms.npz",
    behavior_label="object",
    output_filename="ResNet_training_object.png",
    colors={
        "original": "#480ca8",
        "finetuned": "#0077b6",
        "scratch": "#90e0ef",
        "untrained": "#d4cbb3",
    },
    figures_dir=figures_dir,
)

