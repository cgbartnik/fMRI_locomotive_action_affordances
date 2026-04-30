#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from scipy.spatial.distance import squareform
from statsmodels.stats.multitest import multipletests
from data_paths import get_dnn_rdm_root, rdm_collection_path


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def corr_variability_whole(mean_rdm, feature_rdm, iterations=5):
    """
    Bootstrap correlation between two RDMs.
    """
    rdm_corr_boots = []

    for _ in range(iterations):
        # Create a random index that respects the structure of an RDM
        rdm_idx = np.random.randint(0, len(mean_rdm), size=len(mean_rdm))

        # Subsample from both the reference and the feature RDM
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

    if not corrs:
        raise RuntimeError(f"No .npz layers found in {model_path}")

    highest_layer_idx = np.argmax(corrs)
    highest_layer_correlation = corrs[highest_layer_idx]
    highest_layer = layers[highest_layer_idx]
    p_value = p_values[highest_layer_idx]

    return highest_layer_correlation, highest_layer, p_value


def load_other_spaces():
    """
    Load RDMs of other spaces and compute their RDVs.

    Returns
    -------
    rdms : np.ndarray
        Array of RDM matrices (one per space).
    rdvs : list of np.ndarray
        List of vectorized RDMs (squareform).
    names : list of str
        Names of the spaces in the same order.
    """
    from scipy.spatial.distance import squareform

    base = rdm_collection_path()

    # Online behavior
    action_231_online_rdm = np.load(
        os.path.join(base, "Online_Behavior/VISACT_full/action_online_231_rdm.npy")
    )
    material_231_online_rdm = np.load(
        os.path.join(base, "Online_Behavior/VISACT_full/material_online_231_rdm.npy")
    )
    objects_231_online_rdm = np.load(
        os.path.join(base, "Online_Behavior/VISACT_full/objects_online_231_rdm.npy")
    )
    categories_231_online_rdm = np.load(
        os.path.join(base, "Online_Behavior/VISACT_full/categories_online_231_rdm.npy")
    )
    attributes_231_online_rdm = np.load(
        os.path.join(base, "Online_Behavior/VISACT_full/attributes_online_231_rdm.npy")
    )

    # ADE20k & Places365
    ade20k_231_online_rdm = np.load(
        os.path.join(base, "ADE20k/ade20k_231_rdm.npy")
    )
    Places365_231_online_rdm = np.load(
        os.path.join(base, "Places365/places_365_231_rdm.npy")
    )

    # SUN database
    SunDB_231_nav_func = np.load(
        os.path.join(base, "SUN_DB/VISACT_full/SUN_nav_func_231_rdm.npy")
    )
    SunDB_231_materials = np.load(
        os.path.join(base, "SUN_DB/VISACT_full/SUN_materials_231_rdm.npy")
    )
    SunDB_231_Spatial_env = np.load(
        os.path.join(base, "SUN_DB/VISACT_full/SUN_spatialEnv_231_rdm.npy")
    )

    names = [
        "Online Action",
        "Online Material",
        "Online Objects",
        "Online Categories",
        "Online Attributes",
        "ADE20k",
        "Places365",
        "SunDB Nav Func",
        "SunDB Materials",
        "SunDB Spatial Env",
    ]

    rdms = np.array(
        [
            action_231_online_rdm,
            material_231_online_rdm,
            objects_231_online_rdm,
            categories_231_online_rdm,
            attributes_231_online_rdm,
            ade20k_231_online_rdm,
            Places365_231_online_rdm,
            SunDB_231_nav_func,
            SunDB_231_materials,
            SunDB_231_Spatial_env,
        ]
    )

    rdvs = [squareform(rdm.round(5)) for rdm in rdms]
    return rdms, rdvs, names


def compute_bootstrap_pairwise_diffs(corr_df):
    """
    Compute pairwise bootstrap differences vs 'Online Action' and FDR-correct p-values.

    Parameters
    ----------
    corr_df : pd.DataFrame
        Must contain columns ['space', 'bootstrap'] where 'bootstrap'
        is a list/array of bootstrapped correlations.

    Returns
    -------
    results_df : pd.DataFrame
        Columns: model1, model2, significant, p-value
    """
    # Select pairs where the first space is "Online Action"
    pairs = [
        (corr_df["space"].iloc[i], corr_df["space"].iloc[j])
        for i in range(len(corr_df["space"]))
        for j in range(i + 1, len(corr_df["space"]))
        if corr_df["space"].iloc[i] == "Online Action"
    ]

    # Compute bootstrap differences for each pair
    bootstrapped_diffs = {
        f"{pair[0]} - {pair[1]}": np.subtract(
            np.array(corr_df[corr_df["space"] == pair[0]]["bootstrap"].values[0]),
            np.array(corr_df[corr_df["space"] == pair[1]]["bootstrap"].values[0]),
        )
        for pair in pairs
    }

    # Calculate p-values based on 95% CI excluding 0
    p_values = []
    for diffs in bootstrapped_diffs.values():
        ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])
        if ci_lower > 0 or ci_upper < 0:
            p_values.append(0.01)  # CI excludes 0: treat as "significant"
        else:
            p_values.append(0.5)   # CI includes 0: treat as "non-significant"

    # Apply FDR correction
    reject, pvals_corrected, _, _ = multipletests(
        p_values, alpha=0.05, method="fdr_bh"
    )

    # Pack into DataFrame
    results_df = pd.DataFrame(
        {
            "model1": [comp.split(" - ")[0] for comp in bootstrapped_diffs.keys()],
            "model2": [comp.split(" - ")[1] for comp in bootstrapped_diffs.keys()],
            "significant": reject,
            "p-value": pvals_corrected,
        }
    )

    return results_df


def plot_other_spaces_figure(corr_df, results_df, fig_path):
    """
    Create the bar plot of correlation for each space + bootstrap error,
    and draw lines between significantly different pairs.

    Parameters
    ----------
    corr_df : pd.DataFrame
        Columns: ['space', 'correlation', 'std_bootstrap'] at least.
    results_df : pd.DataFrame
        From compute_bootstrap_pairwise_diffs.
    fig_path : str
        File path to save the figure.
    """
    color_groups = [
        ("Behavioral annotations (Online)", "#709775", [0, 1, 2, 3, 4]),
        ("ADE20k Segmentation label Co-occurence", "#6bbf59", [5]),
        ("Places365", "#0b6e4f", [6]),
        ("Sun Attribute Database", "#08a045", [7, 8, 9]),
    ]

    x_positions = np.arange(len(corr_df["space"]))

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot bars for each color group
    for group_label, color, indices in color_groups:
        group_df = corr_df.iloc[indices]
        ax.bar(
            x_positions[indices],
            group_df["correlation"],
            yerr=group_df["std_bootstrap"],
            capsize=5,
            color=color,
            label=group_label,
        )

    # Draw lines between significantly different pairs (vs Online Action)
    start_value = 0.8  # initial y-level for significance lines

    for _, row in results_df.iterrows():
        if row["significant"]:
            idx1 = corr_df.index[corr_df["space"] == row["model1"]][0]
            idx2 = corr_df.index[corr_df["space"] == row["model2"]][0]
            x1 = x_positions[idx1]
            x2 = x_positions[idx2]
            y = start_value
            ax.plot([x1, x2], [y, y], color="black", linewidth=1)
            start_value += 0.01

    # X-ticks
    ax.set_xticks(x_positions)
    ax.set_xticklabels([
    "Loc. Affordances",
    "Materials",
    "Objects",
    "Categories",
    "Global Properties",
    "ADE20k",
    "Places365",
    "Loc. Affordances",
    "Materials",
    "Global Properties"
], rotation=45, ha='right', fontsize=12)


    # Customize axes
    ax.set_ylabel("Spearman's rho Correlation", fontsize=15)
    plt.yticks(fontsize=12)
    ax.set_ylim(0, 1.0)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    # Remove top/right frame
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------

def main():
    # Model and RDM path for the best model
    model_list = ["vit_base_patch16_224__VISACT_231_RDM"]
    RDM_path = get_dnn_rdm_root()

    # Where to save figure (Figures folder next to this script)
    script_root = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_root, "Figures")
    os.makedirs(figures_dir, exist_ok=True)
    fig_output_path = os.path.join(
        figures_dir, "DNN_CLIP_VIT_30_other_spaces.png"
    )

    # Load spaces
    RMDs, rdvs, RDM_names = load_other_spaces()

    # Build correlation dataframe
    records = []
    for idx, rdv in enumerate(rdvs):
        for model in model_list:
            model_name = model[:-11]  # keep your naming scheme
            model_dir = os.path.join(RDM_path, model)

            highest_layer_correlation, highest_layer, p_value = (
                get_highest_layer_correlation(model_dir, rdv)
            )

            # Load the RDM of the best layer
            DNN_RDM = np.load(
                os.path.join(model_dir, highest_layer)
            )["arr_0"]

            bootstrapped = corr_variability_whole(
                DNN_RDM, RMDs[idx], iterations=5
            )
            std_corr = np.std(bootstrapped)

            records.append(
                {
                    "model": model_name,
                    "space": RDM_names[idx],
                    "correlation": highest_layer_correlation,
                    "layer": highest_layer,
                    "p-value": p_value,
                    "std_bootstrap": std_corr,
                    "bootstrap": bootstrapped,
                }
            )

    corr = pd.DataFrame.from_records(records)

    # Group labels (if needed later)
    groups = [
        "Online",
        "Online",
        "Online",
        "Online",
        "Online",
        "ADE20k",
        "Places365",
        "SunDB",
        "SunDB",
        "SunDB",
    ]
    corr["group"] = groups

    # Save CSV (in the same directory as the script)
    csv_path = os.path.join(script_root, "DNN_other_spaces_bootstraps.csv")
    corr.to_csv(csv_path, index=False)
    print(f"Saved bootstrap correlation results to: {csv_path}")

    # Pairwise bootstrap differences & FDR
    results_df = compute_bootstrap_pairwise_diffs(corr)

    # Plot figure
    plot_other_spaces_figure(corr, results_df, fig_output_path)
    print(f"Saved figure to: {fig_output_path}")


if __name__ == "__main__":
    main()
