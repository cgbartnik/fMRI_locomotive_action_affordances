from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.stats import ttest_1samp, ttest_rel, spearmanr
import pingouin as pg
from data_paths import behavior_path, brain_path, rdm_collection_path

# ---------------------------------------------------------------------
# PATHS & CONSTANTS
# ---------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "Figures")
os.makedirs(FIG_DIR, exist_ok=True)

BEHAVIOR_ACTION_PATH = behavior_path(
    "VISACT_fmri_behavior", "fmri_behavior_action_rdms.npz"
)
BEHAVIOR_OBJECT_PATH = behavior_path(
    "VISACT_fmri_behavior", "fmri_behavior_object_rdms.npz"
)

BRAIN_BASE_PATH = brain_path("average")

ROI_NAMES = ["PPA_mean", "OPA_mean", "RSC_mean"]  # file names
ROI_LABELS = ["PPA", "OPA", "RSC"]                 # pretty x-tick labels


# ---------------------------------------------------------------------
# CORE UTILITIES
# ---------------------------------------------------------------------
def compute_noise_ceilings(subject_rdms: np.ndarray) -> tuple[float, float]:
    """
    Compute lower and upper noise ceilings from subject RDMs.

    subject_rdms : (n_subj, n_cond, n_cond)
    """
    from scipy.spatial.distance import squareform
    from scipy.stats import spearmanr

    n_subj = subject_rdms.shape[0]
    lower = np.zeros(n_subj)
    upper = np.zeros(n_subj)

    group_rdm = subject_rdms.mean(axis=0)
    group_rdv = squareform(group_rdm)

    group_corrs = [
        spearmanr(group_rdv, squareform(rdm)).correlation for rdm in subject_rdms
    ]

    for i in range(n_subj):
        others = np.delete(subject_rdms, i, axis=0)
        other_mean = others.mean(axis=0)
        other_rdv = squareform(other_mean)

        lower[i] = spearmanr(other_rdv, squareform(subject_rdms[i])).correlation
        upper[i] = group_corrs[i]

    return lower.mean(), upper.mean()


def load_behavior_rdms(main_space: str) -> np.ndarray:
    """Return subject x 90 x 90 behavioral RDMs for 'action' or 'object'."""
    if main_space == "action":
        return np.load(BEHAVIOR_ACTION_PATH)["arr_0"]
    elif main_space == "object":
        return np.load(BEHAVIOR_OBJECT_PATH)["arr_0"]
    else:
        raise ValueError("main_space must be 'action' or 'object'.")


def load_brain_rdms(roi_name: str) -> tuple[np.ndarray, list[np.ndarray], float, float]:
    """
    Load average fMRI RDMs for a given ROI.

    Returns
    -------
    mean_rdms : (n_subj, 90, 90)
    brain_rdvs : list of 1D rdvs
    lower_nc, upper_nc : noise ceilings
    """
    path = os.path.join(BRAIN_BASE_PATH, f"fmri_{roi_name}.npz")
    mean_rdms = np.load(path)["arr_0"]
    brain_rdvs = [squareform(mean_rdms[s].round(5)) for s in range(mean_rdms.shape[0])]
    lower_nc, upper_nc = compute_noise_ceilings(mean_rdms)
    return mean_rdms, brain_rdvs, lower_nc, upper_nc


def load_space_rdm(space_path: str) -> np.ndarray:
    """Load external space RDM (either .npz with arr_0 or plain .npy)."""
    arr = np.load(space_path)
    if isinstance(arr, np.lib.npyio.NpzFile):
        arr = arr["arr_0"]
    return arr


# ---------------------------------------------------------------------
# PARTIAL CORRELATIONS: main_space vs another space
# ---------------------------------------------------------------------
def compute_and_plot_partial_correlations(
    main_space: str,
    space_path: str,
    space_name: str,
    plot_color: str,
    save_name: str,
    num_comparisons: int,
) -> None:
    """
    Partial correlations between fMRI and:
        - main behavior space (action/object)
        - another global space (attributes, objects, ...)

    brain ~ main_space | global_space
    brain ~ global_space | main_space
    """

    # --- choose behavior RDMs & base color / label ---
    if main_space == "action":
        behavior_rdms = load_behavior_rdms("action")
        base_color = "#E36858"
        main_label = "nav affordances"
    elif main_space == "object":
        behavior_rdms = load_behavior_rdms("object")
        base_color = "#6FA1BB"
        main_label = "objects"
    else:
        raise ValueError("main_space must be 'action' or 'object'.")

    bar_colors = [base_color, plot_color]

    # --- mean behavioral & global-space RDVs ---
    mean_behavior_rdm = behavior_rdms.mean(axis=0)
    mean_behavior_rdv = squareform(mean_behavior_rdm.round(5))

    global_rdm = load_space_rdm(space_path)
    mean_global_rdv = squareform(global_rdm.round(5))

    # -----------------------------------------------------------------
    # 1) Partial correlations: build corr_df
    # -----------------------------------------------------------------
    rows = []

    for roi_name in ROI_NAMES:
        _, brain_rdvs, lower_nc, upper_nc = load_brain_rdms(roi_name)

        for subj, brain_rdv in enumerate(brain_rdvs):
            sub_name = f"sub_{subj + 1:02d}"

            df = pd.DataFrame(
                {
                    "brain": brain_rdv,
                    "behavior": mean_behavior_rdv,
                    "global": mean_global_rdv,
                }
            )

            # brain ~ behavior | global
            pc_behavior = pg.partial_corr(
                data=df,
                x="brain",
                y="behavior",
                covar="global",
                method="spearman",
            ).round(3)

            rows.append(
                {
                    "behavior": "behavior",
                    "roi": roi_name,
                    "subject": sub_name,
                    "correlation": pc_behavior["r"].values[0],
                    "p-value": pc_behavior["p-val"].values[0],
                    "lower_nc": lower_nc,
                    "higher_nc": upper_nc,
                }
            )

            # brain ~ global | behavior
            pc_global = pg.partial_corr(
                data=df,
                x="brain",
                y="global",
                covar="behavior",
                method="spearman",
            ).round(3)

            rows.append(
                {
                    "behavior": "global",
                    "roi": roi_name,
                    "subject": sub_name,
                    "correlation": pc_global["r"].values[0],
                    "p-value": pc_global["p-val"].values[0],
                    "lower_nc": lower_nc,
                    "higher_nc": upper_nc,
                }
            )

    corr_df = pd.DataFrame(rows)

    # -----------------------------------------------------------------
    # 2) Paired t-tests behavior vs global per ROI
    # -----------------------------------------------------------------
    test_rows = []
    for roi_name in ROI_NAMES:
        beh_vals = corr_df[
            (corr_df["roi"] == roi_name) & (corr_df["behavior"] == "behavior")
        ]["correlation"]
        glob_vals = corr_df[
            (corr_df["roi"] == roi_name) & (corr_df["behavior"] == "global")
        ]["correlation"]

        t_stat, p_val = ttest_rel(beh_vals, glob_vals)
        test_rows.append(
            {
                "roi": roi_name,
                "behavior": "behavior_vs_global",
                "t-statistic": t_stat,
                "p-value": p_val,
            }
        )

    test_result_df = pd.DataFrame(test_rows)
    print("\nPaired t-tests behavior vs global:")
    print(test_result_df.round(3))

    # -----------------------------------------------------------------
    # 3) Overview: mean, SEM, test vs 0
    # -----------------------------------------------------------------
    overview_rows = []
    for label in ["behavior", "global"]:
        for roi_name in ROI_NAMES:
            subset = corr_df[
                (corr_df["roi"] == roi_name) & (corr_df["behavior"] == label)
            ]
            if subset.empty:
                continue

            mean_corr = subset["correlation"].mean()
            sem_corr = subset["correlation"].std(ddof=1) / np.sqrt(len(subset))
            t_stat, p_val = ttest_1samp(subset["correlation"], 0)

            lower_nc = subset["lower_nc"].iloc[0]
            upper_nc = subset["higher_nc"].iloc[0]

            print(
                f"{label} {roi_name}: mean={mean_corr:.3f}, t={t_stat:.2f}, p={p_val:.3f}"
            )

            overview_rows.append(
                {
                    "behavior": label,
                    "roi": roi_name,
                    "subject": "mean",
                    "correlation": mean_corr,
                    "sem": sem_corr,
                    "p-value": p_val,
                    "lower_nc": lower_nc,
                    "upper_nc": upper_nc,
                }
            )

    overview_df = pd.DataFrame(overview_rows)

    # -----------------------------------------------------------------
    # 4) Plot
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))

    alpha = 0.05
    alpha_bonferroni = alpha / num_comparisons
    alpha_bonferroni_pairwise = alpha / num_comparisons
    fontsize = 15

    rois = ROI_NAMES  # keep fixed order
    x = np.arange(len(rois))
    bar_width = 0.4

    pairwise_test_pos = overview_df["upper_nc"].max() + 0.06
    scatter_color = "gray"
    scatter_alpha = 0.5

    for i, roi_name in enumerate(rois):
        # overview rows
        beh_row = overview_df[
            (overview_df["roi"] == roi_name) & (overview_df["behavior"] == "behavior")
        ]
        glob_row = overview_df[
            (overview_df["roi"] == roi_name) & (overview_df["behavior"] == "global")
        ]

        # paired test result
        test_row = test_result_df[test_result_df["roi"] == roi_name]

        # bars
        ax.bar(
            i - bar_width / 2,
            beh_row["correlation"],
            bar_width,
            yerr=beh_row["sem"],
            color=bar_colors[0],
            label=main_label if i == 0 else "",
        )
        ax.bar(
            i + bar_width / 2,
            glob_row["correlation"],
            bar_width,
            yerr=glob_row["sem"],
            color=bar_colors[1],
            label=space_name if i == 0 else "",
        )

        # individual points
        beh_vals = corr_df[
            (corr_df["roi"] == roi_name) & (corr_df["behavior"] == "behavior")
        ]["correlation"]
        glob_vals = corr_df[
            (corr_df["roi"] == roi_name) & (corr_df["behavior"] == "global")
        ]["correlation"]

        ax.scatter(
            np.repeat(i - bar_width / 2, len(beh_vals)),
            beh_vals,
            color=scatter_color,
            alpha=scatter_alpha,
            s=10,
            zorder=1,
        )
        ax.scatter(
            np.repeat(i + bar_width / 2, len(glob_vals)),
            glob_vals,
            color=scatter_color,
            alpha=scatter_alpha,
            s=10,
            zorder=1,
        )

        # paired significance bracket
        if not test_row.empty and test_row["p-value"].values[0] < alpha_bonferroni_pairwise:
            ax.plot(
                [i - bar_width / 2, i + bar_width / 2],
                [pairwise_test_pos, pairwise_test_pos],
                color="black",
                lw=1,
            )
            ax.plot(
                [i - bar_width / 2, i - bar_width / 2],
                [pairwise_test_pos - 0.01, pairwise_test_pos],
                color="black",
                lw=1,
            )
            ax.plot(
                [i + bar_width / 2, i + bar_width / 2],
                [pairwise_test_pos - 0.01, pairwise_test_pos],
                color="black",
                lw=1,
            )
            ax.text(
                i,
                pairwise_test_pos,
                "*",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

        # per-bar significance vs 0
        if beh_row["p-value"].values[0] < alpha_bonferroni:
            ax.text(
                i - bar_width / 2,
                pairwise_test_pos - 0.05,
                "*",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )
        if glob_row["p-value"].values[0] < alpha_bonferroni:
            ax.text(
                i + bar_width / 2,
                pairwise_test_pos - 0.05,
                "*",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

        # noise ceilings
        lower_nc = overview_df[overview_df["roi"] == roi_name]["lower_nc"].values[0]
        upper_nc = overview_df[overview_df["roi"] == roi_name]["upper_nc"].values[0]

        ax.fill_between(
            [i - bar_width, i + bar_width],
            lower_nc,
            upper_nc,
            color="gray",
            alpha=0.2,
        )
        ax.hlines(
            y=lower_nc,
            xmin=i - bar_width,
            xmax=i + bar_width,
            colors="grey",
            linestyles="dashed",
            lw=1.2,
        )
        ax.hlines(
            y=upper_nc,
            xmin=i - bar_width,
            xmax=i + bar_width,
            colors="grey",
            linestyles="dashed",
            lw=1.2,
        )

    # cosmetics
    ax.axhline(0, color="black", linestyle="-", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(ROI_LABELS, fontsize=fontsize)
    ax.set_ylabel("Partial Spearman's Rho Correlation", fontsize=fontsize)
    ax.set_ylim(-0.2, 0.5)

    # legend noise ceilings (dummy handles)
    ax.hlines([], [], [], colors="grey", linestyles="dashed", lw=1.2, label="Upper Noise Ceiling")
    ax.hlines([], [], [], colors="grey", linestyles="dashed", lw=1.2, label="Lower Noise Ceiling")

    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=fontsize - 2)

    for spine in ["top", "bottom", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()

    # handle save name: if no extension, add .png
    if os.path.splitext(save_name)[1] == "":
        save_name = save_name + ".png"
    out_path = os.path.join(FIG_DIR, save_name)

    plt.savefig(out_path, dpi=300, transparent=True)
    plt.close(fig)

    print(f"Saved figure to: {out_path}")


# ---------------------------------------------------------------------
# OTHER SPACES: simple correlations (no partial)
# ---------------------------------------------------------------------
def load_other_spaces():
    """Load all external model RDMs and return rdms, rdvs, and names."""
    action_90_online_rdm = np.load(
        rdm_collection_path("Online_Behavior", "VISACT_fMRI", "action_fMRI_90_rdm.npy")
    )
    material_90_online_rdm = np.load(
        rdm_collection_path("Online_Behavior", "VISACT_fMRI", "material_fMRI_90_rdm.npy")
    )
    objects_90_online_rdm = np.load(
        rdm_collection_path("Online_Behavior", "VISACT_fMRI", "objects_fMRI_90_rdm.npy")
    )
    categories_90_online_rdm = np.load(
        rdm_collection_path("Online_Behavior", "VISACT_fMRI", "categories_fMRI_90_rdm.npy")
    )
    attributes_90_online_rdm = np.load(
        rdm_collection_path("Online_Behavior", "VISACT_fMRI", "attributes_fMRI_90_rdm.npy")
    )

    ade20k_90_online_rdm = np.load(
        rdm_collection_path("ADE20k", "ade20k_90_rdm.npy")
    )
    places365_90_online_rdm = np.load(
        rdm_collection_path("Places365", "places_365_90_rdm.npy")
    )

    sundb_90_nav_func = np.load(
        rdm_collection_path("SUN_DB", "VISACT_fMRI", "SUN_nav_func_90_rdm.npy")
    )
    sundb_90_materials = np.load(
        rdm_collection_path("SUN_DB", "VISACT_fMRI", "SUN_materials_90_rdm.npy")
    )
    sundb_90_spatial_env = np.load(
        rdm_collection_path("SUN_DB", "VISACT_fMRI", "SUN_spatialEnv_90_rdm.npy")
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

    rdms = [
        action_90_online_rdm,
        material_90_online_rdm,
        objects_90_online_rdm,
        categories_90_online_rdm,
        attributes_90_online_rdm,
        ade20k_90_online_rdm,
        places365_90_online_rdm,
        sundb_90_nav_func,
        sundb_90_materials,
        sundb_90_spatial_env,
    ]

    rdvs = [squareform(rdm.round(5)) for rdm in rdms]
    return rdms, rdvs, names


def compute_and_plot_other_space_correlations(save_name: str = "fMRI_correlations_other_spaces") -> None:
    """
    Simple Spearman correlations between fMRI RDMs and a collection of
    external RDMs (other spaces), for each ROI.
    """
    _, other_rdvs, other_names = load_other_spaces()
    n_models = len(other_names)
    n_rois = len(ROI_NAMES)

    # -------------------------------------------------------------
    # 1) Compute correlations for each subject, ROI, and model
    # -------------------------------------------------------------
    rows = []

    for roi_name in ROI_NAMES:
        _, brain_rdvs, lower_nc, upper_nc = load_brain_rdms(roi_name)

        for subj, brain_rdv in enumerate(brain_rdvs):
            sub_name = f"sub_{subj + 1:02d}"

            for model_name, model_rdv in zip(other_names, other_rdvs):
                rho, pval = spearmanr(brain_rdv, model_rdv)

                rows.append(
                    {
                        "behavior": model_name,
                        "roi": roi_name,
                        "subject": sub_name,
                        "correlation": rho,
                        "p-value": pval,
                        "lower_nc": lower_nc,
                        "higher_nc": upper_nc,
                    }
                )

    corr_df = pd.DataFrame(rows)

    # -------------------------------------------------------------
    # 2) Overview: mean, SEM, test vs 0 for each model × ROI
    # -------------------------------------------------------------
    overview_rows = []

    for model_name in other_names:
        for roi_name in ROI_NAMES:
            subset = corr_df[
                (corr_df["behavior"] == model_name) & (corr_df["roi"] == roi_name)
            ]
            if subset.empty:
                continue

            mean_corr = subset["correlation"].mean()
            sem_corr = subset["correlation"].std(ddof=1) / np.sqrt(len(subset))
            t_stat, p_val = ttest_1samp(subset["correlation"], 0)

            lower_nc = subset["lower_nc"].iloc[0]
            upper_nc = subset["higher_nc"].iloc[0]

            print(
                f"{model_name} {roi_name}: t={t_stat:.2f}, p={p_val:.4f}"
            )

            overview_rows.append(
                {
                    "behavior": model_name,
                    "roi": roi_name,
                    "subject": "mean",
                    "correlation": mean_corr,
                    "sem": sem_corr,
                    "p-value": p_val,
                    "lower_nc": lower_nc,
                    "upper_nc": upper_nc,
                }
            )

    overview_df = pd.DataFrame(overview_rows)

    # -------------------------------------------------------------
    # 3) Plot: bars per model within each ROI cluster
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    alpha = 0.05
    # number of tests: n_models * n_rois
    alpha_bonferroni = alpha / (n_models * n_rois)
    fontsize = 15

    x_centers = np.arange(n_rois)  # ROI centers: 0, 1, 2
    cluster_width = 0.9
    bar_width = cluster_width / n_models

    pairwise_test_pos = overview_df["upper_nc"].max() + 0.06
    scatter_color = "gray"
    scatter_alpha = 0.5
    star_y = pairwise_test_pos - 0.02

    # some simple color palette for the 10 models
    base_palette = [
        "#A8DADC",
        "#A8DADC",
        "#A8DADC",
        "#A8DADC",
        "#A8DADC",
        "#1D3557",
        "#E3D5CA",
        "#003049",
        "#003049",
        "#003049",
    ]
    bar_colors = base_palette[:n_models]

    all_centers = []
    all_labels = []

    for i, roi_name in enumerate(ROI_NAMES):
        roi_center = x_centers[i]

        roi_subset = overview_df[overview_df["roi"] == roi_name]
        lower_nc = roi_subset["lower_nc"].iloc[0]
        upper_nc = roi_subset["upper_nc"].iloc[0]

        # noise ceiling patch spanning entire cluster
        left = roi_center - cluster_width / 2 - 0.02
        right = roi_center + cluster_width / 2 + 0.02
        ax.fill_between([left, right], lower_nc, upper_nc, color="gray", alpha=0.2)
        ax.hlines(lower_nc, left, right, colors="grey", linestyles="dashed", lw=1.2)
        ax.hlines(upper_nc, left, right, colors="grey", linestyles="dashed", lw=1.2)

        # bars for each model
        for j, model_name in enumerate(other_names):
            offset_index = j - (n_models - 1) / 2.0
            bar_center = roi_center + offset_index * bar_width

            row = roi_subset[roi_subset["behavior"] == model_name]
            if row.empty:
                continue

            mean_corr = row["correlation"].values[0]
            sem_corr = row["sem"].values[0]
            p_val = row["p-value"].values[0]

            # bar
            ax.bar(
                bar_center,
                mean_corr,
                bar_width,
                color=bar_colors[j],
                edgecolor="white",
            )

            # individual points
            indiv = corr_df[
                (corr_df["roi"] == roi_name)
                & (corr_df["behavior"] == model_name)
            ]["correlation"].values

            ax.scatter(
                np.full_like(indiv, bar_center, dtype=float),
                indiv,
                color=scatter_color,
                alpha=scatter_alpha,
                s=10,
                zorder=1,
            )

            # significance star (vs 0)
            if p_val < alpha_bonferroni:
                ax.text(
                    bar_center,
                    star_y,
                    "*",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

            all_centers.append(bar_center)
            all_labels.append(model_name)

    # cosmetics
    ax.axhline(0, color="black", linestyle="-", lw=1.2)
    ax.set_xticks(all_centers)
    ax.set_xticklabels(all_labels, rotation=90, fontsize=10)

    ax.set_ylabel("Spearman's Rho Correlation", fontsize=fontsize)
    ax.set_ylim(-0.1, max(0.3, overview_df["upper_nc"].max() + 0.1))

    # noise ceiling legend dummy
    ax.hlines([], [], [], colors="grey", linestyles="dashed", lw=1.2, label="Noise ceiling")

    for spine in ["top", "bottom", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()

    if os.path.splitext(save_name)[1] == "":
        save_name = save_name + ".png"
    out_path = os.path.join(FIG_DIR, save_name)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved other-spaces figure to: {out_path}")


# ---------------------------------------------------------------------
# EXAMPLE CALLS
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Example 1: global attributes space
    compute_and_plot_partial_correlations(
        main_space="action",
        space_path=rdm_collection_path(
            "Online_Behavior",
            "VISACT_fMRI",
            "attributes_fMRI_90_rdm.npy",
        ),
        space_name="Global properties",
        plot_color="#dda15e",
        save_name="partial_corr_action_vs_globalprops",
        num_comparisons=18,
    )

    # Example 2: online object space
    compute_and_plot_partial_correlations(
        main_space="action",
        space_path=rdm_collection_path(
            "Online_Behavior",
            "VISACT_fMRI",
            "objects_fMRI_90_rdm.npy",
        ),
        space_name="Objects (online exp.)",
        plot_color="#6FA1BB",
        save_name="partial_corr_action_vs_online_objects",
        num_comparisons=18,
    )

    # Example 3: fMRI correlations with all other spaces
    compute_and_plot_other_space_correlations(
        save_name="fMRI_correlations_other_spaces"
    )
