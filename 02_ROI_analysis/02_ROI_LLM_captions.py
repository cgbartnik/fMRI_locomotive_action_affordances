from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, ttest_rel, ttest_1samp
import pingouin as pg
from data_paths import behavior_path, brain_path, embedding_rdms_path

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

ROI_NAMES = ["PPA_mean", "OPA_mean", "RSC_mean"]
ROI_LABELS = ["PPA", "OPA", "RSC"]

# ---------------------------------------------------------------------
# CORE UTILITIES
# ---------------------------------------------------------------------
def compute_noise_ceilings(subject_rdms: np.ndarray) -> tuple[float, float]:
    """
    Compute lower and upper noise ceilings from subject RDMs.

    subject_rdms : (n_subj, n_cond, n_cond)
    """
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


def load_squareform_rdm(path: str) -> np.ndarray:
    """Load a 90x90 RDM (npy or npz['arr_0']) and return its 1D squareform."""
    arr = np.load(path)
    if isinstance(arr, np.lib.npyio.NpzFile):
        arr = arr["arr_0"]
    return squareform(arr.round(5))


# ---------------------------------------------------------------------
# PARTIAL CORRELATIONS: main behavior vs one external space
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
        - another global space (e.g. attributes, objects, captions)

    brain ~ behavior | global
    brain ~ global | behavior

    Saves a figure into the local Figures/ folder.
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

    global_rdv = load_squareform_rdm(space_path)

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
                    "global": global_rdv,
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
    print(f"\nPaired t-tests {main_space} vs {space_name}:")
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

    x = np.arange(len(ROI_NAMES))
    bar_width = 0.4

    pairwise_test_pos = overview_df["upper_nc"].max() + 0.06
    scatter_color = "gray"
    scatter_alpha = 0.5

    for i, roi_name in enumerate(ROI_NAMES):
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

    if os.path.splitext(save_name)[1] == "":
        save_name = save_name + ".png"
    out_path = os.path.join(FIG_DIR, save_name)
    plt.savefig(out_path, dpi=300, transparent=True)
    plt.close(fig)

    print(f"Saved partial-correlation figure to: {out_path}")


# ---------------------------------------------------------------------
# DIRECT CORRELATIONS: main behavior vs multiple spaces (GPT / captions)
# ---------------------------------------------------------------------
def compute_and_plot_correlations(
    main_space: str,
    spaces: list[tuple[str, str, str]],
    save_name: str,
    num_comparisons: int,
) -> pd.DataFrame:
    """
    Direct correlations between fMRI and:
        - main behavioral space (action or object; subject-mean RDM)
        - several additional spaces

    Parameters
    ----------
    main_space : 'action' or 'object'
    spaces : list of (label, path, color)
        label : name used in 'behavior' column and legend
        path  : npy/npz RDM path
        color : bar color for this space
    save_name : filename (without or with extension) under Figures/
    num_comparisons : for Bonferroni

    Returns
    -------
    corr_df : long DataFrame with all subject-level correlations
    """
    # Load main behavioral RDMs and mean rdv
    behavior_rdms = load_behavior_rdms(main_space)
    mean_behavior_rdv = squareform(behavior_rdms.mean(axis=0).round(5))

    # Load external spaces as RDVs
    space_rdvs = {label: load_squareform_rdm(path) for (label, path, _color) in spaces}

    # -----------------------------------------------------------------
    # 1) Correlations
    # -----------------------------------------------------------------
    rows = []
    for roi_name in ROI_NAMES:
        Mean_RDMs, brain_rdvs, lower_nc, upper_nc = load_brain_rdms(roi_name)

        for subj, rdv in enumerate(brain_rdvs):
            sub_name = f"sub_{subj + 1:02d}"

            # main behavior
            r_beh = spearmanr(rdv, mean_behavior_rdv)
            rows.append(
                {
                    "behavior": main_space,
                    "roi": roi_name,
                    "subject": sub_name,
                    "correlation": r_beh.correlation,
                    "p-value": r_beh.pvalue,
                    "lower_nc": lower_nc,
                    "higher_nc": upper_nc,
                }
            )

            # additional spaces
            for label, space_rdv in space_rdvs.items():
                r_space = spearmanr(rdv, space_rdv)
                rows.append(
                    {
                        "behavior": label,
                        "roi": roi_name,
                        "subject": sub_name,
                        "correlation": r_space.correlation,
                        "p-value": r_space.pvalue,
                        "lower_nc": lower_nc,
                        "higher_nc": upper_nc,
                    }
                )

    corr_df = pd.DataFrame(rows)

    # -----------------------------------------------------------------
    # 2) Paired t-tests: each space vs main_space per ROI
    # -----------------------------------------------------------------
    test_rows = []
    other_behaviors = [b for b, _, _ in spaces]
    for roi_name in ROI_NAMES:
        ref_vals = corr_df[
            (corr_df["roi"] == roi_name) & (corr_df["behavior"] == main_space)
        ]["correlation"]

        for behavior in other_behaviors:
            beh_vals = corr_df[
                (corr_df["roi"] == roi_name) & (corr_df["behavior"] == behavior)
            ]["correlation"]

            t_stat, p_val = ttest_rel(ref_vals, beh_vals)
            test_rows.append(
                {
                    "roi": roi_name,
                    "behavior": behavior,
                    "t-statistic": t_stat,
                    "p-value": p_val,
                }
            )

    test_result_df = pd.DataFrame(test_rows)
    print(f"\nPaired t-tests {main_space} vs other spaces:")
    print(test_result_df.round(3))

    # -----------------------------------------------------------------
    # 3) Overview: mean, SEM, tests vs 0
    # -----------------------------------------------------------------
    overview_rows = []
    all_behaviors = [main_space] + other_behaviors
    for behavior in all_behaviors:
        for roi_name in ROI_NAMES:
            subset = corr_df[
                (corr_df["behavior"] == behavior) & (corr_df["roi"] == roi_name)
            ]
            mean_corr = subset["correlation"].mean()
            sem_corr = subset["correlation"].std(ddof=1) / np.sqrt(len(subset))
            _, p_val = ttest_1samp(subset["correlation"], 0)
            lower_nc = subset["lower_nc"].iloc[0]
            upper_nc = subset["higher_nc"].iloc[0]

            overview_rows.append(
                {
                    "behavior": behavior,
                    "roi": roi_name,
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
    alpha = 0.05
    alpha_bonferroni = alpha / num_comparisons
    fontsize = 15

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(ROI_NAMES))
    bar_width = 0.20

    colors = ["#E36858" if main_space == "action" else "#6FA1BB"] + [
        c for (_lab, _path, c) in spaces
    ]
    labels = [main_space] + [lab for (lab, _path, _c) in spaces]

    pairwise_test_pos_base = overview_df["upper_nc"].max() + 0.07

    for i, behavior in enumerate(labels):
        for j, roi_name in enumerate(ROI_NAMES):
            row = overview_df[
                (overview_df["roi"] == roi_name)
                & (overview_df["behavior"] == behavior)
            ]

            center = x[j] + (i - (len(labels) - 1) / 2) * bar_width

            ax.bar(
                center,
                row["correlation"],
                bar_width,
                yerr=row["sem"],
                color=colors[i],
                label=(
                    "Loc. affordances" if (i == 0 and main_space == "action") else
                    "Objects" if (i == 0 and main_space == "object") else
                    behavior if j == 0 else ""
                ),
            )

            points = corr_df[
                (corr_df["roi"] == roi_name) & (corr_df["behavior"] == behavior)
            ]["correlation"]
            ax.scatter(
                np.repeat(center, len(points)),
                points,
                color="gray",
                alpha=0.5,
                s=10,
                zorder=1,
            )

            if row["p-value"].values[0] < alpha_bonferroni:
                ax.text(
                    center,
                    pairwise_test_pos_base - 0.04,
                    "*",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    fontweight="bold",
                )

            # paired brackets vs main space
            if behavior != main_space:
                test_row = test_result_df[
                    (test_result_df["roi"] == roi_name)
                    & (test_result_df["behavior"] == behavior)
                ]
                if not test_row.empty and test_row["p-value"].values[0] < alpha_bonferroni:
                    pairwise_y = pairwise_test_pos_base
                    ref_center = x[j] + (0 - (len(labels) - 1) / 2) * bar_width
                    ax.plot(
                        [ref_center, center],
                        [pairwise_y, pairwise_y],
                        color="black",
                        lw=1,
                    )
                    ax.plot(
                        [ref_center, ref_center],
                        [pairwise_y - 0.01, pairwise_y],
                        color="black",
                        lw=1,
                    )
                    ax.plot(
                        [center, center],
                        [pairwise_y - 0.01, pairwise_y],
                        color="black",
                        lw=1,
                    )
                    ax.text(
                        (ref_center + center) / 2,
                        pairwise_y - 0.0075,
                        "*",
                        ha="center",
                        va="bottom",
                        fontsize=12,
                        fontweight="bold",
                    )

    # Noise ceilings
    for i, roi_name in enumerate(ROI_NAMES):
        lower_nc = overview_df[overview_df["roi"] == roi_name]["lower_nc"].values[0]
        upper_nc = overview_df[overview_df["roi"] == roi_name]["upper_nc"].values[0]

        ax.fill_between(
            [i - 2 * bar_width, i + 2 * bar_width],
            lower_nc,
            upper_nc,
            color="gray",
            alpha=0.2,
        )
        ax.hlines(
            y=lower_nc,
            xmin=i - 2 * bar_width,
            xmax=i + 2 * bar_width,
            colors="grey",
            linestyles="dashed",
            lw=1.2,
        )
        ax.hlines(
            y=upper_nc,
            xmin=i - 2 * bar_width,
            xmax=i + 2 * bar_width,
            colors="grey",
            linestyles="dashed",
            lw=1.2,
        )

    ax.axhline(0, color="black", linestyle="-", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(ROI_LABELS, fontsize=fontsize)
    ax.set_ylabel("Spearman's rho correlation", fontsize=fontsize)
    ax.set_ylim(-0.2, 0.5)

    # noise-ceiling legend entries
    ax.hlines([], [], [], colors="grey", linestyles="dashed", lw=1.2, label="Noise ceiling")

    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=fontsize - 2)

    for spine in ["top", "bottom", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    if os.path.splitext(save_name)[1] == "":
        save_name = save_name + ".png"
    out_path = os.path.join(FIG_DIR, save_name)
    plt.savefig(out_path, dpi=300, transparent=True)
    plt.close(fig)

    print(f"Saved correlation figure to: {out_path}")
    return corr_df


# ---------------------------------------------------------------------
# MAIN: replicate your previous calls, but using streamlined functions
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # --- GPT / MiniLM caption spaces: direct correlations (multi-space figure) ---
    spaces_minilm = [
        (
            "caption_prompt",
            embedding_rdms_path(
                "minilm", "minilm_caption_prompt_fMRI_sorted_correlation_rdm.npy"
            ),
            "#588157",
        ),
        (
            "affordance_prompt",
            embedding_rdms_path(
                "minilm", "minilm_affordance_prompt_fMRI_sorted_correlation_rdm.npy"
            ),
            "#a3b18a",
        ),
        (
            "object_prompt",
            embedding_rdms_path(
                "minilm", "minilm_object_prompt_fMRI_sorted_correlation_rdm.npy"
            ),
            "#3a5a40",
        ),
    ]

    corr_gpt = compute_and_plot_correlations(
        main_space="action",
        spaces=spaces_minilm,
        save_name="fmri_GPT_embeddings.png",
        num_comparisons=12,
    )

    # --- partial correlations for each caption space separately ---
    compute_and_plot_partial_correlations(
        main_space="action",
        space_path=spaces_minilm[0][1],
        space_name="General Caption",
        plot_color=spaces_minilm[0][2],
        save_name="partial_fmri_GPT_general_caption.png",
        num_comparisons=27,
    )

    compute_and_plot_partial_correlations(
        main_space="action",
        space_path=spaces_minilm[1][1],
        space_name="Affordance Caption",
        plot_color=spaces_minilm[1][2],
        save_name="partial_fmri_GPT_affordance_caption.png",
        num_comparisons=27,
    )

    compute_and_plot_partial_correlations(
        main_space="action",
        space_path=spaces_minilm[2][1],
        space_name="Object Caption",
        plot_color=spaces_minilm[2][2],
        save_name="partial_fmri_GPT_object_caption.png",
        num_comparisons=27,
    )
