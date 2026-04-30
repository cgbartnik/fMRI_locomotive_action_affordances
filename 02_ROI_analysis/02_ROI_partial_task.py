from __future__ import annotations

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import squareform
from scipy.stats import ttest_1samp
import scipy.stats as stats
import pingouin as pg

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_paths import behavior_path, brain_path


# ---------------------------------------------------------------------
# PATHS & CONSTANTS
# ---------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "Figures")
os.makedirs(FIG_DIR, exist_ok=True)

# in-scanner behavior RDMs
BEHAVIOR_ACTION_PATH = behavior_path(
    "VISACT_fmri_behavior", "fmri_behavior_action_rdms.npz"
)
BEHAVIOR_OBJECT_PATH = behavior_path(
    "VISACT_fmri_behavior", "fmri_behavior_object_rdms.npz"
)

# task-specific brain RDMs live under {task}_task/fmri_{ROI}_{task}.npz
BRAIN_TASK_BASE = brain_path()

ROI_TASK_NAMES = ["PPA", "OPA", "RSC"]
TASKS = ["action", "object", "fixation"]
BEHAVIORS = ["action", "object"]
TASK_COMPARISONS = [("action", "object"), ("action", "fixation"), ("object", "fixation")]


# ---------------------------------------------------------------------
# CORE UTILITIES
# ---------------------------------------------------------------------
def compute_noise_ceilings(subject_rdms: np.ndarray) -> tuple[float, float]:
    """
    Compute lower and upper noise ceilings from subject RDMs.

    subject_rdms : (n_subj, n_cond, n_cond)
    """
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


def load_behavior_rdms() -> tuple[np.ndarray, np.ndarray]:
    """Load action and object behavioral RDMs (subject × 90 × 90)."""
    behavior_action_rdms = np.load(BEHAVIOR_ACTION_PATH)["arr_0"]
    behavior_object_rdms = np.load(BEHAVIOR_OBJECT_PATH)["arr_0"]
    return behavior_action_rdms, behavior_object_rdms


# ---------------------------------------------------------------------
# TASK-SPECIFIC PARTIAL CORRELATIONS
# ---------------------------------------------------------------------
def run_task_specific_partial_correlations() -> pd.DataFrame:
    """
    Compute task-specific partial correlations:

        brain ~ action | object
        brain ~ object | action

    for each ROI (PPA/OPA/RSC), each task session (action/object/fixation),
    and each subject.

    Returns
    -------
    corr_df : DataFrame with columns:
        ['behavior', 'task', 'roi', 'subject',
         'correlation', 'p-value', 'lower_nc', 'higher_nc']
    """
    behavior_action_rdms, behavior_object_rdms = load_behavior_rdms()

    # subject-mean RDVs of the two behavioral spaces
    mean_action_rdv = squareform(behavior_action_rdms.mean(axis=0).round(5))
    mean_object_rdv = squareform(behavior_object_rdms.mean(axis=0).round(5))

    rows = []

    for roi in ROI_TASK_NAMES:
        for task in TASKS:
            path = os.path.join(
                BRAIN_TASK_BASE,
                f"{task}_task",
                f"fmri_{roi}_{task}.npz",
            )
            Mean_RDMs = np.load(path)["arr_0"]  # (subjects, 90, 90)
            brain_rdvs = [
                squareform(Mean_RDMs[subj].round(5))
                for subj in range(Mean_RDMs.shape[0])
            ]
            lower_nc, upper_nc = compute_noise_ceilings(Mean_RDMs)

            for subj, rdv in enumerate(brain_rdvs):
                sub_name = f"sub_{subj + 1:02d}"

                df = pd.DataFrame(
                    {
                        "brain": rdv,
                        "action": mean_action_rdv,
                        "object": mean_object_rdv,
                    }
                )

                # brain ~ action | object
                pc_action = pg.partial_corr(
                    data=df,
                    x="brain",
                    y="action",
                    covar="object",
                    method="spearman",
                ).round(3)

                rows.append(
                    {
                        "behavior": "action",
                        "task": task,
                        "roi": roi,
                        "subject": sub_name,
                        "correlation": pc_action["r"].values[0],
                        "p-value": pc_action["p_val"].values[0],
                        "lower_nc": lower_nc,
                        "higher_nc": upper_nc,
                    }
                )

                # brain ~ object | action
                pc_object = pg.partial_corr(
                    data=df,
                    x="brain",
                    y="object",
                    covar="action",
                    method="spearman",
                ).round(3)

                rows.append(
                    {
                        "behavior": "object",
                        "task": task,
                        "roi": roi,
                        "subject": sub_name,
                        "correlation": pc_object["r"].values[0],
                        "p-value": pc_object["p_val"].values[0],
                        "lower_nc": lower_nc,
                        "higher_nc": upper_nc,
                    }
                )

    corr_df = pd.DataFrame(rows)
    return corr_df


def summarize_task_partial_correlations(
    corr_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarize task-specific partial correlations.

    Returns
    -------
    overview_df : one row per (behavior, task, roi) with mean, sem, p vs 0, noise ceilings
    test_result_df : paired t-tests between tasks within each behavior & ROI
                     columns: ['behavior', 'roi', 'comparison', 't-statistic', 'p-value']
    """
    overview_rows = []
    test_rows = []

    total_tests = len(BEHAVIORS) * len(TASKS) * len(ROI_TASK_NAMES)
    print(f"Total Bonferroni tests (in-space): {total_tests}")

    # Means, SEMs, and tests vs 0
    for behavior in BEHAVIORS:
        for task in TASKS:
            for roi in ROI_TASK_NAMES:
                subset = corr_df[
                    (corr_df["behavior"] == behavior)
                    & (corr_df["task"] == task)
                    & (corr_df["roi"] == roi)
                ]
                if subset.empty:
                    continue

                mean_corr = subset["correlation"].mean()
                sem_corr = subset["correlation"].std(ddof=1) / np.sqrt(len(subset))
                t_stat, p_val = ttest_1samp(subset["correlation"], 0)

                lower_nc = subset["lower_nc"].iloc[0]
                upper_nc = subset["higher_nc"].iloc[0]

                print(
                    f"{behavior} {task} {roi}: "
                    f"mean={mean_corr:.3f}, t={t_stat:.2f}, p={p_val:.3f}"
                )

                overview_rows.append(
                    {
                        "behavior": behavior,
                        "task": task,
                        "roi": roi,
                        "subject": "mean",
                        "correlation": mean_corr,
                        "sem": sem_corr,
                        "p-value": p_val,
                        "lower_nc": lower_nc,
                        "upper_nc": upper_nc,
                    }
                )

    # Paired t-tests between tasks within each behavior & ROI
    for behavior in BEHAVIORS:
        for roi in ROI_TASK_NAMES:
            for t1, t2 in TASK_COMPARISONS:
                vals1 = corr_df[
                    (corr_df["behavior"] == behavior)
                    & (corr_df["roi"] == roi)
                    & (corr_df["task"] == t1)
                ]["correlation"]
                vals2 = corr_df[
                    (corr_df["behavior"] == behavior)
                    & (corr_df["roi"] == roi)
                    & (corr_df["task"] == t2)
                ]["correlation"]

                if len(vals1) == len(vals2) and len(vals1) > 0:
                    t_stat, p_val = stats.ttest_rel(vals1, vals2)
                    test_rows.append(
                        {
                            "behavior": behavior,
                            "roi": roi,
                            "comparison": f"{t1}_vs_{t2}",
                            "t-statistic": t_stat,
                            "p-value": p_val,
                        }
                    )

    overview_df = pd.DataFrame(overview_rows)
    test_result_df = pd.DataFrame(test_rows)
    return overview_df, test_result_df


# ---------------------------------------------------------------------
# PLOTTING: TASK-EFFECT BARS
# ---------------------------------------------------------------------
def plot_task_effect_bars(
    overview_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    test_result_df: pd.DataFrame,
    behavior_name: str,
    filename: str,
) -> None:
    """
    Task-effect bar plot for a given behavior ('action' or 'object'):

        - three bars per ROI (action/object/fixation task sessions)
        - individual subject dots
        - noise ceiling band
        - stars for:
            * in-space tests vs 0 (per bar)
            * between-task comparisons (cluster-level)
    """
    roi_order = ROI_TASK_NAMES
    task_order = TASKS
    num_rois = len(roi_order)
    num_tasks = len(task_order)
    bar_width = 0.3
    fontsize = 15

    alpha = 0.05
    total_tests = len(BEHAVIORS) * len(TASKS) * len(ROI_TASK_NAMES)
    alpha_bonferroni_in_space = alpha / total_tests
    alpha_bonferroni_between_space = alpha / total_tests
    star_y = 0.32  # fixed height for in-space stars

    # colors for the three tasks, depending on behavior
    if behavior_name == "action":
        bar_colors = ["#9e483d", "#E36858", "#eb958a"]
    else:  # 'object'
        bar_colors = ["#426070", "#6FA1BB", "#a8c6d6"]

    df = overview_df[overview_df["behavior"] == behavior_name].copy()
    df["roi"] = pd.Categorical(df["roi"], roi_order, ordered=True)
    df["task"] = pd.Categorical(df["task"], task_order, ordered=True)
    df.sort_values(["roi", "task"], inplace=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    x_centers = np.arange(num_rois)  # ROI centers

    # bars, dots, per-task stars, noise ceilings
    for i, roi in enumerate(roi_order):
        roi_center = x_centers[i]
        roi_nc = df[df["roi"] == roi].iloc[0]
        lower_nc = roi_nc["lower_nc"]
        upper_nc = roi_nc["upper_nc"]

        # noise ceiling band over the whole cluster
        nc_left = roi_center - (num_tasks * bar_width) / 2 - 0.05
        nc_right = roi_center + (num_tasks * bar_width) / 2 + 0.05
        ax.fill_between(
            [nc_left, nc_right],
            lower_nc,
            upper_nc,
            color="gray",
            alpha=0.2,
        )
        ax.hlines(
            y=lower_nc,
            xmin=nc_left,
            xmax=nc_right,
            colors="grey",
            linestyles="dashed",
            lw=1.2,
        )
        ax.hlines(
            y=upper_nc,
            xmin=nc_left,
            xmax=nc_right,
            colors="grey",
            linestyles="dashed",
            lw=1.2,
        )

        # bars & individual dots for each task
        for j, task in enumerate(task_order):
            row = df[(df["roi"] == roi) & (df["task"] == task)]
            if row.empty:
                continue

            offset_index = j - (num_tasks - 1) / 2.0
            bar_center = roi_center + offset_index * bar_width

            mean_corr = row["correlation"].values[0]
            sem_corr = row["sem"].values[0]
            p_val = row["p-value"].values[0]

            ax.bar(
                bar_center,
                mean_corr,
                bar_width,
                yerr=sem_corr,
                color=bar_colors[j],
                edgecolor="white",
                capsize=5,
                label=(task + " task session") if (i == 0) else "",
            )

            indiv = corr_df[
                (corr_df["behavior"] == behavior_name)
                & (corr_df["roi"] == roi)
                & (corr_df["task"] == task)
            ]["correlation"].values

            ax.scatter(
                np.full_like(indiv, bar_center, dtype=float),
                indiv,
                color="gray",
                alpha=0.5,
                s=10,
                zorder=1,
            )

            # in-space significance vs 0
            if p_val < alpha_bonferroni_in_space:
                ax.text(
                    bar_center,
                    star_y,
                    "*",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    fontweight="bold",
                )

    # between-task tests: a single star per ROI if any comparison is significant
    for i, roi in enumerate(roi_order):
        roi_tests = test_result_df[
            (test_result_df["roi"] == roi)
            & (test_result_df["behavior"] == behavior_name)
        ]
        if (roi_tests["p-value"] < alpha_bonferroni_between_space).any():
            roi_center = x_centers[i]
            ax.text(
                roi_center,
                star_y + 0.04,
                "*",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

    # cosmetics
    ax.axhline(0, color="black", linestyle="-", linewidth=1)

    ax.set_xlim(x_centers[0] - 0.7, x_centers[-1] + 0.7)
    ax.set_xticks(x_centers)
    ax.set_xticklabels(roi_order, fontsize=fontsize)
    ax.set_ylabel("Partial Spearman's Rho Correlation", fontsize=fontsize)
    ax.set_ylim(-0.17, 0.45)

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=fontsize - 2)

    plt.tight_layout()

    if os.path.splitext(filename)[1] == "":
        filename = filename + ".png"
    out_path = os.path.join(FIG_DIR, filename)
    plt.savefig(out_path, dpi=300, transparent=True)
    plt.close(fig)

    print(f"Saved task-effect plot for {behavior_name} to: {out_path}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main() -> None:
    print("\n=== Task-specific partial correlations (action/object/fixation) ===")
    corr_df = run_task_specific_partial_correlations()
    overview_df, tests_df = summarize_task_partial_correlations(corr_df)

    print("\nPaired task comparisons:")
    print(tests_df.round(3))

    # Plot for behavior "action"
    plot_task_effect_bars(
        overview_df=overview_df,
        corr_df=corr_df,
        test_result_df=tests_df,
        behavior_name="action",
        filename="fMRI_partial_correlations_action_taskeffect.png",
    )

    # Plot for behavior "object"
    plot_task_effect_bars(
        overview_df=overview_df,
        corr_df=corr_df,
        test_result_df=tests_df,
        behavior_name="object",
        filename="fMRI_partial_correlations_object_taskeffect.png",
    )


if __name__ == "__main__":
    main()
