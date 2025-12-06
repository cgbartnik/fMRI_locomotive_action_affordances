from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, ttest_1samp
from scipy.spatial.distance import squareform
import scipy.stats as stats
import pingouin as pg
from data_paths import behavior_path, brain_path

# ---------------------------------------------------------------------
# PATHS & FIGURE DIRECTORY
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

# Average (task-agnostic) fMRI RDMs
BRAIN_BASE_PATH = brain_path("average")

# Task-specific fMRI RDMs
BRAIN_TASK_BASE = brain_path()

# ---------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------
# Task-agnostic ROI file names
ROI_NAMES = ["PPA_mean", "OPA_mean", "RSC_mean"]

# Task-specific ROI names (no "_mean")
ROI_TASK_NAMES = ["PPA", "OPA", "RSC"]

# Behaviors and tasks
BEHAVIORS = ["action", "object"]
TASKS = ["action", "object", "fixation"]

# For task comparisons (within-behavior)
TASK_COMPARISONS = [
    ("action", "object"),
    ("action", "fixation"),
    ("object", "fixation"),
]


# ---------------------------------------------------------------------
# CORE UTILITIES
# ---------------------------------------------------------------------
def compute_noise_ceilings(subject_rdms: np.ndarray) -> tuple[float, float]:
    """
    Compute lower and upper noise ceilings from subject RDMs.

    Parameters
    ----------
    subject_rdms : array, shape (n_subjects, n_cond, n_cond)

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
    group_rdv = squareform(group_rdm)

    # Correlation of each subject with group-average (upper bound)
    group_correlations = [
        spearmanr(group_rdv, squareform(rdm)).correlation for rdm in subject_rdms
    ]

    for i in range(num_subjects):
        # Leave-one-out group-average
        other_subjects_rdms = np.delete(subject_rdms, i, axis=0)
        other_group_rdm = np.mean(other_subjects_rdms, axis=0)
        other_group_rdv = squareform(other_group_rdm)

        lower_bound_correlations[i] = spearmanr(
            other_group_rdv, squareform(subject_rdms[i])
        ).correlation
        upper_bound_correlations[i] = group_correlations[i]

    return lower_bound_correlations.mean(), upper_bound_correlations.mean()


def load_behavior_rdms() -> tuple[np.ndarray, np.ndarray]:
    """Load action and object behavioral RDMs (subject x 90 x 90)."""
    behavior_action_rdms = np.load(BEHAVIOR_ACTION_PATH)["arr_0"]
    behavior_object_rdms = np.load(BEHAVIOR_OBJECT_PATH)["arr_0"]
    return behavior_action_rdms, behavior_object_rdms


def load_brain_rdms(roi_name: str) -> tuple[np.ndarray, list[np.ndarray], float, float]:
    """
    Load brain RDMs for a given ROI (task-agnostic average).

    Returns
    -------
    mean_rdms : (n_subjects, 90, 90)
    brain_rdvs : list of 1D rdvs
    lower_nc, upper_nc : noise ceilings
    """
    path = os.path.join(BRAIN_BASE_PATH, f"fmri_{roi_name}.npz")
    mean_rdms = np.load(path)["arr_0"]
    brain_rdvs = [
        squareform(mean_rdms[subj].round(5)) for subj in range(mean_rdms.shape[0])
    ]
    lower_nc, upper_nc = compute_noise_ceilings(mean_rdms)
    return mean_rdms, brain_rdvs, lower_nc, upper_nc


def roi_label(roi_name: str) -> str:
    """Prettier label: 'PPA_mean' → 'PPA', 'PPA' → 'PPA'."""
    return roi_name.split("_")[0]


# ---------------------------------------------------------------------
# FIGURE HELPERS
# ---------------------------------------------------------------------
def plot_all_subject_rdms(rdms: np.ndarray, title: str, filename: str) -> None:
    """Plot 20 subject RDMs in a 2x10 grid."""
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    axes = axes.ravel()

    for i in range(20):
        im = axes[i].imshow(rdms[i], cmap="mako")
        axes[i].set_title(f"Subject {i + 1}", fontsize=8)
        plt.colorbar(im, ax=axes[i])

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=300)
    plt.close(fig)


def plot_average_rdm(rdm: np.ndarray, title: str, filename: str) -> None:
    """Plot a single average RDM."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(rdm, cmap="mako")
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=300)
    plt.close(fig)


def summarize_correlations(corr_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given a correlation dataframe with columns:
        ['behavior', 'roi', 'subject', 'correlation', 'p-value', 'lower_nc', 'higher_nc']
    compute:
        - overview_df: mean, SEM, noise ceilings, p vs 0
        - test_result_df: paired t-test action vs object for each ROI
    """
    overview_rows = []
    test_rows = []

    # Respect fixed ROI order and only include those present
    present_rois = [r for r in ROI_NAMES if r in corr_df["roi"].unique()]

    # Paired t-tests: action vs object
    for roi in present_rois:
        action_vals = corr_df[
            (corr_df["roi"] == roi) & (corr_df["behavior"] == "action")
        ]["correlation"]
        object_vals = corr_df[
            (corr_df["roi"] == roi) & (corr_df["behavior"] == "object")
        ]["correlation"]

        t_stat, p_val = stats.ttest_rel(action_vals, object_vals)
        test_rows.append(
            {
                "roi": roi,
                "behavior": "action",
                "t-statistic": t_stat,
                "p-value": p_val,
            }
        )

    # Mean, SEM, t-test vs 0 for each behavior x ROI
    for behavior in BEHAVIORS:
        for roi in present_rois:
            subset = corr_df[
                (corr_df["roi"] == roi) & (corr_df["behavior"] == behavior)
            ]
            if subset.empty:
                continue

            mean_corr = np.mean(subset["correlation"])
            sem_corr = np.std(subset["correlation"], ddof=1) / np.sqrt(len(subset))
            t_stat, p_val = ttest_1samp(subset["correlation"], 0)

            lower_nc = subset["lower_nc"].iloc[0]
            upper_nc = subset["higher_nc"].iloc[0]

            print(
                f"{behavior} {roi} mean={mean_corr:.2f}, t={t_stat:.2f}, p={p_val:.3f}"
            )

            overview_rows.append(
                {
                    "behavior": behavior,
                    "roi": roi,
                    "subject": "mean",
                    "correlation": mean_corr,
                    "sem": sem_corr,
                    "p-value": p_val,
                    "lower_nc": lower_nc,
                    "upper_nc": upper_nc,
                }
            )

    overview_df = pd.DataFrame(overview_rows)
    test_result_df = pd.DataFrame(test_rows)
    return overview_df, test_result_df


def plot_correlation_bars(
    overview_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    test_result_df: pd.DataFrame,
    ylabel: str,
    filename: str,
    bar_width: float = 0.35,
) -> None:
    """
    Bar plot of mean correlations with error bars, individual data,
    noise ceilings, and significance markers (action vs object).
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    alpha = 0.05
    alpha_bonferroni = alpha / 9
    alpha_bonferroni_pairwise = alpha / 9
    fontsize = 15

    bar_colors = ["#E36858", "#6FA1BB"]  # action, object
    scatter_color = "gray"
    scatter_alpha = 0.5

    # Respect fixed ROI order and only include present ones
    present_rois = [r for r in ROI_NAMES if r in overview_df["roi"].unique()]
    x = np.arange(len(present_rois))

    pairwise_test_pos = overview_df["upper_nc"].max() + 0.06

    for i, roi in enumerate(present_rois):
        action_row = overview_df[
            (overview_df["roi"] == roi) & (overview_df["behavior"] == "action")
        ]
        object_row = overview_df[
            (overview_df["roi"] == roi) & (overview_df["behavior"] == "object")
        ]
        test_result = test_result_df[test_result_df["roi"] == roi]

        # Bars
        ax.bar(
            i - bar_width / 2,
            action_row["correlation"],
            bar_width,
            yerr=action_row["sem"],
            color=bar_colors[0],
            label="Action space" if i == 0 else "",
        )
        ax.bar(
            i + bar_width / 2,
            object_row["correlation"],
            bar_width,
            yerr=object_row["sem"],
            color=bar_colors[1],
            label="Object space" if i == 0 else "",
        )

        # Individual points
        action_points = corr_df[
            (corr_df["roi"] == roi) & (corr_df["behavior"] == "action")
        ]["correlation"]
        object_points = corr_df[
            (corr_df["roi"] == roi) & (corr_df["behavior"] == "object")
        ]["correlation"]

        ax.scatter(
            np.repeat(i - bar_width / 2, len(action_points)),
            action_points,
            color=scatter_color,
            alpha=scatter_alpha,
            s=10,
            zorder=1,
        )
        ax.scatter(
            np.repeat(i + bar_width / 2, len(object_points)),
            object_points,
            color=scatter_color,
            alpha=scatter_alpha,
            s=10,
            zorder=1,
        )

        # Paired significance bracket
        if (
            not test_result.empty
            and test_result["p-value"].values[0] < alpha_bonferroni_pairwise
        ):
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

        # Per-condition significance
        if action_row["p-value"].values[0] < alpha_bonferroni:
            ax.text(
                i - bar_width / 2,
                pairwise_test_pos - 0.05,
                "*",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )
        if object_row["p-value"].values[0] < alpha_bonferroni:
            ax.text(
                i + bar_width / 2,
                pairwise_test_pos - 0.05,
                "*",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

        # Noise ceilings for each ROI
        lower_nc = overview_df[overview_df["roi"] == roi]["lower_nc"].values[0]
        upper_nc = overview_df[overview_df["roi"] == roi]["upper_nc"].values[0]

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

    # Baseline at 0
    ax.axhline(y=0, color="black", linestyle="-", lw=1.2)

    # X-axis labels as short ROI names, in fixed order
    ax.set_xticks(x)
    ax.set_xticklabels([roi_label(r) for r in present_rois], fontsize=fontsize)

    # Legend entries for noise ceilings (dummy handles)
    ax.hlines(
        [], [], [],
        colors="grey",
        linestyles="dashed",
        lw=1.2,
        label="Upper Noise Ceiling",
    )
    ax.hlines(
        [], [], [],
        colors="grey",
        linestyles="dashed",
        lw=1.2,
        label="Lower Noise Ceiling",
    )

    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_ylim(-0.2, 0.5)

    ax.legend(
        bbox_to_anchor=(1.04, 1),
        loc="upper left",
        fontsize=fontsize - 2,
    )

    for spine in ["top", "bottom", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------
# CORRELATION ANALYSES (TASK-AGNOSTIC)
# ---------------------------------------------------------------------
def run_individual_correlations(
    behavior_action_rdvs: list[np.ndarray],
    behavior_object_rdvs: list[np.ndarray],
) -> pd.DataFrame:
    """
    Correlate each subject's brain RDM with that subject's behavior RDMs.
    Returns corr_df.
    """
    rows = []
    for roi_name in ROI_NAMES:
        _, brain_rdvs, lower_nc, upper_nc = load_brain_rdms(roi_name)

        for subj, brain_rdv in enumerate(brain_rdvs):
            sub_name = f"sub_{subj + 1:02d}"

            # action
            spearman_action = spearmanr(brain_rdv, behavior_action_rdvs[subj])
            rows.append(
                {
                    "behavior": "action",
                    "roi": roi_name,
                    "subject": sub_name,
                    "correlation": spearman_action.correlation,
                    "p-value": spearman_action.pvalue,
                    "lower_nc": lower_nc,
                    "higher_nc": upper_nc,
                }
            )

            # object
            spearman_object = spearmanr(brain_rdv, behavior_object_rdvs[subj])
            rows.append(
                {
                    "behavior": "object",
                    "roi": roi_name,
                    "subject": sub_name,
                    "correlation": spearman_object.correlation,
                    "p-value": spearman_object.pvalue,
                    "lower_nc": lower_nc,
                    "higher_nc": upper_nc,
                }
            )
    return pd.DataFrame(rows)


def run_partial_correlations(
    behavior_action_rdvs: list[np.ndarray],
    behavior_object_rdvs: list[np.ndarray],
) -> pd.DataFrame:
    """
    Partial correlations with subject-individual behavior:
        brain ~ action | object
        brain ~ object | action
    Returns corr_df.
    """
    rows = []
    for roi_name in ROI_NAMES:
        _, brain_rdvs, lower_nc, upper_nc = load_brain_rdms(roi_name)

        for subj, brain_rdv in enumerate(brain_rdvs):
            sub_name = f"sub_{subj + 1:02d}"

            df_partial = pd.DataFrame(
                {
                    "brain": brain_rdv,
                    "action": behavior_action_rdvs[subj],
                    "object": behavior_object_rdvs[subj],
                }
            )

            # brain ~ action | object
            pc_action = pg.partial_corr(
                data=df_partial,
                x="brain",
                y="action",
                covar="object",
                method="spearman",
            ).round(3)

            rows.append(
                {
                    "behavior": "action",
                    "roi": roi_name,
                    "subject": sub_name,
                    "correlation": pc_action["r"].values[0],
                    "p-value": pc_action["p-val"].values[0],
                    "lower_nc": lower_nc,
                    "higher_nc": upper_nc,
                }
            )

            # brain ~ object | action
            pc_object = pg.partial_corr(
                data=df_partial,
                x="brain",
                y="object",
                covar="action",
                method="spearman",
            ).round(3)

            rows.append(
                {
                    "behavior": "object",
                    "roi": roi_name,
                    "subject": sub_name,
                    "correlation": pc_object["r"].values[0],
                    "p-value": pc_object["p-val"].values[0],
                    "lower_nc": lower_nc,
                    "higher_nc": upper_nc,
                }
            )

    return pd.DataFrame(rows)


def run_mean_behavior_correlations(
    mean_action_rdv: np.ndarray,
    mean_object_rdv: np.ndarray,
) -> pd.DataFrame:
    """
    Simple correlations with mean behavioral RDMs across subjects.
    Returns corr_df.
    """
    rows = []
    for roi_name in ROI_NAMES:
        _, brain_rdvs, lower_nc, upper_nc = load_brain_rdms(roi_name)

        for subj, brain_rdv in enumerate(brain_rdvs):
            sub_name = f"sub_{subj + 1:02d}"

            spearman_action = spearmanr(brain_rdv, mean_action_rdv)
            rows.append(
                {
                    "behavior": "action",
                    "roi": roi_name,
                    "subject": sub_name,
                    "correlation": spearman_action.correlation,
                    "p-value": spearman_action.pvalue,
                    "lower_nc": lower_nc,
                    "higher_nc": upper_nc,
                }
            )

            spearman_object = spearmanr(brain_rdv, mean_object_rdv)
            rows.append(
                {
                    "behavior": "object",
                    "roi": roi_name,
                    "subject": sub_name,
                    "correlation": spearman_object.correlation,
                    "p-value": spearman_object.pvalue,
                    "lower_nc": lower_nc,
                    "higher_nc": upper_nc,
                }
            )

    return pd.DataFrame(rows)


def run_mean_behavior_partial_correlations(
    mean_action_rdv: np.ndarray,
    mean_object_rdv: np.ndarray,
) -> pd.DataFrame:
    """
    Partial correlations with mean behavioral RDMs:
        brain ~ action_mean | object_mean
        brain ~ object_mean | action_mean
    Returns corr_df.
    """
    rows = []
    for roi_name in ROI_NAMES:
        _, brain_rdvs, lower_nc, upper_nc = load_brain_rdms(roi_name)

        for subj, brain_rdv in enumerate(brain_rdvs):
            sub_name = f"sub_{subj + 1:02d}"

            df_partial = pd.DataFrame(
                {
                    "brain": brain_rdv,
                    "action": mean_action_rdv,
                    "object": mean_object_rdv,
                }
            )

            # brain ~ action_mean | object_mean
            pc_action = pg.partial_corr(
                data=df_partial,
                x="brain",
                y="action",
                covar="object",
                method="spearman",
            ).round(3)

            rows.append(
                {
                    "behavior": "action",
                    "roi": roi_name,
                    "subject": sub_name,
                    "correlation": pc_action["r"].values[0],
                    "p-value": pc_action["p-val"].values[0],
                    "lower_nc": lower_nc,
                    "higher_nc": upper_nc,
                }
            )

            # brain ~ object_mean | action_mean
            pc_object = pg.partial_corr(
                data=df_partial,
                x="brain",
                y="object",
                covar="action",
                method="spearman",
            ).round(3)

            rows.append(
                {
                    "behavior": "object",
                    "roi": roi_name,
                    "subject": sub_name,
                    "correlation": pc_object["r"].values[0],
                    "p-value": pc_object["p-val"].values[0],
                    "lower_nc": lower_nc,
                    "higher_nc": upper_nc,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# TASK-SPECIFIC ANALYSIS
# ---------------------------------------------------------------------
def run_task_specific_correlations(
    mean_action_rdv: np.ndarray,
    mean_object_rdv: np.ndarray,
) -> pd.DataFrame:
    """
    Correlate each subject's task-specific brain RDMs with mean behavioral RDMs.
    For each ROI (PPA, OPA, RSC), each task (action/object/fixation), and
    both behaviors (action/object).
    """
    rows = []

    for roi in ROI_TASK_NAMES:
        for task in TASKS:
            path = os.path.join(
                BRAIN_TASK_BASE,
                f"{task}_task",
                f"fmri_{roi}_{task}.npz",
            )
            mean_rdms = np.load(path)["arr_0"]  # (subjects, 90, 90)
            brain_rdvs = [
                squareform(mean_rdms[subj].round(5))
                for subj in range(mean_rdms.shape[0])
            ]
            lower_nc, upper_nc = compute_noise_ceilings(mean_rdms)

            for subj, brain_rdv in enumerate(brain_rdvs):
                sub_name = f"sub_{subj + 1:02d}"

                # behavior action
                spearman = spearmanr(brain_rdv, mean_action_rdv)
                rows.append(
                    {
                        "behavior": "action",
                        "task": task,
                        "roi": roi,
                        "subject": sub_name,
                        "correlation": spearman.correlation,
                        "p-value": spearman.pvalue,
                        "lower_nc": lower_nc,
                        "higher_nc": upper_nc,
                    }
                )

                # behavior object
                spearman = spearmanr(brain_rdv, mean_object_rdv)
                rows.append(
                    {
                        "behavior": "object",
                        "task": task,
                        "roi": roi,
                        "subject": sub_name,
                        "correlation": spearman.correlation,
                        "p-value": spearman.pvalue,
                        "lower_nc": lower_nc,
                        "higher_nc": upper_nc,
                    }
                )

    return pd.DataFrame(rows)


def summarize_task_correlations(
    corr_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarize task-specific correlations.

    Returns
    -------
    overview_df : rows per (behavior, task, roi)
    test_result_df : paired t-tests between tasks within each behavior & ROI
                     comparison column: 'action_vs_object', 'action_vs_fixation', ...
    """
    overview_rows = []
    test_rows = []

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

                mean_corr = np.mean(subset["correlation"])
                sem_corr = np.std(subset["correlation"], ddof=1) / np.sqrt(len(subset))
                t_stat, p_val = ttest_1samp(subset["correlation"], 0)

                lower_nc = subset["lower_nc"].iloc[0]
                upper_nc = subset["higher_nc"].iloc[0]

                print(
                    f"{behavior} {task} {roi} mean={mean_corr:.2f}, "
                    f"t={t_stat:.2f}, p={p_val:.3f}"
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


def plot_task_effect_bars(
    overview_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    test_result_df: pd.DataFrame,
    behavior_name: str,
    filename: str,
) -> None:
    """
    Task-effect bar plot:
        - three bars per ROI (action/object/fixation task sessions)
        - individual subject dots
        - noise ceiling band
        - stars for in-space significance at fixed height (y = 0.32)
    """
    roi_order = ["PPA", "OPA", "RSC"]
    task_order = ["action", "object", "fixation"]
    num_rois = len(roi_order)
    num_tasks = len(task_order)
    bar_width = 0.3
    fontsize = 15

    alpha = 0.05
    alpha_bonferroni_in_space = alpha / 18
    alpha_bonferroni_between_space = alpha / 18
    star_y = 0.32

    # bar colors for the three tasks
    if behavior_name == "action":
        bar_colors = ["#9e483d", "#E36858", "#eb958a"]
    else:  # "object"
        bar_colors = ["#426070", "#6FA1BB", "#a8c6d6"]

    # Filter & sort overview data
    df = overview_df[overview_df["behavior"] == behavior_name].copy()
    df["roi"] = pd.Categorical(df["roi"], roi_order, ordered=True)
    df["task"] = pd.Categorical(df["task"], task_order, ordered=True)
    df.sort_values(["roi", "task"], inplace=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    x_centers = np.arange(num_rois)  # ROI centers: 0, 1, 2

    for i, roi in enumerate(roi_order):
        roi_center = x_centers[i]
        roi_rows = df[df["roi"] == roi]
        if roi_rows.empty:
            continue

        # noise ceilings (same across tasks)
        roi_nc = roi_rows.iloc[0]
        lower_nc = roi_nc["lower_nc"]
        upper_nc = roi_nc["upper_nc"]

        # narrower NC band to avoid overlap between ROIs
        cluster_half = (num_tasks * bar_width) / 2 - 0.02  # 0.43 with 3*0.3
        nc_left = roi_center - cluster_half
        nc_right = roi_center + cluster_half

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

        # bars & dots for each task
        for j, task in enumerate(task_order):
            row = roi_rows[roi_rows["task"] == task]
            if row.empty:
                continue

            offset_index = j - (num_tasks - 1) / 2.0  # -1, 0, +1
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

            # star for in-space test vs 0
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

    # optional: if any between-task comparison sig for a ROI, add a star above center
    for i, roi in enumerate(roi_order):
        roi_tests = test_result_df[
            (test_result_df["behavior"] == behavior_name)
            & (test_result_df["roi"] == roi)
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

    ax.axhline(0, color="black", linestyle="-", linewidth=1)
    ax.set_xlim(x_centers[0] - 0.7, x_centers[-1] + 0.7)
    ax.set_xticks(x_centers)
    ax.set_xticklabels(roi_order, fontsize=fontsize)
    ax.set_ylabel("Spearman's Rho Correlation", fontsize=fontsize)
    ax.set_ylim(-0.17, 0.45)

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------
def main() -> None:
    # -------------------------------------------------------------
    # 1. Load behavior RDMs
    # -------------------------------------------------------------
    behavior_action_rdms, behavior_object_rdms = load_behavior_rdms()

    # -------------------------------------------------------------
    # 2. Plot subject-level RDMs and averages
    # -------------------------------------------------------------
    plot_all_subject_rdms(
        behavior_action_rdms,
        "Behavioral RDMs for the action task",
        "behavioral_rdms_action_task.png",
    )
    plot_all_subject_rdms(
        behavior_object_rdms,
        "Behavioral RDMs for the object task",
        "behavioral_rdms_object_task.png",
    )

    avg_behavior_action_rdm = behavior_action_rdms.mean(axis=0)
    avg_behavior_object_rdm = behavior_object_rdms.mean(axis=0)

    plot_average_rdm(
        avg_behavior_action_rdm,
        "Average RDM for the action task",
        "average_behavioral_rdm_action_task.png",
    )
    plot_average_rdm(
        avg_behavior_object_rdm,
        "Average RDM for the object task",
        "average_behavioral_rdm_object_task.png",
    )

    # Precompute subject-level RDVs
    behavior_action_rdvs = [
        squareform(behavior_action_rdms[subj].round(5))
        for subj in range(behavior_action_rdms.shape[0])
    ]
    behavior_object_rdvs = [
        squareform(behavior_object_rdms[subj].round(5))
        for subj in range(behavior_object_rdms.shape[0])
    ]

    # Mean behavior RDVs
    mean_action_rdv = squareform(avg_behavior_action_rdm.round(5))
    mean_object_rdv = squareform(avg_behavior_object_rdm.round(5))

    # -------------------------------------------------------------
    # 3. Subject-individual correlations (brain vs own behavior)
    # -------------------------------------------------------------
    print("\n=== Subject-individual behavior correlations ===")
    corr_individual = run_individual_correlations(
        behavior_action_rdvs, behavior_object_rdvs
    )
    overview_ind, tests_ind = summarize_correlations(corr_individual)
    print("\nPaired t-tests (individual):")
    print(tests_ind.round(3))

    plot_correlation_bars(
        overview_df=overview_ind,
        corr_df=corr_individual,
        test_result_df=tests_ind,
        ylabel="Spearman's Rho Correlation",
        filename="fMRI_correlations_subj_individual_inscanner_behavior.png",
        bar_width=0.35,
    )

    # -------------------------------------------------------------
    # 4. Partial correlations (subject-individual behavior)
    # -------------------------------------------------------------
    print("\n=== Partial correlations (controlling for other behavior; individual) ===")
    corr_partial = run_partial_correlations(
        behavior_action_rdvs, behavior_object_rdvs
    )
    overview_part, tests_part = summarize_correlations(corr_partial)
    print("\nPaired t-tests (partial, individual):")
    print(tests_part.round(3))

    plot_correlation_bars(
        overview_df=overview_part,
        corr_df=corr_partial,
        test_result_df=tests_part,
        ylabel="Partial Spearman's Rho Correlation",
        filename="fMRI_partial_correlations_subj_individual_inscanner_behavior.png",
        bar_width=0.35,
    )

    # -------------------------------------------------------------
    # 5. Correlations with mean behavior RDMs
    # -------------------------------------------------------------
    print("\n=== Correlations with subject-mean behavior RDMs ===")
    corr_mean = run_mean_behavior_correlations(mean_action_rdv, mean_object_rdv)
    overview_mean, tests_mean = summarize_correlations(corr_mean)
    print("\nPaired t-tests (mean behavior):")
    print(tests_mean.round(3))

    plot_correlation_bars(
        overview_df=overview_mean,
        corr_df=corr_mean,
        test_result_df=tests_mean,
        ylabel="Spearman's Rho Correlation",
        filename="fMRI_correlations_subj_average_inscanner_behavior.png",
        bar_width=0.4,
    )

    # -------------------------------------------------------------
    # 6. Partial correlations with mean behavior RDMs
    # -------------------------------------------------------------
    print("\n=== Partial correlations with subject-mean behavior RDMs ===")
    corr_mean_partial = run_mean_behavior_partial_correlations(
        mean_action_rdv, mean_object_rdv
    )
    overview_mean_part, tests_mean_part = summarize_correlations(corr_mean_partial)
    print("\nPaired t-tests (partial, mean behavior):")
    print(tests_mean_part.round(3))

    plot_correlation_bars(
        overview_df=overview_mean_part,
        corr_df=corr_mean_partial,
        test_result_df=tests_mean_part,
        ylabel="Partial Spearman's Rho Correlation",
        filename="fMRI_partial_correlations_subj_average_inscanner_behavior.png",
        bar_width=0.4,
    )

    # -------------------------------------------------------------
    # 7. Task-specific analysis (action/object/fixation sessions)
    # -------------------------------------------------------------
    print("\n=== Task-specific correlations (action/object/fixation tasks) ===")
    corr_task = run_task_specific_correlations(mean_action_rdv, mean_object_rdv)
    overview_task, tests_task = summarize_task_correlations(corr_task)

    print("\nTask-specific paired t-tests:")
    print(tests_task.round(3))

    # Plot for behavior 'action'
    plot_task_effect_bars(
        overview_df=overview_task,
        corr_df=corr_task,
        test_result_df=tests_task,
        behavior_name="action",
        filename="fMRI_correlations_action_taskeffect_subj_average_inscanner_behavior.png",
    )

    # Plot for behavior 'object'
    plot_task_effect_bars(
        overview_df=overview_task,
        corr_df=corr_task,
        test_result_df=tests_task,
        behavior_name="object",
        filename="fMRI_correlations_object_taskeffect_subj_average_inscanner_behavior.png",
    )


if __name__ == "__main__":
    main()
