#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, ttest_1samp, ttest_rel
import scipy.stats as stats
from .data_paths import behavior_path, rdm_collection_path, embedding_rdms_path


# ---------------------------------------------------------------------
# Shared helper functions
# ---------------------------------------------------------------------

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

    # Correlation of each subject with group-average (upper bound)
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


def compute_pairwise_comparisons(corr_df, base_height):
    """
    Compute pairwise t-tests between models (within-subject) with Bonferroni correction.

    Parameters
    ----------
    corr_df : pd.DataFrame
        Must have columns ['model', 'correlation'].
    base_height : float
        Base height for starting y-position (not strictly needed for the logic
        here, but kept for compatibility).

    Returns
    -------
    test_result_df : pd.DataFrame
        Columns: model1, model2, t-statistic, p-value, adjusted p-value,
                 significant (Bonferroni), pos_model1, pos_model2, y_pos
    """
    models = corr_df["model"].unique()
    test_result_df = pd.DataFrame(columns=[
        "model1", "model2", "t-statistic", "p-value"
    ])

    # Pairwise dependent t-tests across subjects
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1 = models[i]
            m2 = models[j]

            corr1 = corr_df[corr_df["model"] == m1]["correlation"]
            corr2 = corr_df[corr_df["model"] == m2]["correlation"]

            if len(corr1) > 0 and len(corr2) > 0:
                t_stat, p_val = stats.ttest_rel(
                    corr1, corr2, nan_policy="omit"
                )
                test_result_df = test_result_df.append(
                    {
                        "model1": m1,
                        "model2": m2,
                        "t-statistic": t_stat,
                        "p-value": p_val,
                    },
                    ignore_index=True,
                )

    num_tests = len(test_result_df)
    alpha = 0.05
    bonferroni_alpha = alpha / num_tests if num_tests > 0 else np.nan

    # Bonferroni-corrected p-values
    test_result_df["adjusted p-value"] = test_result_df["p-value"] * num_tests
    test_result_df["adjusted p-value"] = test_result_df["adjusted p-value"].clip(upper=1.0)
    test_result_df["significant (Bonferroni)"] = (
        test_result_df["adjusted p-value"] < bonferroni_alpha
    )

    # x positions for models (assume order of unique() is plotting order)
    x_pos_map = {m: i for i, m in enumerate(models)}
    test_result_df["pos_model1"] = test_result_df["model1"].map(x_pos_map)
    test_result_df["pos_model2"] = test_result_df["model2"].map(x_pos_map)

    # Stacked y positions
    test_result_df["y_pos"] = base_height + 0.05 * test_result_df.groupby("model1").cumcount()

    return test_result_df


def compute_spearman_squareform(rdm1, rdm2):
    """Spearman correlation between two RDMs via squareform vectorization."""
    rdm1_vec = squareform(rdm1.round(5))
    rdm2_vec = squareform(rdm2.round(5))
    return spearmanr(rdm1_vec, rdm2_vec).correlation


# ---------------------------------------------------------------------
# GPT single / multi / BoW vs behavior
# ---------------------------------------------------------------------

def run_gpt_behavior_analysis(
    behavior_name,
    behavior_rdm_path,
    gpt_rdms_info,
    bar_colors,
    fig_basename,
    script_root,
):
    """
    Run GPT-behavior RSA for a given behavior (action/object) and plot.

    Parameters
    ----------
    behavior_name : str
        'action' or 'object', used in DataFrame & titles.
    behavior_rdm_path : str
        Path to fmri_behavior_*_rdms.npz.
    gpt_rdms_info : list of (name, path)
        e.g. [('single', path1), ('multi', path2), ('bow', path3)].
    bar_colors : dict
        Mapping model name -> color hex.
    fig_basename : str
        File name for the saved figure (e.g. 'GPT_action.png').
    script_root : str
        Root directory of this script (for saving Figures).
    """
    # -----------------------------------------------------------
    # Load behavior RDMs and GPT RDMs
    # -----------------------------------------------------------
    behavior_rdms = np.load(behavior_rdm_path)["arr_0"]
    beh_rdvs = [squareform(behavior_rdms[subj].round(5))
                for subj in range(behavior_rdms.shape[0])]

    lower_nc, upper_nc = compute_noise_ceilings(behavior_rdms)

    model_names = [name for name, _ in gpt_rdms_info]
    gpt_rdvs = {}

    for name, path in gpt_rdms_info:
        rdm = np.load(path)
        gpt_rdvs[name] = squareform(rdm.round(5))

    # -----------------------------------------------------------
    # Subject-level correlations
    # -----------------------------------------------------------
    corr_rows = []

    for subj, rdv in enumerate(beh_rdvs):
        sub_name = f"sub_0{subj + 1}"
        for name in model_names:
            spearman_c, p_val = spearmanr(rdv, gpt_rdvs[name])
            corr_rows.append(
                {
                    "model": name,
                    "behavior": behavior_name,
                    "subject": sub_name,
                    "correlation": spearman_c,
                    "p-value": p_val,
                    "lower_nc": lower_nc,
                    "higher_nc": upper_nc,
                }
            )

    corr = pd.DataFrame.from_records(corr_rows)

    # -----------------------------------------------------------
    # Model-level summary
    # -----------------------------------------------------------
    overview_rows = []
    for name in model_names:
        corr_subset = corr[corr["model"] == name]
        mean_corr = corr_subset["correlation"].mean()
        sem_corr = corr_subset["correlation"].std() / np.sqrt(len(corr_subset["correlation"]))

        t, p_val = ttest_1samp(corr_subset["correlation"], 0.0)
        print(f"{behavior_name} | GPT-{name}: t={t.round(2)} p={p_val.round(2)}")

        lower_nc_model = corr_subset["lower_nc"].iloc[0]
        upper_nc_model = corr_subset["higher_nc"].iloc[0]

        overview_rows.append(
            {
                "model": name,
                "behavior": behavior_name,
                "subject": "mean",
                "correlation": mean_corr,
                "sem": sem_corr,
                "p-value": p_val,
                "lower_nc": lower_nc_model,
                "upper_nc": upper_nc_model,
            }
        )

    overview_df = pd.DataFrame.from_records(overview_rows)

    # -----------------------------------------------------------
    # Pairwise tests between models
    # -----------------------------------------------------------
    pairwise_test_pos = overview_df["upper_nc"].max() + 0.09
    test_result_df = compute_pairwise_comparisons(corr, pairwise_test_pos)

    # -----------------------------------------------------------
    # Plot
    # -----------------------------------------------------------
    df = overview_df.copy()

    alpha = 0.05
    alpha_bonferroni = alpha / len(df["model"])

    fig, ax = plt.subplots(figsize=(5, 5), facecolor="white")

    x_positions = np.arange(len(df["model"]))
    width = 0.8

    # Bar plotting in order of model_names
    for idx, name in enumerate(model_names):
        group_df = df[df["model"] == name]
        color = bar_colors[name]

        ax.bar(
            x_positions[idx],
            group_df["correlation"].values[0],
            width=width,
            yerr=group_df["sem"].values[0],
            capsize=5,
            color=color,
        )

        # scatter for subject-level correlations
        model_df = corr[corr["model"] == name]
        ax.scatter(
            [x_positions[idx]] * len(model_df["correlation"]),
            model_df["correlation"],
            color="gray",
            alpha=0.5,
            s=10,
            zorder=10,
        )

        # significance star for one-sample test vs zero
        p_val = group_df["p-value"].values[0]
        if p_val < alpha_bonferroni:
            ax.text(
                x_positions[idx],
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

    # Pairwise comparison lines
    pairwise_line_height = df["upper_nc"].max() + 0.15
    for _, row in test_result_df.iterrows():
        if row["significant (Bonferroni)"]:
            pos1 = row["pos_model1"]
            pos2 = row["pos_model2"]
            ax.plot(
                [pos1, pos2],
                [pairwise_line_height, pairwise_line_height],
                color="black",
                linestyle="-",
            )
            pairwise_line_height += 0.03

    # Cosmetics
    ax.set_ylabel("Spearman's rho Correlation", fontsize=15)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["Single", "Multi", "bow"], rotation=45, ha="right")
    plt.yticks(fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.8, len(x_positions))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.tight_layout()

    figures_dir = os.path.join(script_root, "Figures")
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = os.path.join(figures_dir, fig_basename)
    plt.savefig(fig_path, dpi=300, transparent=True)
    plt.close(fig)

    print(f"Saved figure: {fig_path}")


# ---------------------------------------------------------------------
# MiniLM caption / affordance / object prompts vs behavior
# ---------------------------------------------------------------------

def plot_prompt_results(
    tasks,
    pretty_labels,
    mean_correlations,
    sem_correlations,
    correlations,
    noise_ceiling_lower,
    noise_ceiling_upper,
    p_values,
    pairwise_p_values,
    color_map,
    fig_basename,
    script_root,
):
    """
    Plot bar + scatter for prompt-based embedding RDM vs behavior.

    Parameters
    ----------
    tasks : list of str
        Internal task names.
    pretty_labels : list of str
        X tick labels for each task.
    mean_correlations : array-like, shape (n_tasks,)
    sem_correlations : array-like, shape (n_tasks,)
    correlations : array-like, shape (n_subj, n_tasks)
    noise_ceiling_lower, noise_ceiling_upper : float
    p_values : list of float
        p-values (vs. zero) per task.
    pairwise_p_values : 2D array
        Pairwise p-values [i, j].
    color_map : dict
        task_name -> color hex.
    fig_basename : str
        Output file name.
    script_root : str
        Path of this script.
    """
    n_tasks = len(tasks)
    x_positions = np.arange(n_tasks)
    width = 0.80

    fig, ax = plt.subplots(figsize=(5, 5), facecolor="white")

    for i, task in enumerate(tasks):
        color = color_map[task]
        ax.bar(
            x_positions[i],
            mean_correlations[i],
            width=width,
            yerr=sem_correlations[i],
            capsize=5,
            color=color,
        )
        # scatter subject-level values
        ax.scatter(
            [x_positions[i]] * correlations.shape[0],
            correlations[:, i],
            color="gray",
            alpha=0.5,
            s=10,
            zorder=10,
        )

    # Noise ceilings
    ax.fill_between(
        [-0.8, len(x_positions) - 0.2],
        noise_ceiling_lower,
        noise_ceiling_upper,
        color="gray",
        alpha=0.2,
    )
    ax.axhline(
        y=noise_ceiling_upper,
        xmin=-0.8,
        xmax=len(x_positions) - 0.2,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Upper Noise Ceiling",
    )
    ax.axhline(
        y=noise_ceiling_lower,
        xmin=-0.8,
        xmax=len(x_positions) - 0.2,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Lower Noise Ceiling",
    )

    # Axes
    ax.set_ylabel("Spearman's rho Correlation", fontsize=15)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(pretty_labels, rotation=45, ha="right")
    plt.yticks(fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.8, len(x_positions))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    # Significance vs zero (Bonferroni across tasks)
    alpha = 0.05
    alpha_bonf = alpha / n_tasks
    for i, p_val in enumerate(p_values):
        if p_val < alpha_bonf:
            ax.text(
                x_positions[i],
                noise_ceiling_upper + 0.02,
                "*",
                ha="center",
                va="bottom",
                fontsize=12,
                color="black",
            )

    # Pairwise significant differences
    current_height = noise_ceiling_upper + 0.1
    height_increment = 0.02
    for (i, j) in itertools.combinations(range(n_tasks), 2):
        if pairwise_p_values[i, j] < alpha_bonf:
            y = current_height
            ax.plot(
                [x_positions[i], x_positions[i], x_positions[j], x_positions[j]],
                [y + 0.05, y + 0.05, y + 0.05, y + 0.05],
                color="black",
            )
            current_height += height_increment

    plt.tight_layout()

    figures_dir = os.path.join(script_root, "Figures")
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = os.path.join(figures_dir, fig_basename)
    fig.savefig(fig_path, dpi=900, transparent=True)
    plt.close(fig)

    print(f"Saved figure: {fig_path}")


def run_prompt_embedding_analysis(behavior_name, script_root):
    """
    Run the MiniLM caption/affordance/object prompt analysis for
    a given behavior ('action' or 'object').
    """
    # Paths to fMRI behavior RDMs
    behavior_rdm_path = behavior_path(
        "VISACT_fmri_behavior", f"fmri_behavior_{behavior_name}_rdms.npz"
    )
    behavior_rdms = np.load(behavior_rdm_path)["arr_0"]

    # Basic shape sanity check (optional, can be adapted)
    if behavior_rdms.shape[0] != 20:
        print(f"Warning: expected 20 subjects, got {behavior_rdms.shape[0]} for {behavior_name}")
    # Compute noise ceilings
    lower_nc, upper_nc = compute_noise_ceilings(behavior_rdms)

    # Embedding settings
    tasks = ["caption_prompt", "affordance_prompt", "object_prompt"]
    pretty_labels = ["Caption", "Affordance", "Objects"]
    metric = "correlation"
    embedding_space = "minilm"

    # Load RDMs for each task
    rdms = []
    for task in tasks:
        path = embedding_rdms_path(
            embedding_space, f"{embedding_space}_{task}_fMRI_sorted_{metric}_rdm.npy"
        )
        rdms.append(np.load(path))

    n_subj = behavior_rdms.shape[0]
    n_tasks = len(tasks)

    # Subject-level correlations for each task
    correlations = np.zeros((n_subj, n_tasks))
    for i, task_rdm in enumerate(rdms):
        for participant in range(n_subj):
            corr_val = compute_spearman_squareform(
                behavior_rdms[participant], task_rdm
            )
            correlations[participant, i] = corr_val

    # Means & SEM
    mean_correlations = correlations.mean(axis=0)
    sem_correlations = correlations.std(axis=0, ddof=1) / np.sqrt(n_subj)

    print(f"\n{behavior_name} | mean correlations per task:", mean_correlations)

    # p-values vs zero
    p_values = [
        ttest_1samp(correlations[:, i], 0).pvalue for i in range(n_tasks)
    ]

    # Pairwise p-values
    pairwise_p_values = np.zeros((n_tasks, n_tasks))
    for i, j in itertools.combinations(range(n_tasks), 2):
        p_val = ttest_rel(correlations[:, i], correlations[:, j]).pvalue
        pairwise_p_values[i, j] = p_val

    # Colors differ for action vs object
    if behavior_name == "action":
        color_map = {
            "caption_prompt": "#c9184a",
            "affordance_prompt": "#e5383b",
            "object_prompt": "#ff7f51",
        }
        fig_basename = "GPT_captions_affordance.png"
    else:  # object
        color_map = {
            "caption_prompt": "#0077b6",
            "affordance_prompt": "#48cae4",
            "object_prompt": "#0096c7",
        }
        fig_basename = "GPT_captions_object.png"

    plot_prompt_results(
        tasks=tasks,
        pretty_labels=pretty_labels,
        mean_correlations=mean_correlations,
        sem_correlations=sem_correlations,
        correlations=correlations,
        noise_ceiling_lower=lower_nc,
        noise_ceiling_upper=upper_nc,
        p_values=p_values,
        pairwise_p_values=pairwise_p_values,
        color_map=color_map,
        fig_basename=fig_basename,
        script_root=script_root,
    )


# ---------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------

def main():
    script_root = os.path.dirname(os.path.abspath(__file__))

    # -----------------------------------------------------------------
    # GPT single / multi / BoW vs ACTION behavior
    # -----------------------------------------------------------------
    action_behavior_rdm_path = behavior_path(
        "VISACT_fmri_behavior", "fmri_behavior_action_rdms.npz"
    )

    gpt_action_rdms = [
        (
            "single",
            rdm_collection_path(
                "CHAT_GPT", "VISACT_fMRI", "single_action_fMRI_order_euclidean_90_rdm.npy"
            ),
        ),
        (
            "multi",
            rdm_collection_path(
                "CHAT_GPT", "VISACT_fMRI", "multi_action_fMRI_order_euclidean_90_rdm.npy"
            ),
        ),
        (
            "bow",
            rdm_collection_path(
                "CHAT_GPT", "VISACT_fMRI", "bow_action_fMRI_order_correlation_90_rdm.npy"
            ),
        ),
    ]

    action_colors = {
        "single": "#e5383b",
        "multi": "#ff7f51",
        "bow": "#c9184a",
    }

    run_gpt_behavior_analysis(
        behavior_name="action",
        behavior_rdm_path=action_behavior_rdm_path,
        gpt_rdms_info=gpt_action_rdms,
        bar_colors=action_colors,
        fig_basename="GPT_action.png",
        script_root=script_root,
    )

    # -----------------------------------------------------------------
    # GPT single / multi / BoW vs OBJECT behavior
    # -----------------------------------------------------------------
    object_behavior_rdm_path = behavior_path(
        "VISACT_fmri_behavior", "fmri_behavior_object_rdms.npz"
    )

    gpt_object_rdms = [
        (
            "single",
            rdm_collection_path(
                "CHAT_GPT", "VISACT_fMRI", "single_object_fMRI_order_euclidean_90_rdm.npy"
            ),
        ),
        (
            "multi",
            rdm_collection_path(
                "CHAT_GPT", "VISACT_fMRI", "multi_object_fMRI_order_correlation_90_rdm.npy"
            ),
        ),
        (
            "bow",
            rdm_collection_path(
                "CHAT_GPT", "VISACT_fMRI", "bow_object_fMRI_order_correlation_90_rdm.npy"
            ),
        ),
    ]

    object_colors = {
        "single": "#48cae4",
        "multi": "#0096c7",
        "bow": "#0077b6",
    }

    run_gpt_behavior_analysis(
        behavior_name="object",
        behavior_rdm_path=object_behavior_rdm_path,
        gpt_rdms_info=gpt_object_rdms,
        bar_colors=object_colors,
        fig_basename="GPT_object.png",
        script_root=script_root,
    )

    # -----------------------------------------------------------------
    # MiniLM caption/affordance/object prompts vs ACTION & OBJECT
    # -----------------------------------------------------------------
    run_prompt_embedding_analysis("action", script_root)
    run_prompt_embedding_analysis("object", script_root)


if __name__ == "__main__":
    main()
