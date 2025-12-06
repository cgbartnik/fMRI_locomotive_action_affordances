import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from nilearn import plotting, image, datasets, surface
from nilearn.image import (
    load_img,
    resample_to_img,
    math_img,
    mean_img,
    get_data,
)
from nilearn.plotting import plot_surf_stat_map
from nilearn.datasets import fetch_atlas_juelich, fetch_surf_fsaverage
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference

from scipy.spatial.distance import squareform

from data_paths import searchlight_base_path, searchlight_partial_path

# ---------------------------------------------------------------------
# Paths / output setup
# ---------------------------------------------------------------------
# Folder where this script lives (fallback to CWD in interactive use)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

# Figures folder next to the script
FIGURES_DIR = os.path.join(SCRIPT_DIR, "Figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def save_fig(filename):
    """Save current matplotlib figure into the Figures folder."""
    out_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out_path, dpi=300, transparent=True)
    print(f"Saved figure: {out_path}")


# ---------------------------------------------------------------------
# ROI utilities
# ---------------------------------------------------------------------
def compute_avg_roi_vec(hemisphere, roi, subs, threshold=4):
    """
    Compute the average ROI vector across all participants and select values over a threshold.

    Parameters
    ----------
    hemisphere : {"left", "right"}
    roi : str
        e.g. "OPA", "PPA", "V1d", ...
    subs : list of str
        e.g. ["subj01", "subj02", ...]
    threshold : int, default 4

    Returns
    -------
    np.ndarray
        Binary vector: 1 where sum across subs > threshold, else 0.
    """

    def get_roi_vec_inner(hemisphere_inner, roi_inner, sub_inner):
        # Define the ROI class based on the selected ROI
        if roi_inner in ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]:
            roi_class = "prf-visualrois"
        elif roi_inner in ["EBA", "FBA-1", "FBA-2", "mTL-bodies"]:
            roi_class = "floc-bodies"
        elif roi_inner in ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"]:
            roi_class = "floc-faces"
        elif roi_inner in ["OPA", "PPA", "RSC"]:
            roi_class = "floc-places"
        elif roi_inner in ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]:
            roi_class = "floc-words"
        elif roi_inner in [
            "early",
            "midventral",
            "midlateral",
            "midparietal",
            "ventral",
            "lateral",
            "parietal",
        ]:
            roi_class = "streams"
        else:
            raise ValueError(f"Unknown ROI: {roi_inner}")

        roi_class_dir = os.path.join(
            f"/data/algonauts_2023/data/{sub_inner}",
            "roi_masks",
            f"{hemisphere_inner[0]}h.{roi_class}_fsaverage_space.npy",
        )
        roi_map_dir = os.path.join(
            f"/data/algonauts_2023/data/{sub_inner}",
            "roi_masks",
            f"mapping_{roi_class}.npy",
        )

        fsaverage_roi_class = np.load(roi_class_dir)
        roi_map = np.load(roi_map_dir, allow_pickle=True).item()

        # Select the vertices corresponding to the ROI of interest
        roi_mapping = list(roi_map.keys())[list(roi_map.values()).index(roi_inner)]
        fsaverage_roi = np.asarray(fsaverage_roi_class == roi_mapping, dtype=int)
        return fsaverage_roi

    # Initialize with zeros using the size of the ROI vector for the first subject
    initial_roi = get_roi_vec_inner(hemisphere, roi, subs[0])
    full_sum = np.zeros_like(initial_roi)

    # Sum the ROI vectors across all subjects
    for sub in subs:
        full_sum += get_roi_vec_inner(hemisphere, roi, sub)

    # Transform the array based on the threshold
    transformed_arr = np.where(full_sum > threshold, 1, 0)
    return transformed_arr


# ---------------------------------------------------------------------
# Shared config (coords, fsaverage, etc.)
# ---------------------------------------------------------------------
subs = ["subj01", "subj02", "subj03", "subj04", "subj05", "subj06", "subj07", "subj08"]

OPA_coords = [(10, -140), (10, -40)]
PPA_coords = [(-20, -40), (-20, 220)]

fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")
fsaverage_meshes = fetch_surf_fsaverage("fsaverage5")
fsaverage_sulcal = {
    "left": fsaverage["sulc_left"],
    "right": fsaverage["sulc_right"],
}


# ---------------------------------------------------------------------
# Helper to run second-level GLM + permutation + masking
# ---------------------------------------------------------------------
def second_level_and_mask(path, space, permutations=1000, two_sided=False):
    """
    Run second-level GLM and non-parametric inference, return:
    - mean_image
    - masked_mean_img (thresholded/permutation-masked mean)
    """
    subjects = [f"{i:03d}" for i in range(1, 21)]
    full_data = [
        image.load_img(f"{path}sub_{sub}_images/sub_{sub}{space}") for sub in subjects
    ]

    design_matrix = pd.DataFrame(
        [1] * len(subjects),
        columns=["intercept"],
    )

    second_level_model = SecondLevelModel(smoothing_fwhm=8.0, n_jobs=2).fit(
        full_data, design_matrix=design_matrix
    )

    p_val = second_level_model.compute_contrast(output_type="p_value")
    n_voxels = np.sum(get_data(second_level_model.masker_.mask_img_))

    # Correcting the p-values for multiple testing and taking negative logarithm
    neg_log_pval = math_img(
        f"-np.log10(np.minimum(1, img * {n_voxels!s}))",
        img=p_val,
    )

    out_dict = non_parametric_inference(
        full_data,
        design_matrix=design_matrix,
        model_intercept=True,
        n_perm=permutations,
        two_sided_test=two_sided,
        smoothing_fwhm=8.0,
        n_jobs=2,
        threshold=0.001,
    )

    mean_image = mean_img(full_data)
    thresholded_img = math_img("img > 1.3", img=out_dict["logp_max_t"])
    masked_mean_img = math_img("img1 * img2", img1=mean_image, img2=thresholded_img)

    return mean_image, masked_mean_img, neg_log_pval, out_dict


# ---------------------------------------------------------------------
# Scene ROIs (Julian atlas) and combined mask
# ---------------------------------------------------------------------
def build_scene_roi_mask(mean_image):
    PPA_roi_right = load_img("/data/Julian_atlas/scene_parcels/rPPA.img")
    TOS_roi_right = load_img("/data/Julian_atlas/scene_parcels/rTOS.img")
    RSC_roi_right = load_img("/data/Julian_atlas/scene_parcels/rRSC.img")

    PPA_roi_left = load_img("/data/Julian_atlas/scene_parcels/lPPA.img")
    TOS_roi_left = load_img("/data/Julian_atlas/scene_parcels/lTOS.img")
    RSC_roi_left = load_img("/data/Julian_atlas/scene_parcels/lRSC.img")

    # Combine both hemispheres for each ROI
    PPA_combined = math_img("img1 + img2", img1=PPA_roi_left, img2=PPA_roi_right)
    TOS_combined = math_img("img1 + img2", img1=TOS_roi_left, img2=TOS_roi_right)
    RSC_combined = math_img("img1 + img2", img1=RSC_roi_left, img2=RSC_roi_right)

    # Resample the ROIs to the space of mean_image
    PPA_resampled = resample_to_img(
        PPA_combined, mean_image, force_resample=True, interpolation="nearest"
    )
    TOS_resampled = resample_to_img(
        TOS_combined, mean_image, force_resample=True, interpolation="nearest"
    )
    RSC_resampled = resample_to_img(
        RSC_combined, mean_image, force_resample=True, interpolation="nearest"
    )

    # Fetch the Juelich atlas and make V1
    juelich = fetch_atlas_juelich("maxprob-thr50-2mm")
    atlas_img = load_img(juelich.maps)
    label_names = juelich.labels
    region_name = "GM Visual cortex V1 BA17"
    region_index = label_names.index(region_name)
    v1_mask_img = math_img(f"img == {region_index}", img=atlas_img)
    v1_resampled = resample_to_img(
        v1_mask_img, mean_image, force_resample=True, interpolation="nearest"
    )

    PPA_numbered = math_img("img * 1", img=PPA_resampled)
    TOS_numbered = math_img("img * 2", img=TOS_resampled)
    RSC_numbered = math_img("img * 3", img=RSC_resampled)
    V1_numbered = math_img("img * 4", img=v1_resampled)

    combined_mask = math_img(
        "ppa + tos + rsc",
        ppa=PPA_numbered,
        tos=TOS_numbered,
        rsc=RSC_numbered,
    )

    final_mask = math_img(
        "combined + v1 * (combined == 0)",
        combined=combined_mask,
        v1=V1_numbered,
    )
    return final_mask


# ---------------------------------------------------------------------
# Helper to plot and save OPA/PPA views for a given surface stat map
#   thresholded=True  -> threshold=0.001, cmap="YlOrRd_r", vmax=0.083
#   thresholded=False -> no threshold, vmin=-0.016, vmax=0.083, default cmap
# ---------------------------------------------------------------------
def plot_and_save_views(surf_img, combined, ROIs, label_prefix, thresholded=True):
    for idx, hemisphere in enumerate(["left", "right"]):

        # Common keyword args
        common_kwargs = dict(
            stat_map=surf_img[hemisphere],
            surf_mesh=fsaverage[f"infl_{hemisphere}"],
            hemi=hemisphere,
            colorbar=True,
            bg_map=fsaverage[f"sulc_{hemisphere}"],
            symmetric_cbar=False,
        )

        # ---------- OPA view ----------
        if thresholded:
            figure = plot_surf_stat_map(
                view=OPA_coords[idx],
                threshold=0.001,
                cmap="YlOrRd_r",
                vmax=0.083,
                **common_kwargs,
            )
        else:
            figure = plot_surf_stat_map(
                view=OPA_coords[idx],
                vmin=-0.016,
                vmax=0.083,
                **common_kwargs,
            )

        plotting.plot_surf_contours(
            surf_mesh=fsaverage[f"infl_{hemisphere}"],
            roi_map=combined[hemisphere],
            hemi=hemisphere,
            levels=[1, 2, 3, 4],
            colors=["#000000", "#208b3a", "#03045e", "#ff0a54"],
            labels=["V1 (NSD)", "PPA (NSD)", "OPA (NSD)", "RSC (NSD)"],
            figure=figure,
        )
        plotting.plot_surf_contours(
            surf_mesh=fsaverage[f"infl_{hemisphere}"],
            roi_map=ROIs[hemisphere],
            hemi=hemisphere,
            levels=[1, 2, 3, 4],
            colors=["#2dc653", "#00b4d8", "#ff758f", "#343a40"],
            labels=["PPA", "TOS", "RSC", "V1"],
            figure=figure,
            legend=True,
        )
        save_fig(f"{label_prefix}_OPA_view_{hemisphere}.png")
        plt.close()

        # ---------- PPA view ----------
        if thresholded:
            figure = plot_surf_stat_map(
                view=PPA_coords[idx],
                threshold=0.001,
                cmap="YlOrRd_r",
                vmax=0.083,
                **common_kwargs,
            )
        else:
            figure = plot_surf_stat_map(
                view=PPA_coords[idx],
                vmin=-0.016,
                vmax=0.083,
                **common_kwargs,
            )

        plotting.plot_surf_contours(
            surf_mesh=fsaverage[f"infl_{hemisphere}"],
            roi_map=combined[hemisphere],
            hemi=hemisphere,
            levels=[1, 2, 3, 4],
            colors=["#000000", "#208b3a", "#03045e", "#ff0a54"],
            labels=["V1 (NSD)", "PPA (NSD)", "OPA (NSD)", "RSC (NSD)"],
            figure=figure,
        )
        plotting.plot_surf_contours(
            surf_mesh=fsaverage[f"infl_{hemisphere}"],
            roi_map=ROIs[hemisphere],
            hemi=hemisphere,
            levels=[1, 2, 3, 4],
            colors=["#2dc653", "#00b4d8", "#ff758f", "#343a40"],
            labels=["PPA", "TOS", "RSC", "V1"],
            figure=figure,
            legend=True,
        )
        save_fig(f"{label_prefix}_PPA_view_{hemisphere}.png")
        plt.close()


# ---------------------------------------------------------------------
# EXAMPLE 1: Base affordance space
# ---------------------------------------------------------------------
base_path = searchlight_base_path()
base_space = "_aff_corr.nii.gz"

mean_image, masked_mean_img, neg_log_pval, out_dict = second_level_and_mask(
    base_path, base_space, permutations=1000, two_sided=False
)

final_mask = build_scene_roi_mask(mean_image)

# Project masks & data to surface (thresholded)
threshold = 3
combined_surf = {}
ROIs_surf = {}
surf_img_thr = {}

for hemisphere in ["left", "right"]:
    left_V1d = compute_avg_roi_vec(hemisphere, "V1d", subs, threshold)
    left_V1v = compute_avg_roi_vec(hemisphere, "V1v", subs, threshold)
    left_PPA = compute_avg_roi_vec(hemisphere, "PPA", subs, threshold) * 2
    left_OPA = compute_avg_roi_vec(hemisphere, "OPA", subs, threshold) * 3
    left_RSC = compute_avg_roi_vec(hemisphere, "RSC", subs, threshold) * 4

    combined = np.sum([left_V1d, left_V1v, left_PPA, left_OPA, left_RSC], axis=0)
    combined_surf[hemisphere] = combined

    ROIs_surf[hemisphere] = surface.vol_to_surf(
        final_mask,
        surf_mesh=fsaverage[f"pial_{hemisphere}"],
        inner_mesh=fsaverage[f"white_{hemisphere}"],
    )
    surf_img_thr[hemisphere] = surface.vol_to_surf(
        masked_mean_img,
        surf_mesh=fsaverage[f"pial_{hemisphere}"],
        inner_mesh=fsaverage[f"white_{hemisphere}"],
    )

# Thresholded base affordance
plot_and_save_views(
    surf_img_thr,
    combined_surf,
    ROIs_surf,
    label_prefix="Base_affordance_thresholded",
    thresholded=True,
)

# Unthresholded base affordance
surf_img_unthr = {}
for hemisphere in ["left", "right"]:
    surf_img_unthr[hemisphere] = surface.vol_to_surf(
        mean_image,
        surf_mesh=fsaverage[f"pial_{hemisphere}"],
        inner_mesh=fsaverage[f"white_{hemisphere}"],
    )

plot_and_save_views(
    surf_img_unthr,
    combined_surf,
    ROIs_surf,
    label_prefix="Base_affordance_unthresholded",
    thresholded=False,
)


# ---------------------------------------------------------------------
# EXAMPLE 2: Affordance–GIST partialed out
# ---------------------------------------------------------------------
partial_path = searchlight_partial_path()
space_aff_gist = "_affordance_gist_partialed_out.nii.gz"

mean_image2, masked_mean_img2, neg_log_pval2, out_dict2 = second_level_and_mask(
    partial_path, space_aff_gist, permutations=1000, two_sided=True
)

combined_surf2 = {}
ROIs_surf2 = {}
surf_img_thr2 = {}

for hemisphere in ["left", "right"]:
    left_V1d = compute_avg_roi_vec(hemisphere, "V1d", subs, threshold)
    left_V1v = compute_avg_roi_vec(hemisphere, "V1v", subs, threshold)
    left_PPA = compute_avg_roi_vec(hemisphere, "PPA", subs, threshold) * 2
    left_OPA = compute_avg_roi_vec(hemisphere, "OPA", subs, threshold) * 3
    left_RSC = compute_avg_roi_vec(hemisphere, "RSC", subs, threshold) * 4

    combined = np.sum([left_V1d, left_V1v, left_PPA, left_OPA, left_RSC], axis=0)
    combined_surf2[hemisphere] = combined

    ROIs_surf2[hemisphere] = surface.vol_to_surf(
        final_mask,
        surf_mesh=fsaverage[f"pial_{hemisphere}"],
        inner_mesh=fsaverage[f"white_{hemisphere}"],
    )
    surf_img_thr2[hemisphere] = surface.vol_to_surf(
        masked_mean_img2,
        surf_mesh=fsaverage[f"pial_{hemisphere}"],
        inner_mesh=fsaverage[f"white_{hemisphere}"],
    )

plot_and_save_views(
    surf_img_thr2,
    combined_surf2,
    ROIs_surf2,
    label_prefix="Affordance_GIST_partialed_out_thresholded",
    thresholded=True,
)
 
surf_img_unthr2 = {}
for hemisphere in ["left", "right"]:
    surf_img_unthr2[hemisphere] = surface.vol_to_surf(
        mean_image2,
        surf_mesh=fsaverage[f"pial_{hemisphere}"],
        inner_mesh=fsaverage[f"white_{hemisphere}"],
    )
        
plot_and_save_views(
    surf_img_unthr2,
    combined_surf2,
    ROIs_surf2,
    label_prefix="Affordance_GIST_partialed_out_unthresholded",
    thresholded=False,
)


space_aff_obj = "_affordance_object_partialed_out.nii.gz"

mean_image3, masked_mean_img3, neg_log_pval3, out_dict3 = second_level_and_mask(
    partial_path, space_aff_obj, permutations=1000, two_sided=True
)

combined_surf3 = {}
ROIs_surf3 = {}
surf_img_thr3 = {}

for hemisphere in ["left", "right"]:
    # NSD ROI maps (same threshold value as above)
    left_V1d = compute_avg_roi_vec(hemisphere, "V1d", subs, threshold)
    left_V1v = compute_avg_roi_vec(hemisphere, "V1v", subs, threshold)
    left_PPA = compute_avg_roi_vec(hemisphere, "PPA", subs, threshold) * 2
    left_OPA = compute_avg_roi_vec(hemisphere, "OPA", subs, threshold) * 3
    left_RSC = compute_avg_roi_vec(hemisphere, "RSC", subs, threshold) * 4

    combined = np.sum([left_V1d, left_V1v, left_PPA, left_OPA, left_RSC], axis=0)
    combined_surf3[hemisphere] = combined

    # Julian atlas ROIs projected to surface
    ROIs_surf3[hemisphere] = surface.vol_to_surf(
        final_mask,
        surf_mesh=fsaverage[f"pial_{hemisphere}"],
        inner_mesh=fsaverage[f"white_{hemisphere}"],
    )

    # Thresholded group map projected to surface
    surf_img_thr3[hemisphere] = surface.vol_to_surf(
        masked_mean_img3,
        surf_mesh=fsaverage[f"pial_{hemisphere}"],
        inner_mesh=fsaverage[f"white_{hemisphere}"],
    )

# Thresholded Affordance–Object partialed out
plot_and_save_views(
    surf_img_thr3,
    combined_surf3,
    ROIs_surf3,
    label_prefix="Affordance_Object_partialed_out_thresholded",
    thresholded=True,
)

# Unthresholded Affordance–Object partialed out
surf_img_unthr3 = {}
for hemisphere in ["left", "right"]:
    surf_img_unthr3[hemisphere] = surface.vol_to_surf(
        mean_image3,
        surf_mesh=fsaverage[f"pial_{hemisphere}"],
        inner_mesh=fsaverage[f"white_{hemisphere}"],
    )

plot_and_save_views(
    surf_img_unthr3,
    combined_surf3,
    ROIs_surf3,
    label_prefix="Affordance_Object_partialed_out_unthresholded",
    thresholded=False,
)
