import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from nilearn.image import load_img, resample_to_img, math_img
from nilearn.masking import apply_mask
from nilearn.datasets import fetch_atlas_juelich
from scipy.stats import sem, wilcoxon

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_paths import searchlight_base_path, searchlight_partial_path


# ---------------------------------------------------------------------
# ROI & atlas setup (loaded once, reused)
# ---------------------------------------------------------------------

# Scene parcel ROIs
PPA_roi_right = load_img("/data/Julian_atlas/scene_parcels/rPPA.img")
TOS_roi_right = load_img("/data/Julian_atlas/scene_parcels/rTOS.img")
RSC_roi_right = load_img("/data/Julian_atlas/scene_parcels/rRSC.img")

PPA_roi_left = load_img("/data/Julian_atlas/scene_parcels/lPPA.img")
TOS_roi_left = load_img("/data/Julian_atlas/scene_parcels/lTOS.img")
RSC_roi_left = load_img("/data/Julian_atlas/scene_parcels/lRSC.img")

# Combine hemispheres for each ROI
PPA_combined = math_img("img1 + img2", img1=PPA_roi_left, img2=PPA_roi_right)
TOS_combined = math_img("img1 + img2", img1=TOS_roi_left, img2=TOS_roi_right)
RSC_combined = math_img("img1 + img2", img1=RSC_roi_left, img2=RSC_roi_right)

# Juelich atlas (V1)
juelich = fetch_atlas_juelich("maxprob-thr50-2mm")
atlas_img = load_img(juelich.maps)
label_names = list(juelich.labels)

region_name = "GM Visual cortex V1 BA17"
region_index = label_names.index(region_name)
v1_mask_img = math_img(f"img == {region_index}", img=atlas_img)


# ---------------------------------------------------------------------
# Function to compute mean z-values per ROI
# ---------------------------------------------------------------------
def compute_roi_means(z_map):
    # Resample ROIs to z_map space
    PPA_resampled = resample_to_img(PPA_combined, z_map, force_resample=True, interpolation="nearest")
    TOS_resampled = resample_to_img(TOS_combined, z_map, force_resample=True, interpolation="nearest")
    RSC_resampled = resample_to_img(RSC_combined, z_map, force_resample=True, interpolation="nearest")
    v1_resampled = resample_to_img(v1_mask_img, z_map, force_resample=True, interpolation="nearest")

    rois = {
        "PPA": PPA_resampled,
        "TOS": TOS_resampled,
        "RSC": RSC_resampled,
        "V1": v1_resampled,
    }

    mean_corr_values = {}
    for roi_name, roi_mask in rois.items():
        roi_values = apply_mask(z_map, roi_mask)
        mean_corr_values[roi_name] = np.mean(roi_values)

    return mean_corr_values


# ---------------------------------------------------------------------
# Paths & file definitions
# ---------------------------------------------------------------------
base_path = searchlight_base_path()
partial_path = searchlight_partial_path()
subjects = [f"{i:03d}" for i in range(1, 21)]  # 001–020

base_files = {
    "aff": "aff_corr.nii.gz",
    "gist": "gist_corr.nii.gz",
    "obj": "obj_corr.nii.gz",
}

partial_files = {
    "object_gist": "object_gist_partialed_out.nii.gz",
    "gist_object": "gist_object_partialed_out.nii.gz",
    "affordance_gist": "affordance_gist_partialed_out.nii.gz",
    "gist_affordance": "gist_afforance_partialed_out.nii.gz",  # keep as-is to match disk
    "affordance_object": "affordance_object_partialed_out.nii.gz",
    "object_affordance": "object_affordance_partialed_out.nii.gz",
}

# Storage
all_data_base = {cond: [] for cond in base_files}
all_data_partial = {cond: [] for cond in partial_files}


# ---------------------------------------------------------------------
# Load data for each subject & condition
# ---------------------------------------------------------------------
for sub in subjects:
    base_folder = os.path.join(base_path, f"sub_{sub}_images")
    partial_folder = os.path.join(partial_path, f"sub_{sub}_images")

    # Base files
    for cond, filename in base_files.items():
        full_name = f"sub_{sub}_{filename}"
        img_path = os.path.join(base_folder, full_name)

        if not os.path.exists(img_path):
            print(f"Warning: Base file {img_path} not found for subject {sub}. Skipping.")
            continue

        img = load_img(img_path)
        data = compute_roi_means(img)
        all_data_base[cond].append(list(data.values()))

    # Partial files
    for cond, filename in partial_files.items():
        full_name = f"sub_{sub}_{filename}"
        img_path = os.path.join(partial_folder, full_name)

        if not os.path.exists(img_path):
            print(f"Warning: Partial file {img_path} not found for subject {sub}. Skipping.")
            continue

        img = load_img(img_path)
        data = compute_roi_means(img)
        all_data_partial[cond].append(list(data.values()))


# Convert lists to arrays
for cond in all_data_base:
    all_data_base[cond] = np.array(all_data_base[cond])

for cond in all_data_partial:
    all_data_partial[cond] = np.array(all_data_partial[cond])


# ---------------------------------------------------------------------
# Mean & SEM across subjects
# ---------------------------------------------------------------------
def compute_mean_sem(data_dict, label):
    mean_dict = {}
    sem_dict = {}
    for cond, arr in data_dict.items():
        if arr.size > 0:
            mean_dict[cond] = np.mean(arr, axis=0)
            sem_dict[cond] = sem(arr, axis=0)
        else:
            mean_dict[cond] = None
            sem_dict[cond] = None
            print(f"No data for {label} condition: {cond}")
    return mean_dict, sem_dict


mean_base, sem_base = compute_mean_sem(all_data_base, "base")
mean_partial, sem_partial = compute_mean_sem(all_data_partial, "partial")


# ---------------------------------------------------------------------
# Wilcoxon test against zero
# ---------------------------------------------------------------------
def test_against_zero(data_array, n_comparisons):
    """Wilcoxon signed-rank test against zero for each ROI (column)."""
    stat_values = []
    p_values = []
    sig = []

    alpha = 0.05 / n_comparisons

    for i in range(data_array.shape[1]):
        stat, p_val = wilcoxon(data_array[:, i], alternative="two-sided")
        stat_values.append(stat)
        p_values.append(p_val)
        sig.append(p_val < alpha)

    return np.array(stat_values), np.array(p_values), np.array(sig, dtype=bool)


# ---------------------------------------------------------------------
# Significance for each condition
# ---------------------------------------------------------------------
n_comparisons = 36

_, _, sig_aff_base = test_against_zero(all_data_base["aff"], n_comparisons)
_, _, sig_gist_base = test_against_zero(all_data_base["gist"], n_comparisons)
_, _, sig_obj_base = test_against_zero(all_data_base["obj"], n_comparisons)

_, _, sig_aff_gist = test_against_zero(all_data_partial["affordance_gist"], n_comparisons)
_, _, sig_aff_obj = test_against_zero(all_data_partial["affordance_object"], n_comparisons)
_, _, sig_gist_aff = test_against_zero(all_data_partial["gist_affordance"], n_comparisons)
_, _, sig_gist_obj = test_against_zero(all_data_partial["gist_object"], n_comparisons)
_, _, sig_obj_aff = test_against_zero(all_data_partial["object_affordance"], n_comparisons)
_, _, sig_obj_gist = test_against_zero(all_data_partial["object_gist"], n_comparisons)

conditions_to_plot = [
    ("aff", "Affordance Base",        mean_base["aff"],              sem_base["aff"],              "#780000"),
    ("affordance_gist", "Affordance-Gist Partial", mean_partial["affordance_gist"], sem_partial["affordance_gist"], "#d62828"),
    ("affordance_object", "Affordance-Object Partial", mean_partial["affordance_object"], sem_partial["affordance_object"], "#bf3c30"),
    ("gist", "GIST Base",             mean_base["gist"],             sem_base["gist"],             "#ff7b00"),
    ("gist_affordance", "Gist-Affordance Partial", mean_partial["gist_affordance"], sem_partial["gist_affordance"], "#ffaa00"),
    ("gist_object", "Gist-Object Partial", mean_partial["gist_object"], sem_partial["gist_object"], "#ffdd00"),
    ("obj", "Object Base",            mean_base["obj"],              sem_base["obj"],              "#023e8a"),
    ("object_affordance", "Object-Affordance Partial", mean_partial["object_affordance"], sem_partial["object_affordance"], "#0096c7"),
    ("object_gist", "Object-Gist Partial", mean_partial["object_gist"], sem_partial["object_gist"], "#90e0ef"),
]

sig_dict = {
    "aff": sig_aff_base,
    "gist": sig_gist_base,
    "obj": sig_obj_base,
    "affordance_gist": sig_aff_gist,
    "affordance_object": sig_aff_obj,
    "gist_affordance": sig_gist_aff,
    "gist_object": sig_gist_obj,
    "object_affordance": sig_obj_aff,
    "object_gist": sig_obj_gist,
}


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
x = np.arange(4)      # 4 ROIs: PPA, OPA, MPA, V1
width = 0.08
offsets = np.linspace(-4, 4, len(conditions_to_plot))

fig, ax = plt.subplots(figsize=(10, 10))

all_max_heights = []
positions_list = []

for i, (key, label, mean_vals, sem_vals, color) in enumerate(conditions_to_plot):
    if mean_vals is None:
        continue

    positions = x + offsets[i] * width
    positions_list.append(positions)

    ax.bar(
        positions,
        mean_vals,
        width=width,
        yerr=sem_vals,
        label=label,
        color=color,
        capsize=5,
        edgecolor="white",
    )

    current_max = np.max(mean_vals + sem_vals)
    all_max_heights.append(current_max)

max_height = max(all_max_heights) if all_max_heights else 0
asterisk_height = max_height + (0.05 * (max_height if max_height != 0 else 1))

# Add significance asterisks
for i, (key, label, mean_vals, sem_vals, color) in enumerate(conditions_to_plot):
    if mean_vals is None:
        continue

    sig_array = sig_dict[key]
    positions = x + offsets[i] * width
    for j, sig in enumerate(sig_array):
        if sig:
            ax.text(
                positions[j],
                asterisk_height,
                "*",
                ha="center",
                va="bottom",
                fontsize=14,
                color="black",
            )

# Cosmetics
ax.axhline(y=0, color="black", linestyle="-")
ax.set_ylabel("Mean Spearman's Correlation", fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(["PPA", "OPA", "MPA", "V1"], fontsize=15)
plt.yticks(fontsize=15)
ax.legend(bbox_to_anchor=(1.01, 1))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)

# ---------------------------------------------------------------------
# Save figure to ./Figures/ (create if necessary)
# ---------------------------------------------------------------------
# Get folder where this script lives (fallback to CWD in interactive use)
script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
figures_dir = os.path.join(script_dir, "Figures")
os.makedirs(figures_dir, exist_ok=True)

fig_path = os.path.join(figures_dir, "Searchlight_ROI_allspaces.svg")
plt.savefig(fig_path, bbox_inches="tight", transparent=True, dpi=300)

plt.show()
