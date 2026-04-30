import sys
import os
import warnings

import nibabel as nib
import numpy as np
import pandas as pd
import pingouin as pg
from nilearn.image import new_img_like, load_img
from rsatoolbox.util.searchlight import get_volume_searchlight, get_searchlight_RDMs
from scipy.spatial.distance import squareform

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
    
from data_paths import bids_output_path, rdm_collection_path

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
fMRI_stim_ordering = [
    "indoor_0021", "indoor_0025", "indoor_0033", "indoor_0055",
    "indoor_0058", "indoor_0066", "indoor_0080", "indoor_0100",
    "indoor_0103", "indoor_0130", "indoor_0136", "indoor_0145",
    "indoor_0146", "indoor_0156", "indoor_0163", "indoor_0212",
    "indoor_0214", "indoor_0215", "indoor_0216", "indoor_0221",
    "indoor_0235", "indoor_0249", "indoor_0266", "indoor_0270",
    "indoor_0271", "indoor_0272", "indoor_0279", "indoor_0281",
    "indoor_0282", "indoor_0283", "outdoor_manmade_0015",
    "outdoor_manmade_0030", "outdoor_manmade_0032", "outdoor_manmade_0040",
    "outdoor_manmade_0063", "outdoor_manmade_0064", "outdoor_manmade_0068",
    "outdoor_manmade_0089", "outdoor_manmade_0110", "outdoor_manmade_0117",
    "outdoor_manmade_0119", "outdoor_manmade_0131", "outdoor_manmade_0133",
    "outdoor_manmade_0147", "outdoor_manmade_0148", "outdoor_manmade_0149",
    "outdoor_manmade_0152", "outdoor_manmade_0154", "outdoor_manmade_0155",
    "outdoor_manmade_0157", "outdoor_manmade_0161", "outdoor_manmade_0165",
    "outdoor_manmade_0167", "outdoor_manmade_0173", "outdoor_manmade_0175",
    "outdoor_manmade_0220", "outdoor_manmade_0256", "outdoor_manmade_0257",
    "outdoor_manmade_0258", "outdoor_manmade_0276", "outdoor_natural_0004",
    "outdoor_natural_0008", "outdoor_natural_0009", "outdoor_natural_0010",
    "outdoor_natural_0011", "outdoor_natural_0017", "outdoor_natural_0023",
    "outdoor_natural_0034", "outdoor_natural_0042", "outdoor_natural_0049",
    "outdoor_natural_0050", "outdoor_natural_0052", "outdoor_natural_0053",
    "outdoor_natural_0062", "outdoor_natural_0079", "outdoor_natural_0091",
    "outdoor_natural_0097", "outdoor_natural_0104", "outdoor_natural_0128",
    "outdoor_natural_0132", "outdoor_natural_0160", "outdoor_natural_0198",
    "outdoor_natural_0200", "outdoor_natural_0207", "outdoor_natural_0246",
    "outdoor_natural_0250", "outdoor_natural_0252", "outdoor_natural_0255",
    "outdoor_natural_0261", "outdoor_natural_0273",
]


# -----------------------------------------------------------------------------
# Utility: partial Spearman correlation (Pingouin)
# -----------------------------------------------------------------------------
def spearman_partial_correlation_pingouin(rdm1, rdm2, control_rdm):
    """
    Computes the Spearman partial correlation between two RDMs,
    controlling for a third RDM, using pingouin.

    Parameters
    ----------
    rdm1, rdm2, control_rdm : array-like
        1D vectors representing the RDMs.

    Returns
    -------
    float
        Partial Spearman correlation coefficient.
    """
    data = pd.DataFrame({
        "rdm1": rdm1,
        "rdm2": rdm2,
        "control_rdm": control_rdm,
    })

    # Pingouin's partial_corr returns a DataFrame; we extract the first 'r' value
    result = pg.partial_corr(
        data=data,
        x="rdm2",
        y="rdm1",
        covar="control_rdm",
        method="spearman"
    )
    return float(result["r"].iloc[0])


# -----------------------------------------------------------------------------
# Evaluate & save partial RDM per subject
# -----------------------------------------------------------------------------
def evaluate_and_save_partial_rdm_individually(
    subject_id,
    SL_RDM,
    rdm_key,
    target_rdm,
    control_rdm,
    save_path,
    mask,
    reference_img,
):
    """
    Evaluate searchlight partial correlations for a given RDM,
    controlling for a control RDM, and save resulting Nifti image.

    Parameters
    ----------
    subject_id : str
    SL_RDM : rsatoolbox RDM object
    rdm_key : str
        Key used in the output filename.
    target_rdm : ndarray
    control_rdm : ndarray
    save_path : str
        Directory where the image for this subject is saved.
    mask : ndarray
        3D boolean mask.
    reference_img : Nifti1Image
        Reference image for shape/affine/header.
    """
    vectors = SL_RDM.get_vectors()
    num_rdms = len(vectors)
    partial_corrs = np.empty(num_rdms)

    for i, rdm in enumerate(vectors):
        partial_corrs[i] = spearman_partial_correlation_pingouin(
            rdm, target_rdm, control_rdm
        )

    x, y, z = mask.shape
    voxel_indices = list(SL_RDM.rdm_descriptors["voxel_index"])

    brain_values = np.zeros(x * y * z)
    brain_values[voxel_indices] = partial_corrs
    brain_values = brain_values.reshape((x, y, z))

    result_img = new_img_like(reference_img, brain_values)

    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, f"sub_{subject_id}_{rdm_key}.nii.gz")
    nib.save(result_img, filename)
    print(f"Image for {rdm_key} (partial correlation) saved to {filename}")

    return result_img


# -----------------------------------------------------------------------------
# Main analysis setup
# -----------------------------------------------------------------------------
bidsroot = bids_output_path()
subjects = [f"{i:03d}" for i in range(1, 21)]  # 001–020

# --- Output directory: Partial_searchlight_data in the same folder as this script ---
script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
output_dir = os.path.join(script_dir, "Partial_searchlight_data")
os.makedirs(output_dir, exist_ok=True)

# Load RDMs
aff_RDM = squareform(
    np.load(
        rdm_collection_path("fMRI_Behavior", "action_fmri_90_rdm.npy")
    ).round(5)
)
obj_RDM = squareform(
    np.load(
        rdm_collection_path("fMRI_Behavior", "object_fmri_90_rdm.npy")
    ).round(5)
)
GIST_RDM = squareform(
    np.load(
        rdm_collection_path("GIST", "VISACT_fMRI", "GIST_1024_RDM_fMRI.npy")
    ).round(5)
)

# -----------------------------------------------------------------------------
# Loop over subjects
# -----------------------------------------------------------------------------
for sub in subjects:
    print(f"\nProcessing subject {sub}...")
    output_folder = os.path.join(output_dir, f"sub_{sub}_images")
    os.makedirs(output_folder, exist_ok=True)

    # --- Prepare fMRI data ---
    volumes = []
    for img_id in fMRI_stim_ordering:
        action_z_map = load_img(
            os.path.join(
                bidsroot,
                "derivatives",
                "analysis",
                "task_glm",
                f"sub-{sub}",
                f"sub-{sub}_task-action_{img_id}_hrf-glover_noise-ar1_fwhm-3_zmap.nii.gz",
            )
        )
        object_z_map = load_img(
            os.path.join(
                bidsroot,
                "derivatives",
                "analysis",
                "task_glm",
                f"sub-{sub}",
                f"sub-{sub}_task-fixation_{img_id}_hrf-glover_noise-ar1_fwhm-3_zmap.nii.gz",
            )
        )
        fixation_z_map = load_img(
            os.path.join(
                bidsroot,
                "derivatives",
                "analysis",
                "task_glm",
                f"sub-{sub}",
                f"sub-{sub}_task-object_{img_id}_hrf-glover_noise-ar1_fwhm-3_zmap.nii.gz",
            )
        )

        mean_volume = np.mean(
            np.stack(
                [
                    action_z_map.get_fdata(),
                    object_z_map.get_fdata(),
                    fixation_z_map.get_fdata(),
                ]
            ),
            axis=0,
        )
        volumes.append(mean_volume.flatten())

    data = np.array(volumes)

    # --- Mask (using action_z_map as reference) ---
    reference_data = action_z_map.get_fdata()
    masked_data = np.where(reference_data == 0.0, np.nan, reference_data)
    mask = ~np.isnan(masked_data)

    # --- Searchlight centers & neighbors ---
    centers, neighbors = get_volume_searchlight(mask, radius=5, threshold=0.5)

    # --- Searchlight RDMs ---
    image_value = np.arange(len(fMRI_stim_ordering))
    SL_RDM = get_searchlight_RDMs(
        data, centers, neighbors, image_value, method="correlation"
    )

    # --- Partial correlations and save ---
    evaluate_and_save_partial_rdm_individually(
        sub,
        SL_RDM,
        "gist_object_partialed_out",
        GIST_RDM,
        obj_RDM,
        output_folder,
        mask,
        action_z_map,
    )

    evaluate_and_save_partial_rdm_individually(
        sub,
        SL_RDM,
        "object_gist_partialed_out",
        obj_RDM,
        GIST_RDM,
        output_folder,
        mask,
        action_z_map,
    )




    evaluate_and_save_partial_rdm_individually(
        sub,
        SL_RDM,
        "affordance_gist_partialed_out",
        aff_RDM,
        GIST_RDM,
        output_folder,
        mask,
        action_z_map
        )


    evaluate_and_save_partial_rdm_individually(
        sub,
        SL_RDM,
        "affordance_object_partialed_out",
        aff_RDM,
        obj_RDM,
        output_folder,
        mask,
        action_z_map
        )



    evaluate_and_save_partial_rdm_individually(
        sub,
        SL_RDM,
        "object_affordance_partialed_out",
        obj_RDM,
        aff_RDM,
        output_folder,
        mask,
        action_z_map
        )


    evaluate_and_save_partial_rdm_individually(
        sub,
        SL_RDM,
        "gist_afforance_partialed_out",
        GIST_RDM,
        aff_RDM,
        output_folder,
        mask,
        action_z_map
        )
 

    
