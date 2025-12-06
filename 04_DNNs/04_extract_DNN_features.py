import os
from net2brain.feature_extraction import FeatureExtractor
from net2brain.rdm_creation import RDMCreator

import shutil

def delete_feature_dir(path):
    """Delete a feature directory, ignoring errors."""
    try:
        shutil.rmtree(path)
        print(f"Deleted feature directory: {path}")
    except FileNotFoundError:
        print(f"Feature directory not found (already deleted?): {path}")
    except OSError as e:
        print(f"Could not fully delete {path}: {e}")

# ---------------------------------------------------------------------
# Your original list of RDM folder names
# ---------------------------------------------------------------------
model_list = [
    "AlexNet_VISACT_RDM",
    "VGG16_VISACT_RDM",
    "ResNet50_VISACT_RDM",
    "Places365_VISACT_RDM",
    "SceneParsing_VISACT_RDM",
    "DINO_VISACT_RDM",
    "CLIP_RN101_VISACT_RDM",
    "x3d_m_VISACT_RDM",
    "slowfast_r101_VISACT_RDM",
    "vit_base_patch16_384__VISACT_90_RDM",
    "DINO_VIT_BASE_P16",
    "CLIP_ViT-B_-_16_VISACT_RDM",
    "CLIP_ViT-B_-_32_VISACT_90_RDM",
]

# ---------------------------------------------------------------------
# Paths relative to this script
# ---------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURE_ROOT = os.path.join(SCRIPT_DIR, "DNN_features")
RDM_ROOT = os.path.join(SCRIPT_DIR, "DNN_RDMS")

os.makedirs(FEATURE_ROOT, exist_ok=True)
os.makedirs(RDM_ROOT, exist_ok=True)

# VISACT 90-image stimulus set
stimuli_path = "/data/VISACT_post_fmri_prep/derivatives/analysis/fmri_image_set"

# ---------------------------------------------------------------------
# Map each RDM folder name to the underlying Net2Brain model + netset
# (adjust model/netset strings here if your local Net2Brain uses
#  slightly different identifiers)
# ---------------------------------------------------------------------
rdm_to_model_spec = {
    "AlexNet_VISACT_RDM": {
        "model": "AlexNet",
        "netset": "Standard",
    },
    "VGG16_VISACT_RDM": {
        "model": "vgg16",
        "netset": "Standard",
    },
    "ResNet50_VISACT_RDM": {
        "model": "resnet50",
        "netset": "Standard",
    },
    "Places365_VISACT_RDM": {
        # e.g., ResNet-50 trained on Places365; adjust if you use another backbone
        "model": "resnet50",
        "netset": "places365",  # <- check exact netset name in your Net2Brain
    },
    "SceneParsing_VISACT_RDM": {
        # e.g., ResNet-101 ADE20K; adjust if needed
        "model": "resnet101",
        "netset": "sceneparsing",  # <- check against Net2Brain docs
    },
    "DINO_VISACT_RDM": {
        # DINO convnet or ViT; choose the one you actually used
        "model": "dino_vitb16",  # <- example; change to your actual DINO model
        "netset": "dino",
    },
    "CLIP_RN101_VISACT_RDM": {
        "model": "RN101",   # typical CLIP ResNet-101 tag
        "netset": "clip",
    },
    "x3d_m_VISACT_RDM": {
        "model": "x3d_m",
        "netset": "video",  # or "pytorchvideo" depending on Net2Brain
    },
    "slowfast_r101_VISACT_RDM": {
        "model": "slowfast_r101",
        "netset": "video",  # or "pytorchvideo"/"slowfast"
    },
    "vit_base_patch16_384__VISACT_90_RDM": {
        "model": "vit_base_patch16_384",
        "netset": "timm",
    },
    "DINO_VIT_BASE_P16": {
        "model": "dino_vitb16",  # adjust to your actual DINO ViT-B/16 id
        "netset": "dino",
    },
    "CLIP_ViT-B_-_16_VISACT_RDM": {
        "model": "ViT-B_-_16",
        "netset": "clip",
    },
    "CLIP_ViT-B_-_32_VISACT_90_RDM": {
        "model": "ViT-B_-_32",
        "netset": "clip",
    },
}

# ---------------------------------------------------------------------
# Main loop: extract features and create RDMs for each model
# ---------------------------------------------------------------------
for rdm_name in model_list:

    spec = rdm_to_model_spec[rdm_name]
    model = spec["model"]
    netset = spec["netset"]

    # Where to save features and RDMs
    feat_path = os.path.join(FEATURE_ROOT, f"{model}_VISACT_Feat")
    save_path = os.path.join(RDM_ROOT, rdm_name)


    # 1) Extract features
    fx = FeatureExtractor(model=model, netset=netset, device="cpu")
    fx.extract(
        data_path=stimuli_path,   
        save_path=feat_path,
    )
    print(f"Finished extracting features for {model}")

    # 2) Create RDMs
    os.makedirs(save_path, exist_ok=True)
    creator = RDMCreator(
        verbose=True,
        device='cpu'
        )

    creator.create_rdms(
        feature_path=feat_path,
        save_path=save_path,
        save_format='npz'
        )
    
    print(f"Finished creating RDMs for {rdm_name} -> {save_path}")

    # 3) Optionally delete features to save space
    # Comment this out if you want to keep the feature files
    delete_feature_dir(feat_path)
    print(f"Deleted feature directory: {feat_path}")

    break
print("\nAll models processed.")
