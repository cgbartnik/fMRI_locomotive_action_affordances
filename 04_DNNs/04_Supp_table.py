#Supplementary Tables
import os
import pandas as pd

from .data_paths import get_dnn_rdm_root


model_list = [ "AlexNet_VISACT_RDM",  "VGG16_VISACT_RDM", "ResNet50_VISACT_RDM", "Places365_VISACT_RDM", "SceneParsing_VISACT_RDM",
               "DINO_VISACT_RDM", "CLIP_RN101_VISACT_RDM", 'x3d_m_VISACT_RDM','slowfast_r101_VISACT_RDM',
                "DINO_VIT_BASE_P16", 'vit_base_patch16_384__VISACT_90_RDM', "CLIP_ViT-B_-_16_VISACT_RDM", "CLIP_ViT-B_-_32_VISACT_90_RDM"]

names = ["AlexNet (ImageNet)", "VGG16 (ImageNet)", "ResNet50 (ImageNet)", "ResNet50 (Places365)", "ResNet50 (SceneParsing)", "DINO ResNet50 (ImageNet)", "CLIP ResNet101", "X3D-m", "Slowfast ResNet101", "DINO ViT Base Patch 16", "ViT Base Patch 16", "CLIP ViT-B 16", "CLIP ViT-B 32"]
RDM_path = get_dnn_rdm_root()
script_dir = os.path.dirname(os.path.abspath(__file__))
output_csv = os.path.join(script_dir, "supp_table_layers.csv")

all_layers = []
for model in model_list:
    layer_list = []
    for l in os.listdir(os.path.join(RDM_path, model)):
        if l.endswith("npz"):
            layer_list.append(l)

    all_layers.append(layer_list)

supp_dataframe = pd.DataFrame(columns=["names", "layer"], data = list(zip(names, all_layers)))

# save the dataframe as csv
supp_dataframe.to_csv(output_csv, index=False)
print(f"Saved supplementary table to {output_csv}")
