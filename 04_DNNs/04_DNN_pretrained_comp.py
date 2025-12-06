import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, ttest_1samp
import scipy.stats as stats
from data_paths import behavior_path, rdm_collection_path, dnn_rdm_path


def compute_noise_ceilings(subject_rdms):
    """
    Compute the lower and upper noise ceilings using an optimized approach.

    Parameters:
    - subject_rdms: list of numpy arrays, each containing a participant's RDM
    - group_rdm: numpy array, the group-average RDM

    Returns:
    - lower_bound: float, the lower noise ceiling
    - upper_bound: float, the upper noise ceiling
    """
    # number of subjects 
    num_subjects = len(subject_rdms)
    # Initialize arrays to store the lower and upper bound correlations
    lower_bound_correlations = np.zeros(num_subjects)
    upper_bound_correlations = np.zeros(num_subjects)

    # Compute the group-average RDM
    group_rdm = np.mean(subject_rdms, axis=0)

    # Precompute the correlation between each participant's RDM and the group-average RDM (upper bound)
    group_correlations = [spearmanr(squareform(group_rdm), squareform(rdm)).correlation for rdm in subject_rdms]

    for i in range(num_subjects):
        # Leave-one-out approach: Exclude the current participant's RDM
        other_subjects_rdms = np.delete(subject_rdms, i, axis=0)

        other_group_rdm = np.mean(other_subjects_rdms, axis=0)
        # Calculate the correlation between the excluded participant's RDM and the average RDM of other participants
        lower_bound_correlation = np.mean([spearmanr(squareform(other_group_rdm), squareform(subject_rdms[i])).correlation])

        # Store the lower and upper bound correlations
        lower_bound_correlations[i] = lower_bound_correlation
        upper_bound_correlations[i] = group_correlations[i]

    # Compute the lower and upper noise ceilings
    lower_bound = np.mean(lower_bound_correlations)
    upper_bound = np.mean(upper_bound_correlations)

    return lower_bound, upper_bound
def get_highest_layer_correlation(model_path, sub_rdv):
    corrs = []
    layers = []
    p_values = []

    for layer in sorted(os.listdir(model_path)):
        if layer.endswith(".npz"):
            try:
                layer_array = np.load(os.path.join(model_path, layer))["arr_0"]
                layer_rdv = squareform(layer_array.round(5))
                spearman = spearmanr(sub_rdv, layer_rdv)
                corrs.append(spearman.correlation)
                layers.append(layer)
                p_values.append(spearman.pvalue)
            except:

                layer_rdv = np.load(os.path.join(model_path, layer))["rdm"]
                spearman = spearmanr(sub_rdv, layer_rdv)
                #print(spearman.correlation, layer)
                corrs.append(spearman.correlation)
                layers.append(layer)
                p_values.append(spearman.pvalue)


    #get index of nan vallues in list 
    nan_idx = [i for i, x in enumerate(corrs) if str(x) == 'nan']
    # remove nan values from list using index 


  
    corrs = np.delete(np.array(corrs), nan_idx)
    layers = np.delete(np.array(layers), nan_idx)
    p_values = np.delete(np.array(p_values), nan_idx)
    
    highest_layer_idx = np.argmax(corrs)
    highest_layer_correlation, highest_layer, p_value = corrs[highest_layer_idx], layers[highest_layer_idx], p_values[highest_layer_idx]

    return highest_layer_correlation, highest_layer, p_value

def compute_pairwise_comparisons(corr_action, base_height):
    import scipy.stats as stats

    # DataFrame to store test results
    test_result_df = pd.DataFrame(columns=["model1", "model2", "t-statistic", "p-value"])

    # List of unique models and subjects
    models = corr_action['model'].unique()

    # Perform pairwise t-tests for each subject

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model1 = models[i]
            model2 = models[j]
            # Get correlations for model1 and model2 for the current subject
            corr1 = corr_action[corr_action['model'] == model1]['correlation']
            corr2 = corr_action[corr_action['model'] == model2]['correlation']
            # Ensure there are enough data points to compare
            if len(corr1) > 0 and len(corr2) > 0:
                t_stat, p_val = stats.ttest_rel(corr1, corr2, nan_policy='omit')  # 'omit' handles cases where data may be missing
                test_result_df = test_result_df.append({"model1": model1, "model2": model2, "t-statistic": t_stat, "p-value": p_val}, ignore_index=True)

    num_tests = len(test_result_df)  # Count the number of tests
    alpha = 0.05  # Common significance level
    bonferroni_alpha = alpha / num_tests  # Corrected alpha level
    print(bonferroni_alpha)
    print(num_tests)

    # Adjusting p-values with Bonferroni correction
    test_result_df['adjusted p-value'] = test_result_df['p-value'] * num_tests
    test_result_df['adjusted p-value'] = test_result_df['adjusted p-value'].apply(lambda p: min(p, 1))  # Cap at 1 if greater

    # Adding column to show significance with Bonferroni correction
    test_result_df['significant (Bonferroni)'] = test_result_df['adjusted p-value'] < bonferroni_alpha

    test_result_df['pos_model1'] = test_result_df['model1'].map(x_positions)
    test_result_df['pos_model2'] = test_result_df['model2'].map(x_positions)

    # add a new col which starts at a base height of 0.5 and then adds 0.02 every row
    test_result_df['y_pos'] = base_height + 0.05 * test_result_df.groupby('model1').cumcount()


    return test_result_df
import os
import pandas as pd
from scipy.spatial.distance import squareform

#model_list = [ "AlexNet_VISACT_RDM",  "VGG16_VISACT_RDM", "ResNet50_VISACT_RDM", "Places365_VISACT_RDM", "SceneParsing_VISACT_RDM",
    #           "DINO_VISACT_RDM", "CLIP_RN101_VISACT_RDM", 'x3d_m_VISACT_RDM','slowfast_r101_VISACT_RDM',
     #           "DINO_VIT_BASE_P16", 'vit_base_patch16_384__VISACT_90_RDM', "CLIP_ViT-B_-_16_VISACT_RDM", "CLIP_ViT-B_-_32_VISACT_90_RDM"]

#RDM_path = "/home/clemens-uva/Github_repos/Visact_fMRI/code/DNN_RDMS/"


action_behavior_rdms = np.load(
    behavior_path("VISACT_fmri_behavior", "fmri_behavior_action_rdms.npz")
)["arr_0"]

#original_path = "/home/clemens-uva/Github_repos/Visact_fMRI/fMRI_folder/VISACT_RDM_collection/RdmFiles/fmri_clemens/affordances/pretrained_original"
original_path = dnn_rdm_path("Places365_VISACT_RDM")
untrained_path = rdm_collection_path("RdmFiles", "fmri_clemens", "affordances", "untrained_places")
pretrained_path = rdm_collection_path("vit_RDMs", "one_unfrozen_pretrained")
scratch_path = rdm_collection_path("RdmFiles", "fmri_clemens", "affordances", "scratch", "val_loss_relu")


model_list = [pretrained_path, scratch_path]
model_names = ["finetuned", "scratch"]


# create pandas dataframe
corr = pd.DataFrame(columns=["model", "behavior", "subject", "fold", "correlation", "layer", "p-value", "lower_nc", "higher_nc"])

behavior_name = "action"

beh_rdvs = [squareform(action_behavior_rdms[subj].round(5)) for subj in range(action_behavior_rdms.shape[0])]

lower_noise_ceiling, upper_noise_ceiling = compute_noise_ceilings(action_behavior_rdms)


for fold in ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"]:
    for subj, rdv in enumerate(beh_rdvs):
        sub_name = "sub_0"+str(subj+1)
        for idx, model in enumerate(model_list):
            model_name = model_names[idx]
            highest_layer_correlation, highest_layer, p_value = get_highest_layer_correlation(os.path.join(model, fold) + "/", rdv)
            corr = corr.append({"model": model_name, "behavior": behavior_name, "subject": sub_name, "fold": fold, 
                                "correlation": highest_layer_correlation, "layer": highest_layer, 
                                "p-value": p_value, "lower_nc": lower_noise_ceiling, "higher_nc":upper_noise_ceiling}, ignore_index=True)


model_list2 = [original_path, untrained_path]
model_names2 = ["original", "untrained"]
for subj, rdv in enumerate(beh_rdvs):
        sub_name = "sub_0"+str(subj+1)
        for idx, model in enumerate(model_list2):
            model_name = model_names2[idx]
            highest_layer_correlation, highest_layer, p_value = get_highest_layer_correlation(os.path.join(model) + "/", rdv)
            corr = corr.append({"model": model_name, "behavior": behavior_name, "subject": sub_name, "fold": "fold_0", 
                                "correlation": highest_layer_correlation, "layer": highest_layer, 
                                "p-value": p_value, "lower_nc": lower_noise_ceiling, "higher_nc":upper_noise_ceiling}, ignore_index=True)


corr_action = corr.copy()

corr_action = corr_action.groupby(['model', 'behavior', 'subject']).agg({
    'correlation': 'mean',
    'p-value': 'mean',
    'lower_nc': 'mean',
    'higher_nc': 'mean'
}).reset_index()


overview_df = pd.DataFrame(columns=["model", "behavior", "subject", "correlation", "sem", "p-value", "lower_nc", "upper_nc", "most_layer"])

for model in corr_action["model"].unique():
    model_name = model
    corr_subset = corr[corr["model"] == model_name]
    behavior_name = corr_subset["behavior"].iloc[0]
    #compute mean and standard error of the mean
    mean = np.mean(corr_subset["correlation"])
    # compute standard error of the mean
    sem = np.std(corr_subset["correlation"]) / np.sqrt(len(corr_subset["correlation"]))
    # test if different from zero
    t, p = ttest_1samp(corr_subset["correlation"], 0)
    #t, p = stats.wilcoxon(corr_subset["correlation"])
    # print result of t-test for each model
    print(f"{model_name} {t.round(2)} {p.round(4)}")
    # noise ceilings
    lower_nc = corr_subset["lower_nc"].iloc[0]
    upper_nc = corr_subset["higher_nc"].iloc[0]
    # check which layer occurs most often
    most_layer = corr_subset["layer"].value_counts().index[0]

    overview_df = overview_df.append({"model": model_name, "behavior": behavior_name, 
                                      "subject": "mean", "correlation": mean, "sem": sem,
                                        "p-value": p, "lower_nc": lower_nc, "upper_nc": upper_nc,
                                          "most_layer": most_layer}, ignore_index=True)

df = overview_df
df.index = df["model"]
df = df.loc[["original", "finetuned", "scratch", "untrained"]]


# Initialize x positions for bars based on model names in df
x_positions = {name: idx for idx, name in enumerate(df.index)}

# position for pairwise t-test results and asterix for significance
pairwise_test_pos = overview_df["upper_nc"].max() + 0.09

test_result_df = compute_pairwise_comparisons(corr_action, pairwise_test_pos)
print(test_result_df)

df = overview_df
df.index = df["model"]
df = df.loc[["original", "finetuned", "scratch", "untrained"]]

print(df)
# Define significance level (adjust as needed)
alpha = 0.05
alpha_bonferroni = alpha / len(df['model'])

# Create the figure and axis
fig, ax = plt.subplots(figsize=(5, 5),facecolor='white')


# Define color groups and labels

color_groups = [
    ('original', '#480ca8', [0]),
    ('finetuned', '#e10b16', [1]),
    ('scratch', '#ff7f51', [2]), 
    ('untrained', '#d4cbb3', [3])]

# Initialize x position for bars
x_positions = np.arange(len(df['model']))

# Plot bars for each color group
for group_label, color, indices in color_groups:
    group_df = df.iloc[indices]

    # Plot the bars with error bars
    bars = ax.bar(x_positions[indices], group_df['correlation'], yerr=group_df['sem'], capsize=5, color=color, label=group_label)

    # for index in indicies get the model name subset the action_df by model and plot scatter points for the individual correlations
    for index in indices:
        model_name = df.iloc[index]["model"]
        model_df = corr_action[corr_action["model"] == model_name]
        ax.scatter([x_positions[index]] * len(model_df["correlation"]), model_df["correlation"], color="gray", alpha=0.5, s=10, zorder=10)

    # Plot asterisks for significant p-values
    for i, p_value in enumerate(group_df['p-value']):
        if p_value < alpha_bonferroni:
            ax.text(x_positions[indices[i]], df['upper_nc'][0] + 0.02, '*', ha='center', fontsize=12)

# Plot dashed lines for lower and upper bounds of noise ceilings
ax.fill_between([-0.8, len(x_positions)], df['lower_nc'][0], df['upper_nc'][0], color='gray', alpha=0.2)
ax.axhline(y=df['upper_nc'][0], xmin=-0.8, xmax=len(x_positions), color='gray', linestyle='--', label='Upper Noise Ceiling', linewidth=1)
ax.axhline(y = df['lower_nc'][0], xmin=-0.8, xmax=len(x_positions), color='gray', linestyle='--', label='Lower Noise Ceiling', linewidth=1)
# Fill the are between the noise ceilings and start at -1



for index, row in test_result_df.iterrows():
        if row['significant (Bonferroni)'] == True:
            # Get positions for the two models
            pos1 = row['pos_model1']
            pos2 = row['pos_model2']
            # Draw a line between the bars at the calculated height
            ax.plot([pos1, pos2], [pairwise_test_pos, pairwise_test_pos], color="black", linestyle='-')
            pairwise_test_pos +=0.03

# Customize the plo
ax.set_ylabel("Spearman's rho Correlation", fontsize=15)
#ax.set_title('Navigational Affordance Behavior in fMRI')
ax.set_xticks(x_positions)
ax.set_xticklabels(["original", 
                "finetuned",
                   "scratch",
                   "untrained"], rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
ax.set_ylim(0, 1)
ax.set_xlim(-0.8, len(x_positions))
# remove the black frame
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Show the plot
plt.tight_layout()
script_root = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(script_root, "Figures")
os.makedirs(figures_dir, exist_ok=True)
plt.savefig(os.path.join(figures_dir, "ResNet_training_action.png"), dpi=300, transparent=True)



import os
import pandas as pd
from scipy.spatial.distance import squareform

#model_list = [ "AlexNet_VISACT_RDM",  "VGG16_VISACT_RDM", "ResNet50_VISACT_RDM", "Places365_VISACT_RDM", "SceneParsing_VISACT_RDM",
    #           "DINO_VISACT_RDM", "CLIP_RN101_VISACT_RDM", 'x3d_m_VISACT_RDM','slowfast_r101_VISACT_RDM',
     #           "DINO_VIT_BASE_P16", 'vit_base_patch16_384__VISACT_90_RDM', "CLIP_ViT-B_-_16_VISACT_RDM", "CLIP_ViT-B_-_32_VISACT_90_RDM"]

#RDM_path = "/home/clemens-uva/Github_repos/Visact_fMRI/code/DNN_RDMS/"


action_behavior_rdms = np.load(
    behavior_path("VISACT_fmri_behavior", "fmri_behavior_object_rdms.npz")
)["arr_0"]

#original_path = "/home/clemens-uva/Github_repos/Visact_fMRI/fMRI_folder/VISACT_RDM_collection/RdmFiles/fmri_clemens/affordances/pretrained_original"
original_path = dnn_rdm_path("Places365_VISACT_RDM")
untrained_path = rdm_collection_path("RdmFiles", "fmri_clemens", "affordances", "untrained_places")
pretrained_path = rdm_collection_path("vit_RDMs", "one_unfrozen_pretrained")
scratch_path = rdm_collection_path("RdmFiles", "fmri_clemens", "affordances", "scratch", "val_loss_relu")


model_list = [pretrained_path, scratch_path]
model_names = ["finetuned", "scratch"]


# create pandas dataframe
corr = pd.DataFrame(columns=["model", "behavior", "subject", "fold", "correlation", "layer", "p-value", "lower_nc", "higher_nc"])

behavior_name = "action"

beh_rdvs = [squareform(action_behavior_rdms[subj].round(5)) for subj in range(action_behavior_rdms.shape[0])]

lower_noise_ceiling, upper_noise_ceiling = compute_noise_ceilings(action_behavior_rdms)


for fold in ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"]:
    for subj, rdv in enumerate(beh_rdvs):
        sub_name = "sub_0"+str(subj+1)
        for idx, model in enumerate(model_list):
            model_name = model_names[idx]
            highest_layer_correlation, highest_layer, p_value = get_highest_layer_correlation(os.path.join(model, fold) + "/", rdv)
            corr = corr.append({"model": model_name, "behavior": behavior_name, "subject": sub_name, "fold": fold, 
                                "correlation": highest_layer_correlation, "layer": highest_layer, 
                                "p-value": p_value, "lower_nc": lower_noise_ceiling, "higher_nc":upper_noise_ceiling}, ignore_index=True)


model_list2 = [original_path, untrained_path]
model_names2 = ["original", "untrained"]
for subj, rdv in enumerate(beh_rdvs):
        sub_name = "sub_0"+str(subj+1)
        for idx, model in enumerate(model_list2):
            model_name = model_names2[idx]
            highest_layer_correlation, highest_layer, p_value = get_highest_layer_correlation(os.path.join(model) + "/", rdv)
            corr = corr.append({"model": model_name, "behavior": behavior_name, "subject": sub_name, "fold": "fold_0", 
                                "correlation": highest_layer_correlation, "layer": highest_layer, 
                                "p-value": p_value, "lower_nc": lower_noise_ceiling, "higher_nc":upper_noise_ceiling}, ignore_index=True)


corr_action = corr.copy()

corr_action = corr_action.groupby(['model', 'behavior', 'subject']).agg({
    'correlation': 'mean',
    'p-value': 'mean',
    'lower_nc': 'mean',
    'higher_nc': 'mean'
}).reset_index()


overview_df = pd.DataFrame(columns=["model", "behavior", "subject", "correlation", "sem", "p-value", "lower_nc", "upper_nc", "most_layer"])

for model in corr_action["model"].unique():
    model_name = model
    corr_subset = corr[corr["model"] == model_name]
    behavior_name = corr_subset["behavior"].iloc[0]
    #compute mean and standard error of the mean
    mean = np.mean(corr_subset["correlation"])
    # compute standard error of the mean
    sem = np.std(corr_subset["correlation"]) / np.sqrt(len(corr_subset["correlation"]))
    # test if different from zero
    t, p = ttest_1samp(corr_subset["correlation"], 0)
    #t, p = stats.wilcoxon(corr_subset["correlation"])
    # print result of t-test for each model
    print(f"{model_name} {t.round(2)} {p.round(4)}")
    # noise ceilings
    lower_nc = corr_subset["lower_nc"].iloc[0]
    upper_nc = corr_subset["higher_nc"].iloc[0]
    # check which layer occurs most often
    most_layer = corr_subset["layer"].value_counts().index[0]

    overview_df = overview_df.append({"model": model_name, "behavior": behavior_name, 
                                      "subject": "mean", "correlation": mean, "sem": sem,
                                        "p-value": p, "lower_nc": lower_nc, "upper_nc": upper_nc,
                                          "most_layer": most_layer}, ignore_index=True)

df = overview_df
df.index = df["model"]
df = df.loc[["original", "finetuned", "scratch", "untrained"]]


# Initialize x positions for bars based on model names in df
x_positions = {name: idx for idx, name in enumerate(df.index)}

# position for pairwise t-test results and asterix for significance
pairwise_test_pos = overview_df["upper_nc"].max() + 0.09

test_result_df = compute_pairwise_comparisons(corr_action, pairwise_test_pos)

df = overview_df
df.index = df["model"]
df = df.loc[["original", "finetuned", "scratch", "untrained"]]

print(df)
# Define significance level (adjust as needed)
alpha = 0.05
alpha_bonferroni = alpha / len(df['model'])

# Create the figure and axis
fig, ax = plt.subplots(figsize=(5, 5),facecolor='white')


# Define color groups and labels

color_groups = [
    ('original', '#480ca8', [0]),
    ('finetuned', '#0077b6', [1]),
    ('scratch', '#90e0ef', [2]), 
    ('untrained', '#d4cbb3', [3])]

# Initialize x position for bars
x_positions = np.arange(len(df['model']))

# Plot bars for each color group
for group_label, color, indices in color_groups:
    group_df = df.iloc[indices]

    # Plot the bars with error bars
    bars = ax.bar(x_positions[indices], group_df['correlation'], yerr=group_df['sem'], capsize=5, color=color, label=group_label)

    # for index in indicies get the model name subset the action_df by model and plot scatter points for the individual correlations
    for index in indices:
        model_name = df.iloc[index]["model"]
        model_df = corr_action[corr_action["model"] == model_name]
        ax.scatter([x_positions[index]] * len(model_df["correlation"]), model_df["correlation"], color="gray", alpha=0.5, s=10, zorder=10)

    # Plot asterisks for significant p-values
    for i, p_value in enumerate(group_df['p-value']):
        if p_value < alpha_bonferroni:
            ax.text(x_positions[indices[i]], df['upper_nc'][0] + 0.02, '*', ha='center', fontsize=12)

# Plot dashed lines for lower and upper bounds of noise ceilings
ax.fill_between([-0.8, len(x_positions)], df['lower_nc'][0], df['upper_nc'][0], color='gray', alpha=0.2)
ax.axhline(y=df['upper_nc'][0], xmin=-0.8, xmax=len(x_positions), color='gray', linestyle='--', label='Upper Noise Ceiling', linewidth=1)
ax.axhline(y = df['lower_nc'][0], xmin=-0.8, xmax=len(x_positions), color='gray', linestyle='--', label='Lower Noise Ceiling', linewidth=1)
# Fill the are between the noise ceilings and start at -1



for index, row in test_result_df.iterrows():
        if row['significant (Bonferroni)'] == True:
            # Get positions for the two models
            pos1 = row['pos_model1']
            pos2 = row['pos_model2']
            # Draw a line between the bars at the calculated height
            ax.plot([pos1, pos2], [pairwise_test_pos, pairwise_test_pos], color="black", linestyle='-')
            pairwise_test_pos +=0.03

# Customize the plo
ax.set_ylabel("Spearman's rho Correlation", fontsize=15)
#ax.set_title('Object Behavior in fMRI')
ax.set_xticks(x_positions)
ax.set_xticklabels(["original", 
                "finetuned",
                   "scratch",
                   "untrained"], rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
ax.set_ylim(0, 1)
ax.set_xlim(-0.8, len(x_positions))
# remove the black frame
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Show the plot
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "ResNet_training_object.png"), dpi=300, transparent=True)


# Example usage (replace with actual inputs)
model_name = "AlexNet_VISACT_RDM"
layer_order_example = [
        "features.0.npz", "features.3.npz", "features.6.npz",
        "features.8.npz", "features.10.npz"
    ]
bar_color_example = "#7209b7"
x_tick_labels = [
        "features0", "features3", "features6",
        "features8", "features10"
    ]
save_path = "AlexNet_V1.svg"
plot_model_correlation(model_name, layer_order_example, bar_color_example, V1_mean_RDM, save_path,  20, x_tick_labels)
