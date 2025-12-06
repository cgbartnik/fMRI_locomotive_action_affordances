# Locomotive Action Affordances

Code and scripts accompanying the analyses reported in:

**Representation of locomotive action affordances in human behavior, brains, and deep neural networks.**  
Bartnik, C. G., Sartzetaki, C., Puigseslloses Sanchez, A., Molenkamp, E., Bommer, S., Vuksic, N., & Groen, I. I. A. (2025). *Proceedings of the National Academy of Sciences (PNAS), 122*(24), e2414005122.  
[https://doi.org/10.1073/pnas.2414005122](https://doi.org/10.1073/pnas.2414005122)


This work investigates how the human visual system represents **locomotive action affordances**, such as walking, climbing, or swimming, when viewing complex real-world scenes. This study integrates **behavioral measurements**, **fMRI**, and **deep neural networks (DNNs)** to map the representational structure of action possibilities in naturalistic environments.

### Key findings
- Scene-selective regions, particularly the **occipital place area (OPA)** and **parahippocampal place area (PPA)**, encode locomotive affordances **independently of other visual and object features**.  
- These findings provide the **first clear evidence** for locomotive action affordance perception in the human brain.  
- Although modern deep neural networks capture certain aspects of affordance structure, they fall short of the **rich, multidimensional affordance geometry** measured in human behavior and brain responses.  
- Training DNNs on affordance labels or using affordance-focused language embeddings improves alignment but still fails to fully account for human perceptual organization.

If you encounter issues with the code, please open an issue or submit a pull request.

---

## Repository layout

- **`00_preprocessing/`** — fMRI preprocessing and ROI mask preparation.  
- **`01_behavioral_analysis/`** — behavioral affordance labels, t-SNE visualizations (Figure 2 in paper).  
- **`02_ROI_analysis/`** — region-of-interest (ROI) RSA (Figure 3).  
- **`03_Searchlight/`** — whole-brain searchlight analyses (Figure 4).  
- **`04_DNNs/`** — DNN/LLM feature extraction, DNN RSA (Figure 5).

---

## Requirements

- Python 3.8+ (tested with 3.10)
- Core scientific libraries:
  - `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`, `nilearn`, `statsmodels`
- Additional dependencies for DNN analyses:
  - `torch`, `timm`, or model-specific packages
  - `net2brain` (optional but recommended for unified DNN feature extraction)

  Data files used in this study are publicly available on OSF: [https://osf.io/v3rcq/overview](https://osf.io/v3rcq/overview)

### Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

