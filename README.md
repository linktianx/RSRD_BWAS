# RSRD_BWAS

This repository contains code supporting the manuscript
“Spontaneous Brain Regional Dynamics Contribute to Generalizable Brain–Behavior Associations.”

---

## Overview
The analyses were performed using preprocessed resting-state fMRI data and behavioral measures from the following datasets:
- **HCP-YA**: Primary analyses  
  (Refined RSRD features based on test–retest reliability; identification of two brain–behavior association modes)
- **HCP-D / MSC**: Reliability validation of refined RSRD features
- **HCP-D / UK Biobank**: Generalization of brain–behavior associations to external datasets
---

## Analysis
Scripts are organized into four subfolders that correspond to the order of analyses reported in the manuscript.

- [**01_RefinedRSRD**](./analysis/01_RefinedRSRD/):  
  Extraction, refinement, and description of RSRD profiles. The script `extract_hctsa_features.m` extracts RSRD features using the [hctsa toolbox](https://github.com/benfulcher/hctsa). Users may choose to compute either the full hctsa feature set (~7700 features) or the 44 refined features used in this study (as defined in `INP_ops_RSRD44.txt`). A demo using a small subset of time series is included. The script `ICC.py` identifies high-reliability features based on test–retest intraclass correlation coefficients (ICCs). For more details, see the subfolder’s `README.md`.

- [**02_CCAHCP**](./analysis/02_CCAHCP/):  
  Performs Canonical Correlation Analysis (CCA) to identify brain–behavior associations in HCP-YA participants. Input includes RSRD matrices (271 regions × 44 features) and 159 behavioral measures. A list of behavioral variables is provided in `hcp_behavior_list.csv`.

- [**03_CCAgeneralization**](./analysis/03_CCAgeneralization/):  
   Generalizes the CCA-based brain–behavior associations established in HCP-YA to independent datasets (HCP-D and UK Biobank).The script `latent_score.py` applies the pre-trained CCA pipeline to external RSRD matrices to generate individualized latent scores. The scripts `hcpd_validation.py` and `ukbb_validation.py` perform partial correlation analyses between latent scores and behavioral measures. Prior to the association analyses, these scripts also include procedures for identifying and preprocessing the target behavioral phenotypes from the respective databases.

- [**04_spatiotemporal_patterns**](./analysis/04_spatiotemporal_patterns/):  
  Cross-cohort analyses of the spatiotemporal patterns underlying each brain–behavior mode. Includes permutation testing based on spatial autocorrelation-preserving null models using the BrainSMASH framework.

---

## Requirements
Computation of RSRD features was performed using the **[hctsa toolbox](https://github.com/benfulcher/hctsa)** in **MATLAB 2020a**. For installation and usage, refer to the [official hctsa manual](https://time-series-features.gitbook.io/hctsa-manual).

All statistical analyses were conducted in **Python 3.6.13**.  
The following packages are required:
```
pandas==1.1.5
numpy==1.19.5
scipy==1.5.4
pingouin==0.3.12
scikit-learn==0.24.2
seaborn==0.9.0
altair==4.1.0
statsmodels==0.12.2
statannotations==0.6.0
brainsmash==0.11.0
```
You may also install these dependencies using the provided `requirements.txt` file
