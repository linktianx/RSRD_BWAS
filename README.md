# RSRD_BWAS

This repository provides the code supporting the manuscript entitled *"Spontaneous Brain Regional Dynamics Contribute to Generalizable Brain-Behavior Associations."* 

## Data/resources

The analysis uses preprocessed rs-fMRI data and behavioral measures from the following studies:
- **HCP-YA**: Establishment of primary findings
- **HCP-D / UK Biobank**: Validation in independent populations


## Code 

### Regional rs-fMRI BOLD Time-Series Feature Extraction
- **hctsa toolbox**: Used for extracting time-series features. For more information, see the [hctsa GitHub repository](https://github.com/benfulcher/hctsa) and the [hctsa manual](https://time-series-features.gitbook.io/hctsa-manual).

### refinedRSRD
- **ICC.py**: Identifies high-reliability features for analysis.
- **README.md**: Details the computation of 44 time-series features in refinedRSRD profiles, with links to the corresponding hctsa package functions.

### CCAanalysis
- **CCAHCP.py**: Analyzes brain-behavior associations in HCP-YA participants using Canonical Correlation Analysis (CCA).

### CCAgeneralization
- **PBS.py**: Computes a cumulative score reflecting externalizing problems and cognitive mode-related regional brain activity patterns in HCP-D and UK Biobank participants.
- **UKB_Externalizing_Problems.py**: Processes behavioral phenotypes related to externalizing problems for the UK Biobank cohort, following the guidelines of Aydogan et al., *Nat Hum Behav*, 5, 787–794 (2021).
