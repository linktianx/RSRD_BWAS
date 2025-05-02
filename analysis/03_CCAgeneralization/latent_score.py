# ==============================================================================
# Description:
# This script applies a pre-trained CCA pipeline to extract latent brain–behavior 
# scores from RSRD features in the HCP-D and UK Biobank datasets.
# ==============================================================================

# Input:
# 1. RSRD_CCA_HCPYA_PIPE.npy — CCA model trained on HCP-YA (output from CCAHCP.py)
# 2. RSRD_Independent.npy — RSRD data from HCP-D and UKBB
#    - Shape: [n_subjects × n_ROIs × n_features]
#    - Must be reshaped to [n_subjects × (n_ROIs × n_features)]
#    - Feature definitions must match the training data exactly
#      (HCP-YA: 271 ROIs × 44 features = 11,924 features per subject)

# Output:
# Latent brain scores for HCP-D and UKBB participants
# ==============================================================================


import numpy as np
import pandas as pd
import seaborn as sns
import h5py

# --- latent score calculation ---
def extract_latent_scores(independent_rsrd, cca_pipeline):
    """
    Apply a pre-trained CCA pipeline to extract latent brain–behavior scores 
    from independent RSRD features.

    Parameters:
    - independent_rsrd : np.ndarray
        2D array of shape [n_subjects, n_roi * n_features]; flattened RSRD features.
    - cca_pipeline : dict
        Dictionary with the pre-trained CCA pipeline components:
            - 'StandardScaler': fitted scaler for normalization
            - 'PCA': fitted PCA model
            - 'CCA_x_rotations_': projection weights from trained CCA model
    Returns:
    - latent_scores : np.ndarray
        2D array [n_subjects × n_components] containing projected canonical scores.
    """
    # -------------------------------------------------------------
    # STEP a: Feature normalization using the fitted StandardScaler
    # -------------------------------------------------------------
    scaled_data = cca_pipeline['StandardScaler'].transform(independent_rsrd)

    # -------------------------------------------------------------
    # STEP b: Dimensionality reduction via PCA
    # -------------------------------------------------------------
    pca_transformed_data = cca_pipeline['PCA'].transform(scaled_data)

    # -------------------------------------------------------------
    # STEP c: Projection onto canonical space
    # -------------------------------------------------------------
    latent_scores = np.matmul(pca_transformed_data, cca_pipeline['CCA_x_rotations_'])
    # Return top 2 components corresponding to :
    #   mode1 (substance use mode,index 0)
    #   mode2 (cognition mode, index 1)
    return latent_scores[:, :2].copy()  

# ==============================================================================
# 1. Load Pre-trained CCA Pipeline for Brain-Behavior Projection (HCP-YA Derived)
# ==============================================================================
# Load the pre-trained CCA pipeline (trained on HCP-YA data) 
# The pipeline includes:
# - StandardScaler: for feature standardization
# - PCA: for dimensionality reduction to 100 components
# - CCA_x_rotations_: for projecting 2 identified canonical brain-behavior modes
RSRD_CCA_HCPYA_PIPE_PATH = '../02_CCAHCP/RSRD_CCA_HCPYA_PIPE.npy'
RSRD_CCA_HCPYA_PIPE = np.load(RSRD_CCA_HCPYA_PIPE_PATH,allow_pickle=True).tolist()

# ===================================================================================
# 2. Extract Latent Brain Scores from HCPD RSRD features Using Pre-trained CCA Pipeline
# ===================================================================================
#  Load RSRD data for HCP-D
with h5py.File(f'./RSRD_HCPD_271ROI_44FEA_4SESS.h5', 'r') as f:
    # Load the full 4D RSRD data array: shape = (subjects, ROIs, features, sessions)
    hcpd_rsrd_3d = f['MAT'][:]
    # Load subject IDs and decode byte strings into standard string format
    hcpd_subject_ids = f['SUBJ'][:].astype(str)
# Retrieve dimensions
nsubj, nrois, nfeatures, nsess = hcpd_rsrd_3d.shape
# Average RSRD features across all sessions per subject
# Resulting shape: (subjects, ROIs, features)
hcpd_rsrd_3d = hcpd_rsrd_3d.mean(axis=-1)
# Flatten the 3D array into 2D for input into CCA pipeline
# Resulting shape: (subjects, ROIs × features)
hcpd_rsrd = hcpd_rsrd_3d.reshape(nsubj, nrois * nfeatures)
# Apply the pre-trained pipeline to extract latent canonical scores
# Output shape: (subjects, canonical_dimensions)
hcpd_latent = extract_latent_scores(hcpd_rsrd, RSRD_CCA_HCPYA_PIPE)

# Saving
hcpd_latent_df = pd.DataFrame({
    'src_subject_id': hcpd_subject_ids,
    'm1s': hcpd_latent[:, 0],  # First CCA score
    'm2s': hcpd_latent[:, 1]   # Second CCA score
})
hcpd_latent_df.to_csv('./hcpd_latent_score.csv', index=False)

# ===================================================================================
# 3. Extract Latent Brain Scores from UKBB RSRD features Using Pre-trained CCA Pipeline
# ===================================================================================
# Load RSRD data for UKBB
with h5py.File(f'./RSRD_UKB_271ROI_44FEA_4SESS.h5', 'r') as f:
    ukb_rsrd = f['MAT'][:]  
    ukb_eids = f['SUBJ'][:].astype(str)
# Retrieve dimensions of the dataset
nsubj, nrois, nfeatures = ukb_rsrd.shape
ukb_rsrd_flattened = ukb_rsrd.reshape(nsubj, nrois * nfeatures)
# Apply the pre-trained pipeline to extract latent canonical scores
ukb_latent = extract_latent_scores(ukb_rsrd_flattened, RSRD_CCA_HCPYA_PIPE)
ukb_latent_df = pd.DataFrame({
    'eid': ukb_eids,
    'm1s': ukb_latent[:, 0],  
    'm2s': ukb_latent[:, 1] 
})
ukb_latent_df.to_csv(f'./ukbb_latent_score.csv', index=False)

