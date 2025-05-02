# =========================================
# Description:
# This script performs data preprocessing and Canonical Correlation Analysis (CCA)
# to identify associations between refined RSRD features and behavioral measures
# in the HCP-YA dataset.
#
# -----------------------------------------
# Input:
#   - RSRD features (3D NumPy array): shape [n_subjects × n_ROIs × n_features]
#   - Behavioral data (CSV): behavioral measures for HCP-YA participants
#   - HCP-YA behavior info / categories: sourced from HCPwiki (https://wiki.humanconnectome.org/)
#   - HCP-YA Family IDs (CSV): used for multi-level block permutation
#   - Confound variables (CSV): including age, motion, and blood-derived measures
#
# Output:
#   - CCA results
#   - Fitted pipeline: used for downstream generalization analysis
# =========================================


import pandas as pd
import numpy as np
import warnings
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA as skCCA
from scipy import stats
import statsmodels.api as sm
from tqdm import tqdm
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Configurable file paths (update to user local paths)
# ------------------------------------------------------------
HCP_INFO_PATH = '/share/data/dataset/hcp_info'   # Path to  user local demographic and behavioral data

def load_csv(file_path):
    """Utility function to load CSV files with file existence check."""
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"File not found: {file_path}")

def generate_data(base_csv):
    """Load and merge subject data with additional blood and behavioral data."""
    if isinstance(base_csv, str):
        subjects_csv = load_csv(base_csv).merge(load_csv(f'{HCP_INFO_PATH}/hcp/blood.csv'))
    elif isinstance(base_csv, pd.DataFrame):
        subjects_csv = base_csv.merge(load_csv(f'{HCP_INFO_PATH}/hcp/blood.csv'))
    
    subjects_csv = subjects_csv.merge(load_csv(f'{HCP_INFO_PATH}/HCP_behavior.csv'))
    subjects_csv = subjects_csv.merge(load_csv(f'{HCP_INFO_PATH}/RESTRICTED_s1200.csv'))
    return subjects_csv

def multi_level_block_permutation_global_shuffle(Y, family_id):
    """Conduct multi-level block permutation based on family structure."""
    family_series = pd.Series(family_id)
    Y_permuted = np.empty_like(Y)
    unique_family_ids = family_series.value_counts()[family_series.value_counts() == 1].index

    for family in family_series.unique():
        family_indices = family_series[family_series == family].index
        if family not in unique_family_ids:
            Y_family = Y[family_indices, :]
            np.random.shuffle(Y_family)
            Y_permuted[family_indices, :] = Y_family

    unique_family_indices = family_series[family_series.isin(unique_family_ids)].index
    Y_unique = Y[unique_family_indices, :]
    np.random.shuffle(Y_unique)
    Y_permuted[unique_family_indices, :] = Y_unique
    return Y_permuted

def get_feature_behavior_corr(feature_mat, beh_vec, nroi=271, p_thres=0.05):
    """Calculate Spearman correlation between feature matrix columns and behavior vector."""
    res_corr = [
        stats.spearmanr(feature_mat[:, i], beh_vec) for i in range(nroi)
    ]
    corrdf = pd.DataFrame(res_corr, columns=['r', 'p'])
    corrdf['r_sig'] = np.where(corrdf['p'] < p_thres, corrdf['r'], 0)
    return corrdf

def normalize_data(data):
    """Min-max normalization of data."""
    data_range = np.max(data) - np.min(data)
    return (data - np.min(data)) / data_range if data_range else data

def clean_confounds(cfdf):
    """Preprocess confounds data by replacing NaNs and standardizing."""
    cf_data = np.nan_to_num(cfdf.values)
    return StandardScaler().fit_transform(cf_data)

def get_residual(data_std, cf_data_std):
    """Perform OLS regression to get residuals after controlling for confounds."""
    return sm.OLS(data_std, cf_data_std).fit().resid

def pca_input(data, n_components=100):
    """Apply PCA to the data and return the transformed values and PCA model."""
    pca_model = PCA(n_components=n_components)
    val_pca = pca_model.fit_transform(data)
    print(f'{n_components} PCs: Explained variance = {pca_model.explained_variance_ratio_.sum():.2f}')
    return val_pca, pca_model

def clean_input_xy(y_df, x, behavior_list):
    """Standardize X and Y matrices for CCA analysis, returning models."""
    x_std_model = StandardScaler()
    xdata_std = x_std_model.fit_transform(x)
    
    y_std_model = StandardScaler()
    ydata_std = y_std_model.fit_transform(y_df[behavior_list].values)
    return xdata_std, ydata_std, x_std_model, y_std_model

def preproc_CCA(Xinp, Yinpdf, Cfdf, behavior_list, cf_list):
    """
    Preprocess input matrices for CCA:
      - Confound regression
      - PCA on residuals
      - Standardization

    Parameters:
        Xinp (np.ndarray): RSRD features [n_subjects × n_ROIs × n_features]
        Yinpdf (pd.DataFrame): Behavioral data
        Cfdf (pd.DataFrame): Confound variables
        behavior_list (list): Target behavioral measures
        cf_list (list): List of confounds to regress out

    Returns:
        dict: Dictionary containing PCA inputs, models, residuals, etc.
    """
    nsub, nroi, nf = Xinp.shape
    Xinp2d = Xinp.reshape(nsub, nroi * nf)
    xclean, yclean, x_std_model, _ = clean_input_xy(Yinpdf, Xinp2d, behavior_list)
    cf_data_std = clean_confounds(Cfdf[cf_list])

    y_residual = get_residual(yclean, cf_data_std)
    x_residual = get_residual(xclean, cf_data_std)
    
    x_pc, x_pca_model = pca_input(x_residual)
    y_pc, _ = pca_input(y_residual)

    return {
        'xpca': x_pc, 
        'ypca': y_pc,
        'x_std_model': x_std_model,
        'x_pca_model': x_pca_model,
        'xclean': xclean, 
        'yclean': yclean,
        'nfeature': nf,
        'ydf': Yinpdf
    }

def cca_analysis(inpdata, n_components=10, permutation=False):
    """Conduct CCA analysis with or without permutation."""
    x_pca = inpdata['xpca']
    y_pca = inpdata['ypca_perm'] if permutation else inpdata['ypca']
    
    cca = skCCA(n_components=n_components)
    cca.fit(x_pca, y_pca)
    
    # Avoid duplicate np.dot calculations
    covmat = np.dot(cca.x_scores_.T, cca.y_scores_) / (cca.x_scores_.shape[0] - 1)
    varE = np.diag(covmat)**2 / np.sum(np.diag(covmat)**2)
    s = np.corrcoef(cca.x_scores_.T, cca.y_scores_.T).diagonal(offset=cca.n_components)
    
    return {
        'model': cca,
        's': s,
        'a': cca.x_weights_,
        'b': cca.y_weights_,
        'varE': varE
    }

# ----------------------------------------------------------------------------
# Load data: behavior, confounds, RSRD features, behavioral variable list
# ----------------------------------------------------------------------------
df_behavior = load_csv(f'./774subject_behavior_data.csv')
cfdf = load_csv(f'./774_subject_confunds_final.csv')
cf_list = ['age', 'BPDiastolic', 'BPSystolic', 'HbA1C', 'Acquisitionint', 'fd_mean']
cfdf['Family_ID'] = cfdf.groupby(['Mother_ID', 'Father_ID']).ngroup()
RSRD_data = np.load(f'./CCA_fmat_X.npy')
measures = load_csv(f'./hcp_behavior_list.csv')['variable'].values.tolist()

# ----------------------------------------------------------------------------
# Run CCA analysis with permutation
# ----------------------------------------------------------------------------
cca_input_dict = preproc_CCA(RSRD_data, df_behavior, cfdf, measures, cf_list)
permu_corr, permu_varE = [], []
Niter = 10000
for _ in tqdm(range(Niter)):
    cca_input_dict['ypca_perm'] = multi_level_block_permutation_global_shuffle(cca_input_dict['ypca'], cfdf['Family_ID'].values)
    cca_res = cca_analysis(cca_input_dict, permutation=True)
    permu_corr.append(cca_res['s'])
    permu_varE.append(cca_res['varE'])
np.save(f'{Niter}iter_per_family_corr', permu_corr)
np.save(f'{Niter}iter_per_family_varE', permu_varE)

# ----------------------------------------------------------------------------
# Save  CCA results
# ----------------------------------------------------------------------------
# Run observed (non-permuted) CCA analysis
cca_res = cca_analysis(cca_input_dict, permutation=False)
# Save results: CCA outputs and preprocessed input dictionary
np.save('./CCA_res', cca_res)
np.save('./CCA_input', cca_input_dict)

# ----------------------------------------------------------------------------
# Extract and save the fitted CCA processing pipeline
# This includes:
#   - StandardScaler used for feature normalization
#   - PCA model used to reduce residualized features
#   - CCA projection matrix (CCA projection weights for new data)
# This object used for downstream generalization analyses
# ----------------------------------------------------------------------------
RSRD_CCA_HCPYA_PIPE = {
    'StandardScaler': cca_input_dict['x_std_model'],
    'PCA': cca_input_dict['x_pca_model'], 
    'CCA_x_rotations_': cca_res['model'].x_rotations_  # CCA projection weights (used to project new data into the canonical space)
}
np.save('RSRD_CCA_HCPYA_PIPE', RSRD_CCA_HCPYA_PIPE)
