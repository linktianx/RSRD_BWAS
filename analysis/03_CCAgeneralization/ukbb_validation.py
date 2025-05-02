# =====================================================================================
# Description:
#
# This script performs UK Biobank (UKBB) generalization analyses, including behavioral phenotype
# precessing, and partial correlation analysis between latent scores and behavioral measures.

# Required Inputs:
# 1. Behavioral & Covariate phenotype data file ( ukb_database.csv and ukb_cf.csv) — to be replaced with the user’s local UKB dataset
# 2. Latent score file (e.g., ukb_latent_score.csv) 

# Outputs:
# 1. Processed UKBB behavioral data files for Mode 1 and Mode 2
# 2. Partial correlation results between latent scores and behavioral measures in the UKBB dataset 

# Note:
# All measures were defined using UK Biobank Data-Field identifiers, 
# accessible via the online data showcase: https://biobank.ndph.ox.ac.uk/.
# For detailed descriptions and source references of each phenotype, 
# please refer to the Supplementary Materials of the manuscript.
# =====================================================================================

# replace with your local path to the UKB database
ukb_database = '/localdata/data/dataset/UKBDataSet/ukb_database.csv' 


import os
import pandas as pd
import numpy as np
from functools import reduce
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pingouin as pg
from statsmodels.stats.multitest import multipletests

latent_score_path = f'./ukbb_latent_score.csv'
df_ukb_latent_score = pd.read_csv(latent_score_path)

# =====================================================================================
# --- Utility function ---
# =====================================================================================

def series_to_df(series, column_name='loading'):
    """Convert a pandas Series into a DataFrame with renamed columns."""
    df = series.to_frame(name=column_name)
    df = df.reset_index().rename(columns={'index': 'phenotype'})
    return df

# --- PCA function ---
def compute_pca_saveout(df, cols, n_pca=None, std=True, save_path=None):
    """Perform PCA on selected columns (cols) of a DataFrame (df)."""
    df_selected = df[cols]
    df_std = StandardScaler().fit_transform(df_selected) if std else df_selected.values
    pca = PCA(n_components=n_pca)
    df_std = np.nan_to_num(df_std)
    pca.fit(df_std)
    pc_values = pca.transform(df_std)
    df_pc = pd.DataFrame(df_std, columns=cols)
    pc_columns = [f'PC{i+1}' for i in range(pc_values.shape[1])]
    for i in range(pc_values.shape[1]):
        df_pc[pc_columns[i]] = pc_values[:, i]
    correlations_dict = {}
    ev_dict = {}
    for i in range(pc_values.shape[1]):
        correlations = df_pc.corr()[f'PC{i+1}'].drop([f'PC{j+1}' for j in range(pc_values.shape[1])])
        correlations_dict[f'PC{i+1}'] = correlations
        ev = pca.explained_variance_ratio_[i] * 100
        ev_dict[f'PC{i+1}'] = ev
    ev_array = np.array([ev_dict[pc] for pc in pc_columns])
    pc_loads = pd.concat([series_to_df(series).assign(PC=pc) for pc, series in correlations_dict.items()])
    evs_dict = dict(zip([f'PC{i+1}' for i in range(n_pca)], ev_array))
    pc_loads['EV'] = pc_loads['PC'].map(evs_dict)
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        np.save(f"{save_path}_correlations.npy", np.array([correlations_dict[pc].values for pc in pc_columns]))
        np.save(f"{save_path}_explained_variance.npy", ev_array)
        np.save(f"{save_path}_components.npy", pca.components_)
        np.save(f"{save_path}_feature_names.npy", np.array(cols))
        pc_loads.to_csv(f"{save_path}_PCA_loading_df.csv", index=False)
    return df_pc[pc_columns], correlations_dict, pca, cols, ev_array, pc_loads

# --- Partial correlation function ---
def perform_partial_correlation_analysis(df, y_list, x='m2s', cf_list=None,
                                         corr_method='spearman',
                                         pcc_method_list=['bonferroni','fdr_bh'],
                                         verbose=True):
    """
    Perform partial correlation analysis between a target variable `x` and a list of dependent variables `y_list`,
    controlling for covariates `cf_list`, with multiple testing correction.
    Parameters:
    - df: pd.DataFrame, input data containing x, y_list, and covariates
    - y_list: list of str, column names for dependent variables (to correlate with x)
    - x: str, column name for independent variable (default='m2s')
    - cf_list: list of str, covariate column names to control for (default=None)
    - corr_method: str, correlation method ('spearman', 'pearson', etc.)
    - pcc_method_list: method for p-value correction ('bonferroni', 'fdr_bh', etc.)
    - verbose: bool, whether to print progress and errors (default=True)

    Returns:
    - pd.DataFrame with partial correlation results and corrected p-values
    """
    if cf_list is None:
        cf_list = []
    corr_results = []
    valid_y_list = []
    if verbose:
        print(f"Processing correlation with x = '{x}', covariates = {cf_list}")
        print(f"Y variables: {y_list}")
    for y_col in y_list:
        try:
            # Drop rows with NaNs in the y variable
            valid_indices = df[y_col].dropna().index
            data_subset = df.loc[valid_indices, [x, y_col] + cf_list].dropna()
            result = pg.partial_corr(data=data_subset, x=x, y=y_col, covar=cf_list, method=corr_method)
            corr_results.append(result)
            valid_y_list.append(y_col)
        except Exception as e:
            if verbose:
                print(f"Error processing column '{y_col}': {e}")
    if not corr_results:
        if verbose:
            print("No valid results to compile.")
        return None
    try:
        df_results = pd.concat(corr_results, ignore_index=True)
        df_results.insert(0, 'Phenotypes', valid_y_list)
        # Multiple testing correction
        for pcc_method in pcc_method_list:
            corrected_pvals = multipletests(df_results['p-val'], method=pcc_method)[1]
            df_results.insert(5, f'p-{pcc_method}', corrected_pvals)
        return df_results
    except Exception as e:
        if verbose:
            print(f"Error compiling results: {e}")
        return None
    

    
# =====================================================================================
# Behavioral Phenotypes Processing: Mode 1
#    - Primary: A composite score representing substance use behavior,
#      including lifetime cannabis use, ever smoking, weekly alcohol intake, and addiction history
#    - Complementary: A broader externalizing problems score that additionally incorporates
#      risky driving, number of lifetime sexual partners, risk-taking tendencies, and irritability
# =====================================================================================

ukbb_mode1_field_id = [2149, 20453, 20116, 1249, 1239, 1568, 1578, 1588, 1598, 1608, 2040, 1100, 31, 20401, 1940]
ukbb_mode1_field_id_serach = [f"{i}-0.0" for i in ukbb_mode1_field_id]
df_behavior_mode1 = pd.read_csv(ukb_database, engine='c', usecols=['eid']+ukbb_mode1_field_id_serach, encoding="ISO-8859-1")
# -------------------------------------------------------------------------------------
# 1. Lifetime cannabis use: 20453-0.0
#    Question: "Have you taken cannabis (marijuana, grass, hash, etc.), even if it was a long time ago?"
# -------------------------------------------------------------------------------------
y = '20453-0.0'
df_cannabis = df_behavior_mode1[['eid', y]]
df_cannabis[y][df_cannabis[y] < 0] = np.nan

# -------------------------------------------------------------------------------------
# 2. Ever Smoking (20116-0.0, 1249-0.0, 1239-0.0)
#    Combined fields representing smoking behavior:
#    - 20116-0.0: Smoking status (current, former, or non-smoker)
#    - 1249-0.0: Past tobacco use frequency
#    - 1239-0.0: Current tobacco use habits
# -------------------------------------------------------------------------------------
smoke_ylist = [f'{i}-0.0' for i in ['20116', '1249', '1239']]
df_smoke_raw = df_behavior_mode1[['eid'] + smoke_ylist].dropna()
df_smoke_bin = df_smoke_raw.loc[(df_smoke_raw[smoke_ylist] >= 0).all(axis=1), ['eid'] + smoke_ylist]
df_smoke_bin['1249-0.0'] = df_smoke_bin['1249-0.0'].map({1: 1, 2: 1, 3: 1, 4: 0})
df_smoke_bin['1239-0.0'] = df_smoke_bin['1239-0.0'].map({0: 0, 1: 1, 2: 1})
df_smoke_bin['ever_smoking'] = (df_smoke_bin[df_smoke_bin.columns[1:]].sum(axis=1) > 0).astype(int)

# -------------------------------------------------------------------------------------
# 3. Weekly alcohol intake ( 1568-0.0, 1578-0.0, 1588-0.0, 1598-0.0, 1608-0.0)
#   weekly intake of various alcoholic beverages  (Sex-specific thresholds)
# -------------------------------------------------------------------------------------
week_drink_fields = [f'{i}-0.0' for i in [1568, 1578, 1588, 1598, 1608]]
SEX_label = '31-0.0'
pheno_data = df_behavior_mode1.loc[(df_behavior_mode1[week_drink_fields] >= 0).all(axis=1), ['eid'] + week_drink_fields]
pheno_data['total_weekly_intake'] = pheno_data[week_drink_fields].sum(axis=1)
pheno_data = pheno_data.merge(df_behavior_mode1[['eid', SEX_label]])
# Exclude heavy drinkers (thresholds from Daviet et al., 2022)
Data_fem = pheno_data[(pheno_data[SEX_label] == 0) & (pheno_data['total_weekly_intake'] < 18)]
Data_men = pheno_data[(pheno_data[SEX_label] == 1) & (pheno_data['total_weekly_intake'] < 24)]
df_alcohol = pd.concat([Data_fem, Data_men])
df_alcohol['Unstd_log_intake'] = np.log(df_alcohol['total_weekly_intake'] + 1)

# -------------------------------------------------------------------------------------
# 4. Ever addicted (20401-0.0)
# "Have you been addicted to or dependent on one or more things, 
# including substances (not cigarettes/coffee) or behaviours (such as gambling)?"
# -------------------------------------------------------------------------------------
df_ever_addicted = df_behavior_mode1[['eid', '20401-0.0']].replace({'20401-0.0': {-818: np.nan, -121: np.nan}}).dropna()

# -------------------------------------------------------------------------------------
# 5. 1100-0.0: Driving over the speed limit
#    "How often do you drive faster than the speed limit on the motorway?"
# -------------------------------------------------------------------------------------
dfspeed = df_behavior_mode1[['eid', '1100-0.0']]
dfspeed = dfspeed[(dfspeed['1100-0.0'] >= 0) & (dfspeed['1100-0.0'] < 5)]

# -------------------------------------------------------------------------------------
# 6. 1940-0.0: irritability
#    "Are you an irritable person?"
# -------------------------------------------------------------------------------------
df_general_irritability = df_behavior_mode1[['eid', '1940-0.0']].replace({'1940-0.0': {-1: np.nan, -3: np.nan}}).dropna()

# -------------------------------------------------------------------------------------
# 7. 2040-0.0: Risky behavior
#    "Would you describe yourself as someone who takes risks?"
# -------------------------------------------------------------------------------------
df_risky = df_behavior_mode1[['eid', '2040-0.0']].replace({'2040-0.0': {-1: np.nan, -3: np.nan}}).dropna()

# -------------------------------------------------------------------------------------
# 8. 2149-0.0: Number of lifetime sexual partners
#    "About how many sexual partners have you had in your lifetime?"
# -------------------------------------------------------------------------------------
dfsex = df_behavior_mode1[['eid', '2149-0.0']].replace({'2149-0.0': {-1: np.nan, -3: np.nan}}).dropna()

# --- Merge all variables ---
data_resume = reduce(lambda left, right: pd.merge(left, right, on='eid'), 
                     [dfspeed, dfsex, df_cannabis, df_smoke_bin, df_alcohol, df_risky, df_general_irritability, df_ever_addicted])
data_resume = data_resume.rename(columns={
    '20453-0.0': 'Lifetime_cannabis_use',
    '1100-0.0': 'Drive_speed',
    '2149-0.0': 'Lifetime_sex_partner_num',
    'Unstd_log_intake': 'weekly_alcohol_intake',
    '2040-0.0': 'Risk_taking',
    '1940-0.0': 'irritability',
    '20401-0.0': 'ever_addicted'
})

# --- Filter for individuals with CCA latent scores ---
data_resume = data_resume.loc[data_resume['eid'].isin(df_ukb_latent_score['eid'])].reset_index(drop=True)

# --- PCA on substance use ---
substance_use_items = ['Lifetime_cannabis_use', 'ever_smoking', 'weekly_alcohol_intake', 'ever_addicted']
df_data_sub = data_resume[['eid']+substance_use_items]
pca_scores_sub, _, _, _, _, _ = compute_pca_saveout(
    df=df_data_sub,
    cols=substance_use_items,
    n_pca=3,
    save_path='./ukbb_substance_PCA/'
)
df_with_pc_items_sub = pd.concat([df_data_sub, pca_scores_sub], axis=1)
df_with_pc_items_sub.to_csv('./ukbb_substance_PCA/_PCA_scores.csv', index=False)

# --- PCA on externalizing problems ---
externalizing_prob_items = [
    'Lifetime_cannabis_use', 'ever_smoking', 'weekly_alcohol_intake', 'ever_addicted',
    'Drive_speed', 'Lifetime_sex_partner_num', 'Risk_taking', 'irritability'
]
df_data_ext = data_resume[['eid']+externalizing_prob_items]
pca_scores_ext, _, _, _, _, _ = compute_pca_saveout(
    df=df_data_ext,
    cols=externalizing_prob_items,
    n_pca=5,
    save_path='./ukbb_externalizing_PCA/'
)
df_with_pc_items_ext = pd.concat([df_data_ext, pca_scores_ext], axis=1)
df_with_pc_items_ext.to_csv('./ukbb_externalizing_PCA/_PCA_scores.csv', index=False)


# =====================================================================================
# Behavioral Phenotype Processing: Mode 2 (Cognitive Traits)
#    - Five commonly used task-based measures of cognitive performance
# =====================================================================================
# field IDs for Mode 2 (cognitive measures)
mode2_field_ids =[ "4282-2.0", "20016-2.0", "6373-2.0", "21004-2.0", "23324-2.0"]
cognitive_field_map = {'20016-2.0': 'Fluid intelligence score', # Fluid intelligence/reasoning
                     '6373-2.0': 'Number of puzzles correctly solved', # Matrix pattern completion
                     '23324-2.0': 'Number of symbol digit matches made correctly', # Symbol digit substitution
                     '4282-2.0': 'Maximum digits remembered correctly', #Numeric memory
                     '21004-2.0': 'Number of puzzles correct'# Tower rearranging
                   }
# Load behavioral data for Mode 2
df_behavior_mode2 = pd.read_csv(
    ukb_database, 
    usecols=['eid'] + mode2_field_ids, 
    encoding="ISO-8859-1", 
    engine='c'
)
# Filter for subjects included in the latent score dataset & Save processed Mode 2 behavioral data
df_behavior_mode2 = df_behavior_mode2[df_behavior_mode2['eid'].isin(df_ukb_latent_score['eid'])].reset_index(drop=True)
df_behavior_mode2.rename(columns=cognitive_field_map, inplace=True)
df_behavior_mode2[list(cognitive_field_map.values())] = df_behavior_mode2[list(cognitive_field_map.values())].replace({-1: np.nan})
df_behavior_mode2.to_csv('./ukbb_mode2_behavior.csv', index=False)


# =====================================================================================
# CORRELATION ANALYSIS: Partial correlation between latent scores and behavioral measures
# =====================================================================================
# List of covariates to include in partial correlation analyses (to control for confounding effects)
cf_regress = [
    '31-0.0',         # Sex 
    '21003-2.0',      # Age 
    'age2',           # Age squared
    'age3',           # Age cubed
    '189-0.0',        # Socioeconomic status (SES) 
    '25010-2.0',      # Total brain volume 
    '25741-2.0',      # Mean head motion 
    '21001-2.0',      # Body Mass Index 
    '1707-0.0',       # Handedness
]

# Load covariate data
ukb_cf_df = pd.read_csv(f'./ukbb_cf.csv')
ukb_cf_df['age2'] = ukb_cf_df['21003-2.0'] ** 2
ukb_cf_df['age3'] = ukb_cf_df['21003-2.0'] ** 3
# Correlation method to be used in all analyses
corr_method = 'spearman'

# ------------------ Mode 1 ------------------
# Merge Mode data with covariates,latent scores
df_with_pc_items_sub_corr = df_with_pc_items_sub.merge(ukb_cf_df, on='eid').reset_index(drop=True)
df_with_pc_items_sub_corr = df_with_pc_items_sub_corr.merge(df_ukb_latent_score, on='eid').reset_index(drop=True)
df_with_pc_items_ext_corr = df_with_pc_items_ext.merge(ukb_cf_df, on='eid').reset_index(drop=True)
df_with_pc_items_ext_corr = df_with_pc_items_ext_corr.merge(df_ukb_latent_score, on='eid').reset_index(drop=True)

# Compute partial correlation between Mode 1 latent score and substance use PCA score (PC1)
mode1_ukb_res_sub = pg.partial_corr(
    data=df_with_pc_items_sub_corr,
    x='m1s',
    y='PC1',
    covar=list(set(cf_regress) - {'31-0.0'}),
    method=corr_method
)

# Compute partial correlation between Mode 1 latent score and externalizing PCA score (PC1)
mode1_ukb_res_ext = pg.partial_corr(
    data=df_with_pc_items_ext_corr,
    x='m1s',
    y='PC1',
    covar=list(set(cf_regress) - {'31-0.0'}),
    method=corr_method
)
# Save results to CSV
mode1_ukb_res_sub.to_csv('./ukbb_mode1_generalization_substance.csv', index=False)
mode1_ukb_res_ext.to_csv('./ukbb_mode1_generalization_externalizing.csv', index=False)

# ------------------ Mode 2 ------------------
# Merge Mode 2 behavioral scores with covariates, latent scores
df_behavior_mode2_corr = df_behavior_mode2.merge(ukb_cf_df, on='eid').reset_index(drop=True)
df_behavior_mode2_corr = df_behavior_mode2_corr.merge(df_ukb_latent_score, on='eid').reset_index(drop=True)

# Perform partial correlation between Mode 2 latent score and each cognitive measure
mode2_ukb_res = perform_partial_correlation_analysis(
    df_behavior_mode2_corr,
    x='m2s',
    cf_list=cf_regress,
    corr_method=corr_method,
    y_list=list(cognitive_field_map.values())
)
# Save results to CSV
mode2_ukb_res[['Phenotypes','n','r','p-val','p-bonferroni']].to_csv('./ukbb_mode2_generalization.csv', index=False)
