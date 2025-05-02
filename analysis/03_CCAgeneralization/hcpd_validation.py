# ==============================================================================
# Description:
# This script performs HCP-D generalization analyses, including behavioral data 
# processing and partial correlation analysis between latent scores and behavioral measures.
# ==============================================================================

# Inputs:
# 1. HCP-D database (e.g., HCPD_BASE_DIR) — replace with the user’s local HCP-D dataset
# 2. Latent scores (e.g., hcpd_latent_score.csv) — generated from latent_score.py

# Outputs:
# 1. Processed behavioral data files for Mode 1 and Mode 2 (HCP-D)
# 2. Partial correlation results between latent scores and behavioral phenotypes

# Note:
# Details of the behavioral variables and source files are provided in code comments,
# with reference to the NDA website.
# ==============================================================================


# Replace the path below with your local directory before use.
HCPD_BASE_DIR = '/share/data/dataset/HCPD/behavior_data/raw/'                    # <<< Replace with your local HCP-D data path (folder containing raw behavioral data (txt files))

import os
import numpy as np
import pandas as pd
import seaborn as sns
import altair as alt
import matplotlib as mpl
import pingouin as pg
from statsmodels.stats.multitest import multipletests

# =====================================================================================
# Behavioral Phenotypes Processing: Mode 1
#    - Includes variables from ASR (Adult Self-Report), CBCL (Child Behavior Checklist),
#      and Screentime surveys.
#    - Substance use behaviors from the PhenX survey were reviewed but excluded due to
#      limited sample size and data usability.
# =====================================================================================

# -------------------------------------------------------------------------------------
# Load ASR (Adult Self-Report) Data
# Reference: https://nda.nih.gov/data-structure/asr01
# -------------------------------------------------------------------------------------
asr_externalizing_list = [
    'adh_total', 'aggressive_behavior_total',
    'rule_breaking_behavior_total', 'externalizing_problems_total'
]
df_asr = pd.read_csv(f'{HCPD_BASE_DIR}/asr01.txt', sep='\t')
df_asr_take = df_asr[['src_subject_id'] + asr_externalizing_list].iloc[1:, :]
df_asr_take[asr_externalizing_list] = df_asr_take[asr_externalizing_list].apply(pd.to_numeric, errors='coerce')

# -------------------------------------------------------------------------------------
# Load CBCL (Child Behavior Checklist) Data
# Reference: https://nda.nih.gov/data-structure/cbcl01
# -------------------------------------------------------------------------------------
cbcl_externalizing_list = [
    'cbcl_rulebreak_raw', 'cbcl_adhd_raw',
    'cbcl_external_raw', 'cbcl_aggressive_raw'
]
df_cbcl = pd.read_csv(f'{HCPD_BASE_DIR}/cbcl01.txt', sep='\t')
df_cbcl_take = df_cbcl[['src_subject_id'] + cbcl_externalizing_list].iloc[1:, :]
df_cbcl_take[cbcl_externalizing_list] = df_cbcl_take[cbcl_externalizing_list].apply(pd.to_numeric, errors='coerce')

# -------------------------------------------------------------------------------------
# Load Screentime Data (m-rated games)
# Reference: https://nda.nih.gov/data-structure/screentime01
# -------------------------------------------------------------------------------------
mrated_games_list = ['screentime13_y']
df_screentime = pd.read_csv(f'{HCPD_BASE_DIR}/screentime01.txt', sep='\t')
df_screentime = df_screentime[df_screentime['comqother'] == 'subject about self'].reset_index(drop=True)
df_screentime_take = df_screentime[['src_subject_id'] + mrated_games_list]
df_screentime_take[mrated_games_list] = df_screentime_take[mrated_games_list].apply(pd.to_numeric, errors='coerce')

# -------------------------------------------------------------------------------------
# PhenX Substance Use Survey (checked but not used)
#    - Substance use behaviors were reviewed but excluded due to limited sample size
# Reference: https://nda.nih.gov/data-structure/phenx_su01
# -------------------------------------------------------------------------------------
df_sub = pd.read_csv(f'{HCPD_BASE_DIR}/phenx_su01.txt', sep='\t').iloc[1:, :]
SUB_behavior_list = [
    'alc_30day_freq', 'alc_30day_freqcode', 'alc_30day_quantity', 'alc_30day_quantitycode',
    'frquency_sedatives', 'frquency_sedatives_code', 'frquency_tranqulizers', 'frquency_tranquilzers_code',
    'frquency_painkillers', 'frquency_painkillers_code', 'frquency_stimulants', 'frquency_stimulants_code',
    'frquency_marijuana', 'frquency_marijuana_code', 'frquency_cocaine', 'frquency_cocaine_code',
    'frquency_crack', 'frquency_crack_code', 'frquency_hallucinogens', 'frquency_hallucinogens_code',
    'frquency_inhalant', 'frquency_inhalant_code', 'frquency_heroin', 'frquency_heroin_code',
    'frquency_other', 'frquency_other_code', 'cigarette_30day_smoking',
    'freq_30day_count', 'freq_30day_countcoded', 'freq_30day_estimate', 'quantity_30day'
]
# Count valid entries
y_counts = {'Item': [], 'Count': []}
for y in SUB_behavior_list:
    df_sub[y] = df_sub[y].astype(float)
    valid_count = df_sub[y].dropna().shape[0]
    y_counts['Item'].append(y)
    y_counts['Count'].append(valid_count)

y_counts_plot = pd.DataFrame(y_counts)
# Plot valid count statistics
chart_count = alt.Chart(y_counts_plot).mark_bar(size=18, color='grey').encode(
    x=alt.X('Item:N', sort=alt.EncodingSortField(field='Count', order='descending'),
            axis=alt.Axis(title='Phenotypes', labelAngle=-45)),
    y=alt.Y('Count:Q', axis=alt.Axis(title='Count'))
).properties(
    width=750, height=150,
    title='Statistics of Valid Data for Relevant Behavioral Phenotypes from the Phenx_su01 Scale'
)
chart_count.save(f'./phenx_count.html')

# =====================================================================================
# Summary of Behavioral Phenotypes applied in the Mode 1 generalization analysis
# =====================================================================================
# | Variable Name (key)              | Display Name                                | Source File         |
# |----------------------------------|---------------------------------------------|---------------------|
# | 'adh_total'                      | ASR:AD/H Problems                           | asr01.txt           |
# | 'aggressive_behavior_total'      | ASR:Aggressive Behavior                     | asr01.txt           |
# | 'rule_breaking_behavior_total'   | ASR:Rule Breaking Behavior                  | asr01.txt           |
# | 'externalizing_problems_total'   | ASR:Externalizing Problems                  | asr01.txt           |
# | 'cbcl_rulebreak_raw'             | CBCL:Rule Breaking Behavior                 | cbcl01.txt          |
# | 'cbcl_adhd_raw'                  | CBCL:AD/H Problems                          | cbcl01.txt          |
# | 'cbcl_external_raw'              | CBCL:Externalizing Problems                 | cbcl01.txt          |
# | 'cbcl_aggressive_raw'            | CBCL:Aggressive Behavior                    | cbcl01.txt          |
# | 'screentime13_y'                 | Frequency of playing M-rated games         | screentime01.txt    |
# Note: PhenX Substance Use Survey variables were reviewed but excluded due to limited sample size and data usability.

mode1_y_dict = dict(zip(
    [
        'adh_total', 'aggressive_behavior_total', 'rule_breaking_behavior_total', 'externalizing_problems_total',
        'cbcl_rulebreak_raw', 'cbcl_adhd_raw', 'cbcl_external_raw', 'cbcl_aggressive_raw', 'screentime13_y'
    ],
    [
        'ASR:AD/H Problems', 'ASR:Aggressive Behavior', 'ASR:Rule Breaking Behavior', 'ASR:Externalizing Problems',
        'CBCL:Rule Breaking Behavior', 'CBCL:AD/H Problems', 'CBCL:Externalizing Problems', 'CBCL:Aggressive Behavior',
        'Frequency of playing M-rated games'
    ]
))
# Merge all Mode 1 related measures
df_merged_mode1 = pd.merge(df_asr_take, df_cbcl_take, on='src_subject_id', how='outer')
df_merged_mode1 = pd.merge(df_merged_mode1, df_screentime_take, on='src_subject_id', how='outer')
df_merged_mode1 = df_merged_mode1.rename(columns=mode1_y_dict)
df_merged_mode1.to_csv(f'./hcpd_mode1_pheno_data.csv',index = False)

# =====================================================================================
# Behavioral Phenotypes Processing: Mode 2 
#    - Includes composite scores and task measures from the NIH Toolbox,
#      IQ measures (WAIS-IV and WISC-V)
# =====================================================================================

# -------------------------------------------------------------------------------------
# Cognition Composite Scores
# Reference: https://nda.nih.gov/data-structure/cogcomp01
# -------------------------------------------------------------------------------------
cogcomp01_list = [
    'nih_crycogcomp_unadjusted',     # Crystallized cognition composite score
    'nih_eccogcomp_unadjusted',      # Early childhood cognition composite
    'nih_fluidcogcomp_unadjusted',   # Fluid cognition composite score
    'nih_totalcogcomp_unadjusted'    # Total cognition composite score
]

df_cogcomp01 = pd.read_csv(f'{HCPD_BASE_DIR}/cogcomp01.txt', sep='\t')
df_cogcomp01_take = df_cogcomp01[['src_subject_id'] + cogcomp01_list].iloc[1:, :]
df_cogcomp01_take[cogcomp01_list] = df_cogcomp01_take[cogcomp01_list].apply(pd.to_numeric, errors='coerce')

# -------------------------------------------------------------------------------------
# Delay Discounting Task (PennCNP version)
# Reference: https://nda.nih.gov/data-structure/deldisk01
# -------------------------------------------------------------------------------------
deldisk01_list = ['auc_40000']  # Area under the curve (AUC) for delay discounting
df_deldisk01 = pd.read_csv(f'{HCPD_BASE_DIR}/deldisk01.txt', sep='\t')
df_deldisk01 = df_deldisk01[df_deldisk01['version_form'] == 'PennCNP version']
df_deldisk01_take = df_deldisk01[['src_subject_id'] + deldisk01_list].iloc[1:, :]
df_deldisk01_take[deldisk01_list] = df_deldisk01_take[deldisk01_list].apply(pd.to_numeric, errors='coerce')

# -------------------------------------------------------------------------------------
# Flanker Task
# Reference: https://nda.nih.gov/data-structure/flanker01
# -------------------------------------------------------------------------------------
df_flanker01 = pd.read_csv(f'{HCPD_BASE_DIR}/flanker01.txt', sep='\t')
df_flanker01_take = df_flanker01[['src_subject_id', 'nih_flanker_unadjusted']].iloc[1:, :]
df_flanker01_take['nih_flanker_unadjusted'] = df_flanker01_take['nih_flanker_unadjusted'].apply(pd.to_numeric, errors='coerce')

# -------------------------------------------------------------------------------------
# Dimensional Change Card Sort Test (DCCS)
# Reference: https://nda.nih.gov/data-structure/dccs01
# -------------------------------------------------------------------------------------
df_dccs01 = pd.read_csv(f'{HCPD_BASE_DIR}/dccs01.txt', sep='\t')
df_dccs01_take = df_dccs01[['src_subject_id', 'nih_dccs_unadjusted']].iloc[1:, :]
df_dccs01_take['nih_dccs_unadjusted'] = df_dccs01_take['nih_dccs_unadjusted'].apply(pd.to_numeric, errors='coerce')

# -------------------------------------------------------------------------------------
# Picture Sequence Memory Test (PSM)
# Reference: https://nda.nih.gov/data-structure/psm01
# -------------------------------------------------------------------------------------
df_psm01 = pd.read_csv(f'{HCPD_BASE_DIR}/psm01.txt', sep='\t')
df_psm01_take = df_psm01[['src_subject_id', 'nih_picseq_unadjusted']].iloc[1:, :]
df_psm01_take['nih_picseq_unadjusted'] = df_psm01_take['nih_picseq_unadjusted'].apply(pd.to_numeric, errors='coerce')

# -------------------------------------------------------------------------------------
# Oral Reading Recognition Test (ORRT)
# Reference: https://nda.nih.gov/data-structure/orrt01
# -------------------------------------------------------------------------------------
df_orrt01 = pd.read_csv(f'{HCPD_BASE_DIR}/orrt01.txt', sep='\t')
df_orrt01_take = df_orrt01[['src_subject_id', 'tbx_reading_score']].iloc[1:, :]
df_orrt01_take['tbx_reading_score'] = df_orrt01_take['tbx_reading_score'].apply(pd.to_numeric, errors='coerce')

# -------------------------------------------------------------------------------------
# Pattern Comparison Processing Speed Test (PCPS)
# Reference: https://nda.nih.gov/data-structure/pcps01
# -------------------------------------------------------------------------------------
df_pcps01 = pd.read_csv(f'{HCPD_BASE_DIR}/pcps01.txt', sep='\t')
df_pcps01_take = df_pcps01[['src_subject_id', 'nih_patterncomp_unadjusted']].iloc[1:, :]
df_pcps01_take['nih_patterncomp_unadjusted'] = df_pcps01_take['nih_patterncomp_unadjusted'].apply(pd.to_numeric, errors='coerce')

# -------------------------------------------------------------------------------------
# IQ Measures (WAIS-IV and WISC-V)
# References:
# https://nda.nih.gov/data-structure/wais_iv_part101
# https://nda.nih.gov/data-structure/wisc_v01
# -------------------------------------------------------------------------------------
# WAIS-IV Matrix Reasoning (ages 204–263 months)
df_iq1 = pd.read_csv(f'{HCPD_BASE_DIR}/wais_iv_part101.txt', sep='\t')
df_iq1 = pd.DataFrame(df_iq1.values[1:, :], columns=df_iq1.columns)  # Skip metadata row
df_iq1_take = df_iq1[['scoring_matrixreasoning_raw', 'src_subject_id']]
df_iq1_take = df_iq1_take.rename(columns={'scoring_matrixreasoning_raw': 'matrixreasoning'})

# WISC-V Matrix Reasoning (ages 73–203 months)
df_iq2 = pd.read_csv(f'{HCPD_BASE_DIR}/wisc_v01.txt', sep='\t')
df_iq2 = pd.DataFrame(df_iq2.values[1:, :], columns=df_iq2.columns)  # Skip metadata row
df_iq2_take = df_iq2[['matrixreason_score', 'src_subject_id']]
df_iq2_take = df_iq2_take.rename(columns={'matrixreason_score': 'matrixreasoning'})

# Combine IQ measures
df_iq = pd.concat([df_iq2_take, df_iq1_take], axis=0).reset_index(drop=True)
df_iq['matrixreasoning'] = df_iq['matrixreasoning'].apply(pd.to_numeric, errors='coerce')


# =====================================================================================
# Summary of Behavioral Phenotypes applied in the Mode 2 generalization analysis
# =====================================================================================
# | Variable Name (key)                   | Display Name                                              | Source File              |
# |--------------------------------------|-----------------------------------------------------------|--------------------------|
# | 'nih_crycogcomp_unadjusted'          | Crystallized Cognition Composite                          | cogcomp01.txt            |
# | 'nih_eccogcomp_unadjusted'           | Early Childhood Cognition Composite                       | cogcomp01.txt            |
# | 'nih_fluidcogcomp_unadjusted'        | Fluid Cognition Composite                                 | cogcomp01.txt            |
# | 'nih_totalcogcomp_unadjusted'        | Total Cognition Composite                                 | cogcomp01.txt            |
# | 'auc_40000'                          | Self-regulation                                           | deldisk01.txt (PennCNP)  |
# | 'nih_flanker_unadjusted'             | Executive Function/Inhibition                             | flanker01.txt            |
# | 'nih_dccs_unadjusted'                | Executive Function/Cognitive Flexibility                  | dccs01.txt               |
# | 'nih_picseq_unadjusted'              | Picture Sequence Memory                                   | psm01.txt                |
# | 'tbx_reading_score'                  | Oral Reading Recognition                                  | orrt01.txt               |
# | 'nih_patterncomp_unadjusted'         | Pattern Comparison Processing Speed                       | pcps01.txt               |
# | 'matrixreasoning'                    | Fluid Intelligence                                        | wais_iv_part101.txt + wisc_v01.txt |

mode2_y_dict = dict(zip(
    [
        'nih_crycogcomp_unadjusted', 'nih_eccogcomp_unadjusted', 'nih_fluidcogcomp_unadjusted', 'nih_totalcogcomp_unadjusted',
        'auc_40000', 'nih_flanker_unadjusted', 'nih_dccs_unadjusted', 'nih_picseq_unadjusted',
        'tbx_reading_score', 'nih_patterncomp_unadjusted', 'matrixreasoning'
    ],
    [
        'Crystallized Cognition Composite', 'Early Childhood Cognition Composite', 'Fluid Cognition Composite', 'Total Cognition Composite',
        'Self-regulation', 'Executive Function/Inhibition', 'Executive Function/Cognitive Flexibility', 'Picture Sequence Memory',
        'Oral Reading Recognition', 'Pattern Comparison Processing Speed', 'Fluid Intelligence'
    ]
))

# Merge all Mode 2 related measures
df_merged_mode2 = df_cogcomp01_take
for df in [df_deldisk01_take, df_flanker01_take, df_dccs01_take, 
           df_psm01_take, df_orrt01_take, df_pcps01_take, df_iq]:
    df_merged_mode2 = pd.merge(df_merged_mode2, df, on='src_subject_id', how='outer')
df_merged_mode2 = df_merged_mode2.rename(columns=mode2_y_dict)
df_merged_mode2.to_csv(f'./hcpd_mode2_pheno_data.csv',index = False)


# ====================================================================================
#  partial correlation between latent scores and behavioral measures
#  - Mode 1: m1s ~ ASR, CBCL, Screentime
#  - Mode 2: m2s ~ NIH Toolbox, IQ measures
# ====================================================================================
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
    
# latent score & confounds
hcpd_latent_df = pd.read_csv(f'./hcpd_latent_score.csv')
hcpd_cf = pd.read_csv(f'./hcpd_confounds.csv')

# Perform partial correlation analysis for Mode 1 (mode1 latent score ~ mode1 variables) 
df_merged_mode1_corr = hcpd_latent_df.merge(df_merged_mode1, on='src_subject_id').reset_index(drop=True)
df_merged_mode1_corr = df_merged_mode1_corr.merge(hcpd_cf, on='src_subject_id').reset_index(drop=True)
m1_res = perform_partial_correlation_analysis(df_merged_mode1_corr,x = 'm1s',
                                    cf_list = [ 'age', 'FD_4sess_mean'],
                                    corr_method='spearman',
                                    y_list = list(mode1_y_dict.values()))
m1_res[['Phenotypes','n','r','p-val','p-bonferroni']].to_csv('./hcpd_mode1_res.csv', index=False)

# Perform partial correlation analysis for Mode 2 (mode2 latent score ~ mode2 variables) 
df_merged_mode2_corr = hcpd_latent_df.merge(df_merged_mode2, on='src_subject_id').reset_index(drop=True)
df_merged_mode2_corr = df_merged_mode2_corr.merge(hcpd_cf, on='src_subject_id').reset_index(drop=True)
m2_res = perform_partial_correlation_analysis(df_merged_mode2_corr,x = 'm2s',
                                    cf_list = [ 'age', 'FD_4sess_mean','sex'],
                                    corr_method='spearman',
                                     y_list = list(mode2_y_dict.values()))
m2_res[['Phenotypes','n','r','p-val','p-bonferroni']].to_csv('./hcpd_mode2_res.csv', index=False)
