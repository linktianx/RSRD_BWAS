# Description: This script use PCA to extract the "UKB externalizing problems"
# =========================================
# Field ID Information Summary:
# =========================================
# 1. 20453-0.0: Lifetime cannabis use
#    Question: "Have you taken cannabis (marijuana, grass, hash, etc.), even if it was a long time ago?"
#    https://biobank.ndph.ox.ac.uk/ukb/field.cgi?id=20453

# 2. Ever Smoking (20116-0.0, 1249-0.0, 1239-0.0)
#    Combined fields representing smoking behavior:
#    - 20116-0.0: Smoking status (current, former, or non-smoker)
#    - 1249-0.0: Past tobacco use frequency
#    - 1239-0.0: Current tobacco use habits
#    https://biobank.ndph.ox.ac.uk/ukb/field.cgi?id=20116
#    https://biobank.ndph.ox.ac.uk/ukb/field.cgi?id=1249
#    https://biobank.ndph.ox.ac.uk/ukb/field.cgi?id=1239

# 3. 1100-0.0: Driving over the speed limit
#    "How often do you drive faster than the speed limit on the motorway?"
#    https://biobank.ndph.ox.ac.uk/ukb/field.cgi?id=1100

# 4. 1568-0.0, 1578-0.0, 1588-0.0, 1598-0.0, 1608-0.0, 5364-0.0: Weekly alcohol intake
#    Fields representing weekly intake of various alcoholic beverages
#    https://biobank.ndph.ox.ac.uk/ukb/field.cgi?id=5364

# 5. 2040-0.0: Risky behavior
#    "Would you describe yourself as someone who takes risks?"
#    https://biobank.ndph.ox.ac.uk/ukb/field.cgi?id=2040

# 6. 2149-0.0: Number of lifetime sexual partners
#    "About how many sexual partners have you had in your lifetime?"
#    https://biobank.ndph.ox.ac.uk/ukb/field.cgi?id=2149

# =========================================
# End of Field ID Information Summary
# =========================================

import numpy as np
import pandas as pd
from functools import reduce
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def compute_pca(df, cols, n_pca=None, std=True):
    df_selected = df[cols]
    # Standardize data if specified
    if std:
        scaler = StandardScaler()
        df_std = scaler.fit_transform(df_selected)
    else:
        df_std = df_selected

    pca = PCA(n_components=n_pca)
    pc_values = pca.fit_transform(df_std)
    df_pc = pd.DataFrame(df_std, columns=cols)
    for i in range(pc_values.shape[1]):
        df_pc[f'PC{i+1}'] = pc_values[:, i]
    # Display loadings and explained variance
    for i in range(pc_values.shape[1]):
        correlations = df_pc.corr()[f'PC{i+1}'].drop([f'PC{j+1}' for j in range(pc_values.shape[1])])
        print(f"PC {i+1} loadings:\n{correlations}")
    for i, variance_ratio in enumerate(pca.explained_variance_ratio_, start=1):
        print(f"PC {i} EV: {variance_ratio:.2f}%")
    return df_pc[[f'PC{i+1}' for i in range(pc_values.shape[1])]]

# Define list of field IDs for Mode 1 analysis
ukbb_mode1_field_id = [2149, 20453, 20116, 1249, 1239, 1568, 1578, 1588, 1598, 1608, 2040, 1100, 5364]
df_behavior_mode1 = pd.read_csv('/share/user_data/tianx/BWAS_RSRD/data/CCA_generalize/Mode1_field_ids_all.csv')

# =========================================
# 1. Lifetime Cannabis Use (Field ID: 20453-0.0)
# =========================================
y = '20453-0.0'
df_cannabis = df_behavior_mode1[['eid', y]]
df_cannabis[y][df_cannabis[y] < 0] = np.nan

# =========================================
# 2. Ever Smoking (Field IDs: 20116-0.0, 1249-0.0, 1239-0.0)
# =========================================
smoke_ylist = [f'{i}-0.0' for i in ['20116', '1249', '1239']]
df_smoke_raw = df_behavior_mode1[['eid'] + smoke_ylist].dropna()
df_smoke_bin = df_smoke_raw.loc[(df_smoke_raw[smoke_ylist] >= 0).all(axis=1), ['eid'] + smoke_ylist]
# Recode smoking frequency items in `df_smoke_bin` for consistent interpretation:
# - `1249-0.0` (Past tobacco use) and `1239-0.0` (Current tobacco use) fields are recoded to 
#   ensure that higher values represent higher smoking frequency, as the original scale 
#   inversely relates the values to frequency (higher values indicate lower smoking frequency).
#   Field details:
#   - `1249-0.0` (Past tobacco use): 
#       * Original scale: 1 = "Smoked on most or all days", 2 = "Smoked occasionally", 
#                         3 = "Just tried once or twice", 4 = "Never smoked".
#       * Mapping: Recoded to binary (1 for any smoking history, 0 for none).
#   - `1239-0.0` (Current tobacco use): 
#       * Original scale: 1 = "Yes, on most or all days", 2 = "Only occasionally", 0 = "No".
#       * Mapping: Recoded to binary (1 for any current smoking, 0 for none).
df_smoke_bin['1249-0.0'] = df_smoke_bin['1249-0.0'].map({1: 1, 2: 1, 3: 1, 4: 0})
df_smoke_bin['1239-0.0'] = df_smoke_bin['1239-0.0'].map({0: 0, 1: 1, 2: 1})
# Create a combined 'ever_smoking' column indicating any history of smoking behavior.
df_smoke_bin['ever_smoking'] = (df_smoke_bin[df_smoke_bin.columns[1:]].sum(axis=1) > 0).astype(int)

# =========================================
# 3. Driving Over Speed Limit (Field ID: 1100-0.0)
# =========================================
dfspeed = df_behavior_mode1[['eid', '1100-0.0']]
dfspeed = dfspeed[(dfspeed['1100-0.0'] >= 0) & (dfspeed['1100-0.0'] < 5)]

# =========================================
# 4. Weekly Alcohol Intake (Field IDs: 1568, 1578, 1588, 1598, 1608, 5364, instance 0)
# =========================================
week_drink_fields = [f'{i}-0.0' for i in [1568, 1578, 1588, 1598, 1608, 5364]]
SEX_label = '31-0.0'
pheno_data = df_behavior_mode1.loc[(df_behavior_mode1[week_drink_fields] >= 0).all(axis=1), ['eid'] + week_drink_fields]
pheno_data['total_weekly_intake'] = pheno_data[week_drink_fields].sum(axis=1)

# Filter by total weekly intake based on sex:
# To minimize potential neurotoxic effects associated with high alcohol consumption,
# we exclude heavy drinkers (females consuming >18 drinks/week and males consuming >24 drinks/week).
# Thresholds adopted from Aydogan et al. (2021), *Nature Human Behaviour*:
# Aydogan, G., Daviet, R., Karlsson Linnér, R., et al.
# "Genetic underpinnings of risky behaviour relate to altered neuroanatomy."
# Nat Hum Behav 5, 787–794 (2021). https://www.nature.com/articles/s41562-020-01027-y#Sec8
Data_fem = pheno_data[(pheno_data[SEX_label] == 0) & (pheno_data['total_weekly_intake'] < 18)]
Data_men = pheno_data[(pheno_data[SEX_label] == 1) & (pheno_data['total_weekly_intake'] < 24)]
df_alcohol = pd.concat([Data_fem, Data_men])
# Log-transform total weekly intake for further analysis
df_alcohol['Unstd_log_intake'] = np.log(df_alcohol['total_weekly_intake'] + 1)

# =========================================
# 5. Risky Behavior (Field ID: 2040-0.0)
# =========================================
y = '2040-0.0'
df_risky = df_behavior_mode1[['eid', y]].replace({y: {-3: np.nan}}).dropna()

# =========================================
# 6. Number of Lifetime Sexual Partners (Field ID: 2149-0.0)
# =========================================
dfsex = df_behavior_mode1[['eid', '2149-0.0']].replace({'2149-0.0': {-3: np.nan}}).dropna()

# Merge datasets for PCA
data_pca = reduce(lambda left, right: pd.merge(left, right, on='eid'), 
                  [dfspeed, dfsex, df_cannabis, df_smoke_bin, df_alcohol, df_risky])

# Rename columns for PCA input
data_pca = data_pca.rename(columns={
    '20453-0.0': 'Lifetime_cannabis_use',
    '1100-0.0': 'Drive_speed',
    '2149-0.0': 'Lifetime_sex_partner_num',
    'Unstd_log_intake': 'weekly_alcohol_intake',
    '2040-0.0': 'Risk_taking'
})

# Define columns to include in PCA
pca_input_col = ['Drive_speed', 'Lifetime_sex_partner_num', 'Lifetime_cannabis_use', 
                 'ever_smoking', 'weekly_alcohol_intake', 'Risk_taking']

# Load brain score data and merge with behavioral data
df_pbs = pd.read_csv('/share/user_data/tianx/BWAS_RSRD/data/CCA_generalize/UKBsamples_brainscore.csv')
df_all_valid = data_pca.merge(df_pbs, on='eid').reset_index(drop=True)

# Perform PCA and store result
res_pc = compute_pca(df_all_valid, pca_input_col, n_pca=5)
res_pc.to_csv('/share/user_data/tianx/BWAS_RSRD/data/CCA_generalize/UKB_Externalizing_Problems_PC.csv', index=False)