# =====================================================================================
# Description:
#
# This script performs spatial autocorrelation-preserving null modeling using
# the BrainSMASH framework. It evaluates the statistical significance of spatial 
# correlations between key spatiotemporal patterns identified in two modes 
# across cohorts.
#
# Inputs:
# 1. Brain maps derived from different cohorts, containing region-wise correlations 
#    between key temporal features and behavioral measures:
#    - Cohorts: HCP-YA, HCP-D, UK Biobank
# 2. MNI coordinates for brain regions of interest (ROIs)
#
# Outputs:
# 1. Surrogate correlation arrays generated using spatially-constrained null models
# 2. Visualizations of empirical vs. null correlation distributions
# 3. Empirical p-values based on the BrainSMASH null distribution
# =====================================================================================


from brainsmash.mapgen.eval import sampled_fit
from brainsmash.mapgen.stats import pearsonr, pairwise_r
from brainsmash.workbench.geo import volume
from brainsmash.mapgen.sampled import Sampled
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import os
import brainsmash

def calculate_p_brainsmash(test_stat, surrogate_corrs):
    """
    Calculate the p-value (p_brainsmash) of the test_stat relative to the surrogate correlations.
    Parameters:
    - test_stat: float, the empirical statistic whose significance is to be tested.
    - surrogate_corrs: array, correlation coefficients derived from spatially constrained surrogate maps (brainsmash).

    Returns:
    - p_brainsmash: float, the p-value indicating the probability of observing a test_stat as or more extreme under the null.
    """
    more_extreme = np.sum(surrogate_corrs >= test_stat)  # One-tailed test
    p_brainsmash = more_extreme / len(surrogate_corrs)
    return p_brainsmash


def plot_brainsmash_results(
    brainmap1_val,
    brainmap2_val,
    surrogate_corrs,
    naive_pairwise_corrs,
    brainmap1_label,
    brainmap2_label,
    output_dir,
    save_prefix="correlation_distribution"
):
    """
    Visualize results of brainsmash-based spatial autocorrelation-preserving test.
    # derived from brainsmash https://brainsmash.readthedocs.io/en/latest/example.html
    Parameters:
    - brainmap1_val: np.ndarray, values of the first brain map
    - brainmap2_val: np.ndarray, values of the second brain map
    - surrogate_corrs: np.ndarray, correlations between brainmap2 and brainsmash surrogates of brainmap1
    - naive_pairwise_corrs: np.ndarray, correlations between random permutations of brainmap1
    - brainmap1_label, brainmap2_label: str, map names
    - output_dir: str, save directory
    - save_prefix: str, output filename prefix
    Returns:
    - p_brainsmash: float, p-value from brainsmash null model
    """
    test_stat = stats.pearsonr(brainmap1_val, brainmap2_val)[0]
    p_brainsmash = calculate_p_brainsmash(test_stat, surrogate_corrs)

    sac = '#377eb8'  # SAC-preserving
    rc = '#e41a1c'   # Random control
    bins = np.linspace(-1, 1, 51)
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes([0.2, 0.25, 0.6, 0.6])
    ax2 = ax.twinx()
    ax.axvline(test_stat, 0, 0.8, color='k', linestyle='dashed', lw=1)
    ax.hist(surrogate_corrs, bins=bins, color=sac, alpha=1, density=True)
    ax2.hist(naive_pairwise_corrs, bins=bins, color=rc, alpha=0.7, density=True)
    ax.set_xticks(np.arange(-1, 1.1, 0.5))
    ax.spines['left'].set_color(sac)
    ax.tick_params(axis='y', colors=sac)
    ax2.spines['right'].set_color(rc)
    ax2.tick_params(axis='y', colors=rc)
    [s.set_visible(False) for s in [ax.spines['top'], ax.spines['right'], ax2.spines['top'], ax2.spines['left']]]
    ax.text(0.97, 1.1, 'Random Permuted', ha='right', va='bottom', color=rc, transform=ax.transAxes)
    ax.text(0.97, 1.03, 'SAC-Preserving', ha='right', va='bottom', color=sac, transform=ax.transAxes)
    ax.text(test_stat, 1.65, f"{brainmap1_label}/{brainmap2_label}\nmap", ha='center', va='bottom')
    ax.text(0.5, -0.2, f"{brainmap1_label}/{brainmap2_label}\n p = {p_brainsmash:.4f}",
            ha='center', va='top', transform=ax.transAxes)
    ax.text(-0.3, 0.5, "Density", rotation=90, ha='left', va='center', transform=ax.transAxes)
    fig.savefig(f'{output_dir}/{save_prefix}.svg', dpi=200)
    fig.savefig(f'{output_dir}/{save_prefix}.png', dpi=200)
    plt.close(fig)
    print(f"Plot saved to {output_dir}.")


def run_brainsmash_pipeline(coord_file,
                            mode_label,
                            brainmap1_label,
                            brainmap2_label,
                            n_perum=10000,
                            nsurr=30,
                            output_root="./brainsmash_results",
                            kwargs=None):
    """
    Pipeline for conducting spatial autocorrelation-preserving analysis using brainsmash.

    Parameters:
    - coord_file: str, path to MNI coordinate file
    - mode_label: str, identifier for the analysis mode
    - brainmap1_label, brainmap2_label: str, map labels
    - n_perum: int, number of brainsmash surrogates to generate
    - nsurr: int, number of surrogates for fitting diagnostics
    - output_root: str, root output directory
    - kwargs: dict, parameters for Sampled()

    Outputs:
    - Surrogate correlation arrays and visualizations saved in output folder
    """
    if kwargs is None:
        kwargs = {
            'ns': 271, 'knn': 60,'pv': 70,'nh': 13,'deltas': [0.3, 0.5, 0.7, 0.9],
        }

    output_dir = f"{output_root}/surrogate/{mode_label}/{brainmap1_label}_{brainmap2_label}"
    os.makedirs(output_dir, exist_ok=True)
    filenames = volume(coord_file, output_dir)
    brainmap1_path = f"{output_root}/{mode_label}/{brainmap1_label}.txt"
    brainmap2_path = f"{output_root}/{mode_label}/{brainmap2_label}.txt"
    brainmap1_val = np.loadtxt(brainmap1_path)
    brainmap2_val = np.loadtxt(brainmap2_path)
    sampled_fit(brainmap1_path, filenames['D'], filenames['index'], nsurr=nsurr, **kwargs)
    gen = Sampled(x=brainmap1_path, D=filenames['D'], index=filenames['index'], **kwargs)

    try:
        surrogate_maps = gen(n=n_perum)
        print("Brainsmash surrogate maps generated successfully.")
    except ValueError as e:
        print("Error during surrogate generation:", e)
        return

    surrogate_brainmap_corrs = pearsonr(brainmap2_val, surrogate_maps).flatten()
    surrogate_pairwise_corrs = pairwise_r(surrogate_maps, flatten=True)

    naive_surrogates = np.array([np.random.permutation(brainmap1_val) for _ in range(1000)])
    naive_brainmap_corrs = pearsonr(brainmap2_val, naive_surrogates).flatten()
    naive_pairwise_corrs = pairwise_r(naive_surrogates, flatten=True)

    np.save(f"{output_dir}/surrogate_brainmap_corrs.npy", surrogate_brainmap_corrs)
    np.save(f"{output_dir}/surrogate_pairwise_corrs.npy", surrogate_pairwise_corrs)
    np.save(f"{output_dir}/naive_surrogates.npy", naive_surrogates)
    np.save(f"{output_dir}/naive_brainmap_corrs.npy", naive_brainmap_corrs)

    p_brainsmash = plot_brainsmash_results(
        brainmap1_val=brainmap1_val,
        brainmap2_val=brainmap2_val,
        surrogate_corrs=surrogate_brainmap_corrs,
        naive_pairwise_corrs=naive_pairwise_corrs,
        brainmap1_label=brainmap1_label,
        brainmap2_label=brainmap2_label,
        output_dir=output_dir
    )
    return p_brainsmash

# ===============================
# Load behavioral mode-specific maps and prepare for BrainSMASH
# ===============================
df_m1 = pd.read_csv('./mode1_spatial_cross_cohorts.csv')
df_m2 = pd.read_csv('./mode2_spatial_cross_cohorts.csv')

# Write Mode 1 brain maps to txt for brainsmash
brainsmash_base_dir = './bs_permutation'
mode_label = 'mode1'
os.makedirs(f'{brainsmash_base_dir}/mode1',exist_ok=True)
for col in df_m1.columns:
    vals = df_m1[col].values.tolist()
    with open(f'{brainsmash_base_dir}/{mode_label}/{col}.txt', 'w') as f:
        for item in vals:
            f.write("%s\n" % item)

# Write Mode 2 brain maps to txt for brainsmash
os.makedirs(f'{brainsmash_base_dir}/mode2',exist_ok=True)
for col in df_m2.columns:
    vals = df_m2[col].values.tolist()
    with open(f'{brainsmash_base_dir}/{mode_label}/{col}.txt', 'w') as f:
        for item in vals:
            f.write("%s\n" % item)

# ===============================
# Run BrainSMASH test across all pairwise comparisons
# ===============================

run_brainsmash_pipeline(
    coord_file='./271roi_coords.txt',mode_label='mode1',
    brainmap1_label='HCP-YA',brainmap2_label='HCP-D',
    output_root = brainsmash_base_dir, n_perum=10000
)

run_brainsmash_pipeline(
    coord_file='./271roi_coords.txt',mode_label='mode1',
    brainmap1_label='HCP-YA',brainmap2_label='UK Biobank',  
    output_root = brainsmash_base_dir, n_perum=10000
)

run_brainsmash_pipeline(
    coord_file='./271roi_coords.txt',mode_label='mode1',
    brainmap1_label='HCP-D',brainmap2_label='UK Biobank',  
    output_root = brainsmash_base_dir, n_perum=10000
)

run_brainsmash_pipeline(
    coord_file='./271roi_coords.txt',mode_label='mode2',
    brainmap1_label='HCP-YA',brainmap2_label='HCP-D',
    output_root = brainsmash_base_dir, n_perum=10000
)

run_brainsmash_pipeline(
    coord_file='./271roi_coords.txt',mode_label='mode2',
    brainmap1_label='HCP-YA',brainmap2_label='UK Biobank',  
    output_root = brainsmash_base_dir, n_perum=10000
)

run_brainsmash_pipeline(
    coord_file='./271roi_coords.txt',mode_label='mode2',
    brainmap1_label='HCP-D',brainmap2_label='UK Biobank',  
    output_root = brainsmash_base_dir, n_perum=10000
)