#  RefinedRSRD

## Feature selection through ICC
We selected these 44 features based on ICC analysis(RefinedRSRD/ICC.py) of 4945 time-series feature (Dense RSRD profiles) from four rfMRI scans. The data used for feature selection was derived from 100 blood-unrelated HCP-YA subjects, characterized by minimal head movement effects. 

## Time-series features in Refined RSRD profiles
The computation of these features is based on the [hctsa](https://github.com/benfulcher/hctsa) toolbox. Detailed information regarding these features is provided in the **Table S3**. 
For the specific computation methods and references, please refer to the code and annotations officially provided by the [hctsa](https://github.com/benfulcher/hctsa/tree/main/Operations) toolbox.


**Table of the 44 features and their calculation methods** :

|          **Feature name in the manu**          |       **Feature name in hctsa toolbox**        |               **Corresponding code in hctsa toolbox**                |
| :--------------------------------: | :--------------------------------------------: | :----------------------------------------------------------: |
|                mean                |                      mean                      | [DN_mean](https://github.com/benfulcher/hctsa/blob/main/Operations/DN_Mean.m) |
|           harmonic\_mean           |                 harmonic\_mean                 | [DN\_hmean](https://github.com/benfulcher/hctsa/blob/main/Operations/DN_Mean.m) |
|               median               |                     median                     | [DN\_median](https://github.com/benfulcher/hctsa/blob/main/Operations/DN_Mean.m) |
|          trimmed\_mean\.1          |                trimmed\_mean\_1                | [DN\_TrimmedMean\_1](https://github.com/benfulcher/hctsa/blob/main/Operations/DN_TrimmedMean.m) |
|          trimmed\_mean\.5          |                trimmed\_mean\_5                | [DN\_TrimmedMean\_5](https://github.com/benfulcher/hctsa/blob/main/Operations/DN_TrimmedMean.m) |
|         trimmed\_mean\.10          |               trimmed\_mean\_10                | [DN\_TrimmedMean\_10](https://github.com/benfulcher/hctsa/blob/main/Operations/DN_TrimmedMean.m) |
|         trimmed\_mean\.25          |               trimmed\_mean\_25                | [DN\_TrimmedMean\_25](https://github.com/benfulcher/hctsa/blob/main/Operations/DN_TrimmedMean.m) |
|         trimmed\_mean\.50          |               trimmed\_mean\_50                | [DN\_TrimmedMean\_50](https://github.com/benfulcher/hctsa/blob/main/Operations/DN_TrimmedMean.m) |
|              midhinge              |                    midhinge                    |                 [DN\_midhinge](https://github.com/benfulcher/hctsa/blob/main/Operations/DN_Mean.m)                 |
|                rms                 |                      rms                       |                      [DN\_rms](https://github.com/benfulcher/hctsa/blob/main/Operations/DN_Mean.m)                      |
|        standard\_deviation         |              standard\_deviation               | [DN\_Spread\_std](https://github.com/benfulcher/hctsa/blob/main/Operations/DN_Spread.m) |
|         skewness\_pearson          |               skewness\_pearson                | [DN\_CustomSkewness\_pearson](https://github.com/benfulcher/hctsa/blob/main/Operations/DN_CustomSkewness.m) |
|    FitKernelSmoothraw\.entropy     |        DN\_FitKernelSmoothraw\_entropy         | [DN\_FitKernelSmoothraw\.entropy](https://github.com/benfulcher/hctsa/blob/main/Operations/DN_FitKernelSmooth.m) |
|    CompareKSFit\.rayleigh\.psx     |        DN\_CompareKSFit\_rayleigh\_psx         | [DN\_CompareKSFit\_rayleigh\.peaksepx](https://github.com/benfulcher/hctsa/blob/main/Operations/DN_CompareKSFit.m) |
|          rawHRVmeas\.SD1           |              MD\_rawHRVmeas\_SD1               | [MD\_rawHRVmeas\.SD1](https://github.com/benfulcher/hctsa/blob/main/Operations/MD_rawHRVmeas.m) |
|            StdNthDer\.2            |                SY\_StdNthDer\_2                | [SY\_StdNthDer\_2](https://github.com/benfulcher/hctsa/blob/main/Operations/SY_StdNthDer.m) |
|            StdNthDer\.3            |                SY\_StdNthDer\_3                | [SY\_StdNthDer\_3](https://github.com/benfulcher/hctsa/blob/main/Operations/SY_StdNthDer.m) |
|            StdNthDer\.4            |                SY\_StdNthDer\_4                | [SY\_StdNthDer\_4](https://github.com/benfulcher/hctsa/blob/main/Operations/SY_StdNthDer.m) |
|            StdNthDer\.5            |                SY\_StdNthDer\_5                | [SY\_StdNthDer\_5](https://github.com/benfulcher/hctsa/blob/main/Operations/SY_StdNthDer.m) |
|           StdNthDer\.10            |               SY\_StdNthDer\_10                | [SY\_StdNthDer\_10](https://github.com/benfulcher/hctsa/blob/main/Operations/SY_StdNthDer.m) |
|         crinkle\_statistic         |             DK\_crinkle\_statistic             | [DK\_crinkle\_statistic](https://github.com/benfulcher/hctsa/blob/main/Toolboxes/Danny_Kaplan/DK_crinkle.m) |
|           trev\.1\.denom           |               CO\_trev\_1\_denom               | [CO\_trev\_1\.denom](https://github.com/benfulcher/hctsa/blob/main/Operations/CO_trev.m) |
|           trev\.2\.denom           |               CO\_trev\_2\_denom               | [CO\_trev\_2\.denom](https://github.com/benfulcher/hctsa/blob/main/Operations/CO_trev.m) |
|           trev\.3\.denom           |               CO\_trev\_3\_denom               | [CO\_trev\_3\.denom](https://github.com/benfulcher/hctsa/blob/main/Operations/CO_trev.m) |
|         Embed2\.Dist\.mean         |         CO\_Embed2\_Dist\_tau\_d\_mean         | [CO\_Embed2\_Dist\_tau\.d\_mean](https://github.com/benfulcher/hctsa/blob/main/Operations/CO_Embed2_Dist.m) |
|      LocalSimple\.meanabserr       |       FC\_LocalSimple\_lfit2\_meanabserr       | [FC\_LocalSimple\_lfit2\.meanabserr](https://github.com/benfulcher/hctsa/blob/main/Operations/FC_LocalSimple.m) |
|        LocalSimple\.stderr         |         FC\_LocalSimple\_lfit3\_stderr         | [FC\_LocalSimple\_lfit3\.stderr](https://github.com/benfulcher/hctsa/blob/main/Operations/FC_LocalSimple.m) |
|          Nlpe\.mi\_msqerr          |          NL\_MS\_nlpe\_2\_mi\_msqerr           | [NL\_MS\_nlpe\_2\_mi\.msqerr](https://github.com/benfulcher/hctsa/blob/main/Operations/NL_MS_nlpe.m) |
| VisibilityGraph\.horiz\.gaussnlogL |     NW\_VisibilityGraph\_horiz\_gaussnlogL     | [NW\_VisibilityGraph\_horiz\.gaussnlogL](https://github.com/benfulcher/hctsa/blob/main/Operations/NW_VisibilityGraph.m) |
|    VisibilityGraph\.norm\.meank    |        NW\_VisibilityGraph\_norm\_meank        | [NW\_VisibilityGraph\_norm\.meank](https://github.com/benfulcher/hctsa/blob/main/Operations/NW_VisibilityGraph.m) |
|  VisibilityGraph\.norm\.expnlogL   |      NW\_VisibilityGraph\_norm\_expnlogL       | [NW\_VisibilityGraph\_norm\.expnlogL](https://github.com/benfulcher/hctsa/blob/main/Operations/NW_VisibilityGraph.m) |
|  VisibilityGraph\.norm\.evparam1   |      NW\_VisibilityGraph\_norm\_evparam1       | [NW\_VisibilityGraph\_norm\.evparam1](https://github.com/benfulcher/hctsa/blob/main/Operations/NW_VisibilityGraph.m) |
|          Walker\.distdiff          |       PH\_Walker\_prop\_09\_sw\_distdiff       | [PH\_Walker\_prop\_09\.sw\_distdiff](https://github.com/benfulcher/hctsa/blob/main/Operations/PH_Walker.m) |
|          PPtest\.meanstat          |       SY\_PPtest\_0\_5\_ar\_t1\_meanstat       | [SY\_PPtest\_0\_5\_ar\_t1\.meanstat](https://github.com/benfulcher/hctsa/blob/main/Operations/SY_PPtest.m) |
|          PPtest\.maxstat           |       SY\_PPtest\_0\_5\_ar\_t1\_maxstat        | [SY\_PPtest\_0\_5\_ar\_t1\.maxstat](https://github.com/benfulcher/hctsa/blob/main/Operations/SY_PPtest.m) |
|          PPtest\.minstat           |       SY\_PPtest\_0\_5\_ar\_t1\_minstat        | [SY\_PPtest\_0\_5\_ar\_t1\.minstat](https://github.com/benfulcher/hctsa/blob/main/Operations/SY_PPtest.m) |
|      VarRatioTest\.2\.0\.stat      |          SY\_VarRatioTest\_2\_0\_stat          | [SY\_VarRatioTest\_2\_0\.stat](https://github.com/benfulcher/hctsa/blob/main/Operations/SY_VarRatioTest.m) |
|      VarRatioTest\.2\.1\.stat      |          SY\_VarRatioTest\_2\_1\_stat          | [SY\_VarRatioTest\_2\_1\.stat](https://github.com/benfulcher/hctsa/blob/main/Operations/SY_VarRatioTest.m) |
|      VarRatioTest\.4\.0\.stat      |          SY\_VarRatioTest\_4\_0\_stat          | [SY\_VarRatioTest\_4\_0\.stat](https://github.com/benfulcher/hctsa/blob/main/Operations/SY_VarRatioTest.m) |
|      VarRatioTest\.4\.1\.stat      |          SY\_VarRatioTest\_4\_1\_stat          | [SY\_VarRatioTest\_4\_1\.stat](https://github.com/benfulcher/hctsa/blob/main/Operations/SY_VarRatioTest.m) |
|       VarRatioTest\.meanstat       | SY\_VarRatioTest\_24682468\_00001111\_meanstat | [SY\_VarRatioTest\_24682468\_00001111\.meanstat](https://github.com/benfulcher/hctsa/blob/main/Operations/SY_VarRatioTest.m) |
|       VarRatioTest\.maxstat        | SY\_VarRatioTest\_24682468\_00001111\_maxstat  | [SY\_VarRatioTest\_24682468\_00001111\.maxstat](https://github.com/benfulcher/hctsa/blob/main/Operations/SY_VarRatioTest.m) |
|       VarRatioTest\.minstat        | SY\_VarRatioTest\_24682468\_00001111\_minstat  | [SY\_VarRatioTest\_24682468\_00001111\.minstat](https://github.com/benfulcher/hctsa/blob/main/Operations/SY_VarRatioTest.m) |






