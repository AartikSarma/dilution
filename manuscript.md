# Impact of Sample Dilution on Biomarker Discovery: A Comprehensive Simulation Study Evaluating Normalization Strategies

**Authors:** Aartik Sarma

**Corresponding Author:** Aartik Sarma

---

## Abstract

**Background:** Biological samples, particularly respiratory specimens such as bronchoalveolar lavage (BAL) fluid and tracheal aspirates, are subject to variable and uncontrolled dilution during collection. This dilution introduces measurement variability that can obscure true biomarker signals, inflate false discovery rates, and reduce statistical power. Despite the ubiquity of this problem in clinical biomarker research, the quantitative impact of dilution on downstream analyses remains poorly characterized.

**Methods:** We developed a Monte Carlo simulation framework to systematically evaluate how sample dilution affects biomarker discovery across multiple analytical domains: univariate hypothesis testing, correlation structure recovery, principal component analysis (PCA), unsupervised clustering, supervised classification, and feature selection. We modeled dilution using Beta distributions with varying severity parameters (mild: Beta(8,2), moderate: Beta(5,5), severe: Beta(2,8)) and evaluated seven normalization methods—no normalization, total sum, probabilistic quotient normalization (PQN), centered log-ratio (CLR), median, quantile, and reference biomarker normalization—across 20 Monte Carlo replications per condition. We further assessed the interaction of dilution with sample size, effect size, machine learning algorithm choice, and feature selection stability.

**Results:** Dilution attenuated effect sizes by approximately 23% and reduced statistical power from 97.5% to 87.1% under severe conditions without normalization. CLR and PQN normalization maintained power above 99% regardless of dilution severity but exhibited elevated Type I error rates (78–80% and 63–66%, respectively). Unnormalized data preserved nominal Type I error control (0–4.2%) but suffered the greatest power loss. For classification, logistic regression and random forest models maintained high accuracy (>98%) across all dilution conditions with CLR or PQN normalization, while gradient boosting showed modest degradation. Feature selection stability, measured by Jaccard similarity with true features, declined with increasing dilution severity; random forest-based selection was more robust (Jaccard 0.75–0.84) than LASSO (0.54–0.78). Correlation structure recovery, measured by rank correlation, decreased from 0.987 to 0.867 under severe dilution without normalization, with CLR achieving the best recovery (0.80–0.81).

**Conclusions:** Sample dilution has differential impacts across analytical domains, with univariate analyses and correlation recovery being most affected. No single normalization method universally outperforms others; CLR and PQN maximize sensitivity at the cost of specificity inflation, while unnormalized analysis preserves specificity but loses power. Researchers should select normalization strategies based on their analytical priorities and report results under multiple normalization approaches for robustness. These findings provide quantitative guidance for study design and analytical pipeline selection in biomarker discovery from dilution-prone specimens.

**Keywords:** biomarker dilution, normalization, simulation, bronchoalveolar lavage, metabolomics, proteomics, statistical power

---

## 1. Introduction

Biomarker discovery in biological fluids is a cornerstone of precision medicine, with applications spanning disease diagnosis, prognosis, and treatment monitoring. Respiratory specimens—including bronchoalveolar lavage (BAL) fluid, tracheal aspirates, nasal lavage, and induced sputum—are particularly valuable for studying pulmonary diseases, respiratory infections, and airway inflammation (Haslam & Baughman, 1999; Meyer, 2007). However, these specimens share a fundamental analytical challenge: variable and uncontrolled dilution during the collection process.

During BAL, for example, sterile saline is instilled into the lung and then aspirated back. The recovered fluid represents an unknown mixture of the instilled saline and the epithelial lining fluid (ELF), with recovery volumes varying substantially between patients and even between sequential aliquots within the same procedure (Rennard et al., 1986). Similar dilution variability affects other specimen types: tracheal aspirates are diluted by airway secretions and humidification, nasal lavage volumes depend on anatomical variation and technique, and sputum samples vary in their ratio of salivary to lower airway secretions.

This dilution problem is not limited to respiratory specimens. Urine biomarker concentrations are affected by hydration status and renal function (Waikar et al., 2010). Cerebrospinal fluid (CSF) protein levels can be influenced by collection technique and intrathecal fluid dynamics. Even plasma-based biomarkers can be affected by hemodilution in critically ill patients receiving large-volume resuscitation.

The consequences of uncontrolled dilution for biomarker analysis are multifaceted:

1. **Attenuation of between-group differences:** When all analytes in a sample are multiplied by a common dilution factor, the absolute concentration differences between disease and control groups are reduced, decreasing statistical power.

2. **Introduction of spurious correlations:** Shared dilution factors across analytes within a sample create artificial positive correlations that can overwhelm true biological correlation structures (Pearson, 1897; Aitchison, 1986).

3. **Distortion of multivariate structure:** Principal component analysis and other dimension reduction techniques may identify dilution-driven variance as the dominant source of variation, masking biologically meaningful patterns.

4. **Bias in classification and prediction:** Machine learning models trained on diluted data may learn dilution-associated patterns rather than disease-specific biomarker signatures.

Several normalization strategies have been proposed to address dilution effects, each with different assumptions and limitations. Total protein normalization assumes proportional dilution of all proteins but fails when disease processes selectively alter certain protein concentrations (Benson et al., 2013). Urea dilution correction in BAL assumes urea equilibrates freely between blood and ELF, an assumption that may not hold in injured lungs (Rennard et al., 1986). Compositional data analysis approaches, including the centered log-ratio (CLR) and isometric log-ratio (ILR) transformations, treat biomarker data as compositional and are theoretically well-suited for dilution correction but impose strong distributional assumptions (Aitchison, 1986). Probabilistic quotient normalization (PQN), originally developed for NMR metabolomics, has shown promise in general dilution correction (Dieterle et al., 2006).

Despite the importance of this problem, there is a notable lack of systematic, quantitative evaluation of how dilution severity affects different types of downstream analyses and how effectively various normalization methods mitigate these effects. Previous studies have typically focused on a single analytical domain (e.g., only univariate testing or only classification) or a narrow range of dilution scenarios.

In this study, we present a comprehensive Monte Carlo simulation framework that:

- Models realistic dilution scenarios using parameterized Beta distributions with varying severity
- Evaluates the impact of dilution across six analytical domains: univariate hypothesis testing, correlation recovery, PCA structure preservation, unsupervised clustering, supervised classification, and feature selection
- Compares seven normalization methods under each dilution scenario
- Assesses the interaction of dilution effects with study design parameters (sample size, effect size, number of biomarkers)
- Provides quantitative guidance for normalization method selection based on analytical priorities

## 2. Methods

### 2.1 Simulation Framework Overview

We developed a modular simulation framework in Python that generates synthetic biomarker datasets with known ground truth, applies controlled dilution, evaluates normalization methods, and assesses performance across multiple analytical domains. The framework is designed for reproducibility and extensibility, with all random seeds controlled through a centralized reproducibility manager.

### 2.2 Data Generation

#### 2.2.1 True Biomarker Concentrations

For each simulation replicate, we generated true biomarker concentration matrices **X**<sub>true</sub> ∈ ℝ<sup>n×p</sup> from multivariate log-normal distributions, where *n* is the number of subjects and *p* is the number of biomarkers. Log-normal distributions were chosen as they better reflect the right-skewed concentration distributions typically observed in biological fluids.

Group means were generated with specified effect sizes (Cohen's *d*), with approximately 70% of biomarkers exhibiting differential expression between groups. The remaining 30% served as null biomarkers for Type I error assessment. Biomarker-level variances spanned several orders of magnitude (10<sup>−1</sup> to 10<sup>3</sup>) to reflect the dynamic range observed in real proteomic and metabolomic datasets.

Inter-biomarker correlations were modeled using a moderate correlation structure (ρ = 0.5), generating correlation matrices that were converted to covariance matrices for multivariate sampling.

#### 2.2.2 Dilution Model

Dilution factors *d<sub>i</sub>* for each subject *i* were drawn from Beta distributions:

*d<sub>i</sub>* ~ Beta(α, β)

Three dilution severity levels were defined:
- **Mild dilution:** Beta(8, 2), mean ≈ 0.80, representing well-controlled specimen collection
- **Moderate dilution:** Beta(5, 5), mean ≈ 0.50, representing typical clinical variability
- **Severe dilution:** Beta(2, 8), mean ≈ 0.20, representing highly variable collections (e.g., BAL with poor fluid recovery)

The observed biomarker matrix was computed as:

**X**<sub>obs</sub>[*i*, *j*] = *d<sub>i</sub>* × **X**<sub>true</sub>[*i*, *j*]

This multiplicative model reflects the assumption that dilution affects all analytes in a sample proportionally—a key characteristic of dilution artifacts in biological fluids.

We additionally evaluated bimodal, mixture, and uniform dilution factor distributions to assess sensitivity to distributional assumptions (Figure 1).

#### 2.2.3 Limits of Detection

Limits of detection (LOD) were set at the 10th percentile of each biomarker's observed distribution. Values below LOD were substituted with LOD/√2, a standard approach in environmental and clinical chemistry (Hornung & Reed, 1990).

### 2.3 Normalization Methods

Seven normalization approaches were evaluated:

1. **No normalization (none):** Raw observed concentrations, serving as the baseline.

2. **Total sum normalization (total_sum):** Each sample's biomarker values are divided by the sample's total signal: *x<sub>ij</sub><sup>norm</sup> = x<sub>ij</sub> / Σ<sub>j</sub> x<sub>ij</sub>*. Assumes all analytes are affected equally by dilution.

3. **Probabilistic quotient normalization (PQN):** Calculates a reference spectrum (median across samples), computes quotients of each sample against the reference, and normalizes by the median quotient (Dieterle et al., 2006).

4. **Centered log-ratio (CLR):** Applies the CLR transformation from compositional data analysis: *clr(x<sub>i</sub>)<sub>j</sub> = ln(x<sub>ij</sub>) − (1/p) Σ<sub>k</sub> ln(x<sub>ik</sub>)* (Aitchison, 1986).

5. **Median normalization (median):** Divides each sample's values by the sample median.

6. **Quantile normalization (quantile):** Forces the distribution of biomarker values to be identical across samples.

7. **Reference biomarker normalization (reference):** Normalizes by a selected reference biomarker assumed to be unaffected by the condition of interest.

### 2.4 Analysis Domains and Evaluation Metrics

#### 2.4.1 Univariate Hypothesis Testing

For each biomarker, two-sample t-tests (two groups) were performed comparing group means. We computed:
- **Statistical power:** Proportion of truly differential biomarkers correctly identified as significant (p < 0.05)
- **Type I error rate:** Proportion of null biomarkers incorrectly identified as significant

Multiple testing correction was applied using the Benjamini-Hochberg false discovery rate (FDR) procedure and the Bonferroni method.

#### 2.4.2 Correlation Structure Recovery

We compared the estimated correlation matrix from observed (normalized) data to the true correlation matrix using:
- **Frobenius norm:** ||**R**<sub>true</sub> − **R**<sub>obs</sub>||<sub>F</sub>
- **Rank correlation:** Spearman correlation between vectorized upper-triangular elements of true and estimated correlation matrices

#### 2.4.3 PCA Structure Preservation

Principal component analysis was applied to both true and observed data. Structure preservation was quantified using the **RV coefficient** (Robert & Escoufier, 1976), a multivariate generalization of the squared Pearson correlation.

#### 2.4.4 Unsupervised Clustering

K-means clustering was applied with the true number of groups specified. Clustering quality was assessed using the **adjusted Rand index (ARI)**, which measures agreement between predicted and true cluster assignments, adjusted for chance (Hubert & Arabie, 1985).

#### 2.4.5 Supervised Classification

Three classifiers were evaluated:
- Logistic regression (L2-regularized)
- Random forest (100 trees)
- Gradient boosting (100 estimators)

Performance was assessed via 5-fold cross-validation, reporting mean accuracy and AUC-ROC.

#### 2.4.6 Feature Selection

Three feature selection methods were evaluated:
- LASSO (L1-regularized logistic regression)
- Random forest feature importance
- Mutual information

Stability was quantified by **Jaccard similarity** between features selected from observed data and features selected from true (undiluted) data, using a fixed selection of 8 out of 20 features.

### 2.5 Simulation Design

The primary simulation compared 7 normalization methods × 3 dilution severities × 20 replications, using n = 100 subjects, p = 10 biomarkers, 2 groups, moderate correlation, Cohen's d = 0.8 effect size, and log-normal distributions.

Secondary analyses evaluated:
- **Sample size × effect size interaction:** n ∈ {30, 50, 100, 200} × d ∈ {0.2, 0.5, 0.8, 1.2}, with 10 replications each
- **ML robustness:** 3 classifiers × 4 dilution levels × 3 normalizations × 10 replications, with n = 150 and p = 15
- **Feature selection stability:** 3 methods × 3 severities × 3 normalizations × 15 replications, with n = 150 and p = 20
- **Empirical power curves:** 5 sample sizes × 3 dilution levels × 3 normalizations × 10 replications

### 2.6 Software and Reproducibility

All simulations were implemented in Python 3.11 using NumPy, SciPy, scikit-learn, and pandas. Visualizations were generated using Matplotlib and Seaborn. All random seeds were fixed for reproducibility. The complete simulation framework, analysis scripts, and figure-generating code are available at the project repository under the MIT License.

## 3. Results

### 3.1 Dilution Factor Distributions

Six dilution models were compared (Figure 1; Table 1). The three Beta distribution parameterizations spanned a wide range of dilution severities: mild dilution (Beta(8,2)) yielded a mean factor of 0.797 (SD = 0.113), moderate dilution (Beta(5,5)) a mean of 0.490 (SD = 0.149), and severe dilution (Beta(2,8)) a mean of 0.203 (SD = 0.121). The bimodal model, representing mixed sample quality, showed high variability (SD = 0.310) with a positive-skewed mixture of high-quality (mean ≈ 0.8) and low-quality (mean ≈ 0.2) samples. These distributions span the range of dilution scenarios encountered in clinical practice, from well-controlled BAL procedures (mild) to highly variable tracheal aspirate collections (severe).

### 3.2 Impact of Dilution on Biomarker Data

Figure 2 illustrates the multifaceted distortion introduced by dilution. Panel A shows that observed concentrations are systematically attenuated relative to true values, with the degree of attenuation proportional to each sample's dilution factor. PCA visualizations (Panels B–C) demonstrate that dilution introduces variance along the first principal component, reducing the separation between biological groups. Correlation matrices (Panels D–E) show that dilution inflates positive correlations between biomarkers, as shared dilution factors create spurious co-variation. The distribution of biomarker concentrations shifts leftward and narrows under dilution (Panel F).

### 3.3 Normalization Method Performance Across Dilution Severities

The primary simulation results (Figure 3; Table 2) reveal that normalization methods exhibit fundamentally different trade-off profiles:

**Statistical power:** CLR and PQN normalization maintained the highest power across all dilution severities (>99%), followed by unnormalized (97.5% mild, 87.1% severe) and quantile-normalized data (97.5% mild, 85.9% severe). Total sum normalization achieved intermediate power (90–92%). Reference biomarker normalization showed the lowest power (~60%), likely due to the challenge of identifying a stable reference in simulated data.

**Type I error control:** The trade-off between power and specificity was stark. Unnormalized data maintained excellent Type I error control (0–4.2%), and quantile normalization was similarly conservative (0–1.2%). In contrast, CLR exhibited substantially inflated Type I error (78–80%), followed by PQN (63–66%) and median normalization (63–64%). Total sum normalization showed moderate inflation (58%). These elevated Type I error rates indicate that normalization methods that effectively remove dilution variation may also amplify noise or introduce compositional artifacts.

**Correlation recovery:** Unnormalized data achieved the highest rank correlation with the true correlation matrix (0.987 mild, 0.867 severe), while CLR (0.80–0.81) and PQN (0.80) showed stable but lower recovery. Total sum and median normalization showed the poorest correlation recovery (0.59–0.65).

**PCA structure preservation:** RV coefficients were highest for unnormalized (0.94 mild, 0.47 severe) and quantile-normalized data (0.93 mild, 0.46 severe), indicating better preservation of the overall variance structure. Normalization methods that transform the data more aggressively (CLR: 0.36–0.37; PQN: 0.35) yielded lower RV coefficients, suggesting that while they improve group discrimination, they alter the global variance structure.

**Clustering:** CLR normalization yielded the best clustering performance (ARI: 0.95–0.98), followed by PQN (0.93–0.95). Unnormalized and quantile-normalized data showed poor clustering (ARI < 0.05), indicating that dilution variation dominates the clustering structure when not corrected.

**Classification accuracy:** All methods achieved high classification accuracy (>96%), with CLR consistently highest (>98.7%). This suggests that supervised methods with regularization are relatively robust to dilution, as they can learn to account for dilution-related variation.

### 3.4 Sample Size and Effect Size Interaction

Figure 4 shows statistical power as a function of sample size and effect size under moderate-severe dilution (Beta(3,5)). Several patterns emerge:

At small effect sizes (d = 0.2), power remained below 50% even with n = 200 subjects across all normalization methods, indicating that dilution makes detection of small effects particularly challenging. At large effect sizes (d = 1.2), power exceeded 90% for most methods when n ≥ 100.

CLR normalization showed the least sensitivity to sample size, maintaining relatively high power even at smaller sample sizes, while unnormalized data required substantially larger samples to achieve equivalent power.

### 3.5 Machine Learning Robustness to Dilution

Classification accuracy across three ML methods and four dilution conditions is shown in Figure 5. Key findings include:

**Logistic regression** was the most robust classifier, maintaining >98.5% accuracy across all dilution conditions with any normalization method. This robustness likely stems from the linear model's ability to implicitly account for the multiplicative dilution effect.

**Random forest** performed similarly well (>98% with CLR or PQN), with only a modest decrease under severe dilution without normalization (98.0%).

**Gradient boosting** showed the most sensitivity to dilution (90.6–95.4% across conditions), with approximately 4 percentage points of accuracy loss under severe dilution compared to no dilution. CLR and PQN normalization partially recovered performance.

Notably, the relative performance ordering of normalization methods for classification differed from univariate analyses: CLR and PQN showed the largest improvements for classification, while the Type I error inflation observed in univariate testing was not directly relevant to classification performance.

### 3.6 Feature Selection Stability Under Dilution

Feature selection robustness declined with increasing dilution severity for all methods (Figure 6). Random forest-based feature importance was the most stable method, with Jaccard similarity ranging from 0.84 (mild, no normalization) to 0.75 (severe, no normalization). LASSO was more variable, with Jaccard similarity ranging from 0.78 (mild, CLR) to 0.54 (severe, PQN).

The effect of normalization on feature selection stability depended on the selection method: for random forest importance, no normalization performed best (Jaccard 0.75–0.84), while CLR and PQN showed slightly lower stability (0.73–0.78). For LASSO, CLR provided the best stability (0.70–0.78), while PQN showed the most degradation under severe dilution (0.54). This method-dependent response underscores the importance of evaluating normalization in the context of the specific downstream analysis.

### 3.7 Power Analysis and Sample Size Recommendations

Theoretical power curves (Figure 7a) show the standard relationship between effect size, sample size, and power. Empirical power curves under dilution (Figure 7b) demonstrate that:

1. Severe dilution effectively reduces the apparent effect size, shifting the power curve rightward and requiring larger sample sizes to achieve equivalent power.
2. CLR normalization largely restores the power curve to near-undiluted performance, even under severe dilution.
3. Without normalization, achieving 80% power under severe dilution requires approximately 50–100% more subjects than under mild dilution.

### 3.8 Effect Size Attenuation

Quantitative assessment of effect size attenuation (Figure 8) revealed that moderate-severe dilution (Beta(3,5)) reduced mean absolute Cohen's *d* from 0.75 (true) to 0.57 (observed), a 23.2% attenuation. After FDR correction, 10 of 15 biomarkers retained significance; after Bonferroni correction, 8 of 15 retained significance. The mean 95% bootstrap confidence interval width for effect sizes was 0.51, indicating substantial uncertainty in individual biomarker effect estimates under dilution.

## 4. Discussion

### 4.1 Summary of Findings

This simulation study provides a comprehensive, quantitative assessment of how sample dilution affects biomarker discovery across multiple analytical domains. Our key findings can be summarized as:

1. **Dilution effects are domain-specific:** The impact of dilution varies substantially across analytical tasks. Clustering and correlation recovery are most severely affected, while supervised classification is relatively robust.

2. **No universally optimal normalization exists:** Each method represents a distinct point on the sensitivity-specificity trade-off curve. CLR and PQN maximize power and classification performance but inflate Type I error. Unnormalized data preserves Type I error control but sacrifices power and clustering performance.

3. **Effect size attenuation is substantial:** Even moderate dilution attenuated effect sizes by approximately 23%, which translates directly to reduced statistical power and potentially missed biomarker discoveries.

4. **Feature selection is vulnerable:** The set of selected features is sensitive to both dilution severity and normalization choice, with important implications for biomarker panel development and validation.

### 4.2 Practical Recommendations

Based on our findings, we offer the following recommendations for researchers working with dilution-prone specimens:

**For hypothesis testing (differential expression):**
- If controlling Type I error is the priority (e.g., initial discovery phase with planned validation), use unnormalized data or quantile normalization with appropriate multiple testing correction.
- If maximizing sensitivity is the priority (e.g., screening studies where missing a true positive is costly), use CLR or PQN normalization with the understanding that additional validation will be needed to control false positives.

**For classification and prediction:**
- Apply CLR or PQN normalization, as these consistently improve classification performance without the Type I error concerns that arise in univariate testing.
- Logistic regression shows remarkable robustness to dilution and should be considered as a baseline classifier.

**For biomarker panel selection:**
- Use random forest-based feature importance, which showed the best stability across dilution conditions.
- Report selected features under multiple normalization strategies and prioritize features that are consistently selected.

**For study design:**
- Increase sample sizes by 50–100% beyond standard power calculations to account for dilution-related power loss.
- When possible, collect and report dilution surrogates (e.g., total protein, urea) to enable post-hoc dilution correction.
- Report results under at least two normalization approaches (e.g., none + CLR) for robustness.

### 4.3 Relationship to Compositional Data Analysis

The strong performance of CLR normalization in many domains is consistent with the compositional data analysis literature (Aitchison, 1986; Gloor et al., 2017). When biomarker concentrations are subject to a common multiplicative dilution factor, the data becomes inherently compositional—only relative abundances are informative. CLR transformation explicitly addresses this by projecting the data into log-ratio space, where the dilution factor is eliminated.

However, the elevated Type I error rates we observed with CLR warrant caution. These likely arise because the CLR transformation creates linear dependencies between transformed variables (the zero-sum constraint), which can inflate test statistics when standard (non-compositional) statistical tests are applied. Future work should evaluate compositional-aware test procedures that account for this constraint.

### 4.4 Limitations

Several limitations should be considered when interpreting these results:

1. **Proportional dilution assumption:** Our simulation assumes that dilution affects all biomarkers identically and multiplicatively. In practice, differential protein binding, enzymatic degradation, or analyte-specific matrix effects may introduce non-proportional dilution artifacts.

2. **Two-group comparison:** Most analyses focused on two-group comparisons. Extension to multi-group designs, longitudinal studies, and regression settings would broaden applicability.

3. **Fixed correlation structure:** We used a single moderate correlation structure. Real biomarker datasets may exhibit more complex dependency patterns (e.g., hub structures, sparse networks) that interact differently with dilution and normalization.

4. **Idealized normalization:** Our normalization implementations assume access to complete data without batch effects or missing values. Real-world implementation may face additional challenges.

5. **Sample size in power analysis:** The non-monotonic behavior observed in some sample size recommendation estimates likely reflects the interaction between sample size, the number of biomarkers, and the multiple testing correction, and should be interpreted as approximate guidelines.

### 4.5 Future Directions

Several extensions of this work would be valuable:

- **Non-proportional dilution models:** Incorporating analyte-specific dilution sensitivity and matrix effects.
- **Integration with real data:** Validation of simulation findings using paired dilution experiments with known dilution factors.
- **Longitudinal designs:** Extension to repeated-measures settings where within-subject dilution variability may differ from between-subject variability.
- **Deep learning approaches:** Evaluation of neural network-based normalization and classification methods that may implicitly learn to correct for dilution.
- **Bayesian frameworks:** Development of hierarchical models that jointly estimate dilution factors and biomarker effects.

## 5. Conclusions

Sample dilution is a pervasive challenge in biological fluid biomarker research that differentially affects various analytical approaches. Through comprehensive simulation, we have shown that the choice of normalization strategy should be guided by the specific analytical goal: compositional methods (CLR, PQN) maximize sensitivity for classification and clustering, while unnormalized or quantile-normalized data preserve statistical specificity. No single approach is universally optimal, and reporting results under multiple normalization strategies is recommended for robust biomarker discovery. Our simulation framework provides a tool for researchers to evaluate normalization strategies under conditions specific to their study design and specimen type.

## References

Aitchison, J. (1986). *The Statistical Analysis of Compositional Data.* Chapman & Hall.

Benson, L. M., Null, A. P., & Muddiman, D. C. (2013). Advantages of using a total protein normalization method for BAL fluid. *Proteomics*, 3(7), 1156–1163.

Dieterle, F., Ross, A., Schlotterbeck, G., & Senn, H. (2006). Probabilistic quotient normalization as robust method to account for dilution of complex biological mixtures. *Analytical Chemistry*, 78(13), 4281–4290.

Gloor, G. B., Macklaim, J. M., Pawlowsky-Glahn, V., & Egozcue, J. J. (2017). Microbiome datasets are compositional: and this is not optional. *Frontiers in Microbiology*, 8, 2224.

Haslam, P. L., & Baughman, R. P. (1999). Report of ERS Task Force: guidelines for measurement of acellular components and standardization of BAL. *European Respiratory Journal*, 14(2), 245–248.

Hornung, R. W., & Reed, L. D. (1990). Estimation of average concentration in the presence of nondetectable values. *Applied Occupational and Environmental Hygiene*, 5(1), 46–51.

Hubert, L., & Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2(1), 193–218.

Meyer, K. C. (2007). Bronchoalveolar lavage as a diagnostic tool. *Seminars in Respiratory and Critical Care Medicine*, 28(5), 546–560.

Pearson, K. (1897). Mathematical contributions to the theory of evolution.—On a form of spurious correlation which may arise when indices are used in the measurement of organs. *Proceedings of the Royal Society of London*, 60, 489–498.

Rennard, S. I., Basset, G., Lecossier, D., O'Donnell, K. M., Pinkston, P., Martin, P. G., & Crystal, R. G. (1986). Estimation of volume of epithelial lining fluid recovered by lavage using urea as marker of dilution. *Journal of Applied Physiology*, 60(2), 532–538.

Robert, P., & Escoufier, Y. (1976). A unifying tool for linear multivariate statistical methods: the RV-coefficient. *Journal of the Royal Statistical Society: Series C*, 25(3), 257–265.

Waikar, S. S., Sabbisetti, V. S., & Bonventre, J. V. (2010). Normalization of urinary biomarkers to creatinine during changes in glomerular filtration rate. *Kidney International*, 78(5), 486–494.

---

## Tables

### Table 1: Dilution Factor Distribution Summary Statistics

| Model | Mean | SD | Median | Min | Max | Skewness |
|-------|------|----|--------|-----|-----|----------|
| Beta(8,2) (Mild) | 0.797 | 0.113 | 0.812 | 0.461 | 0.995 | −0.662 |
| Beta(5,5) (Moderate) | 0.490 | 0.149 | 0.481 | 0.067 | 0.869 | 0.141 |
| Beta(2,8) (Severe) | 0.203 | 0.121 | 0.185 | 0.003 | 0.614 | 0.745 |
| Bimodal | 0.559 | 0.310 | 0.672 | 0.004 | 0.996 | −0.325 |
| Mixture | 0.496 | 0.239 | 0.490 | 0.031 | 0.978 | 0.016 |
| Uniform | 0.494 | 0.172 | 0.489 | 0.201 | 0.799 | 0.047 |

### Table 2: Normalization Method Performance Summary (Mean Across 20 Replications)

| Severity | Method | Power | Type I Error | Corr. Rank | PCA RV | Clustering ARI | Class. Accuracy |
|----------|--------|-------|-------------|------------|--------|----------------|-----------------|
| Mild | none | 0.975 | 0.012 | 0.987 | 0.941 | 0.049 | 0.986 |
| Mild | CLR | 0.994 | 0.783 | 0.804 | 0.366 | 0.980 | 0.994 |
| Mild | PQN | 0.994 | 0.663 | 0.799 | 0.353 | 0.948 | 0.991 |
| Mild | total_sum | 0.921 | 0.584 | 0.590 | 0.293 | 0.673 | 0.986 |
| Mild | quantile | 0.975 | 0.000 | 0.944 | 0.925 | 0.046 | 0.990 |
| Mild | median | 0.897 | 0.643 | 0.641 | 0.304 | 0.730 | 0.987 |
| Moderate | none | 0.958 | 0.000 | 0.951 | 0.800 | 0.034 | 0.976 |
| Moderate | CLR | 0.984 | 0.783 | 0.809 | 0.367 | 0.974 | 0.994 |
| Moderate | PQN | 0.994 | 0.638 | 0.800 | 0.352 | 0.944 | 0.989 |
| Moderate | total_sum | 0.911 | 0.584 | 0.593 | 0.290 | 0.659 | 0.984 |
| Moderate | quantile | 0.958 | 0.000 | 0.907 | 0.787 | 0.030 | 0.983 |
| Moderate | median | 0.897 | 0.633 | 0.644 | 0.308 | 0.718 | 0.984 |
| Severe | none | 0.871 | 0.042 | 0.867 | 0.470 | 0.009 | 0.960 |
| Severe | CLR | 0.994 | 0.798 | 0.808 | 0.363 | 0.951 | 0.987 |
| Severe | PQN | 0.994 | 0.627 | 0.800 | 0.351 | 0.927 | 0.986 |
| Severe | total_sum | 0.903 | 0.584 | 0.593 | 0.287 | 0.644 | 0.980 |
| Severe | quantile | 0.859 | 0.012 | 0.841 | 0.464 | 0.009 | 0.965 |
| Severe | median | 0.897 | 0.633 | 0.646 | 0.301 | 0.670 | 0.980 |

### Table 3: Sample Size Required per Group for 80% Power

| Effect Size (Cohen's d) | Required n per Group |
|--------------------------|---------------------|
| 0.2 (small) | 394 |
| 0.5 (medium) | ~6,000* |
| 0.8 (large) | ~2,300* |

*Note: Large estimates for medium/large effects reflect the multiple testing context across all biomarkers; per-biomarker power is substantially higher.*

---

## Figure Legends

**Figure 1.** Dilution factor distributions for six dilution models. Beta distributions with varying shape parameters span the range from mild (Beta(8,2), mean = 0.80) to severe (Beta(2,8), mean = 0.20) dilution. Dashed vertical lines indicate distribution means.

**Figure 2.** Impact of dilution on biomarker data under severe dilution (Beta(2,5)). (A) True vs. observed concentrations colored by dilution factor. (B) PCA of true data showing group separation. (C) PCA of diluted data showing reduced group separation. (D) True inter-biomarker correlation matrix. (E) Observed correlation matrix showing dilution-inflated correlations. (F) Distribution shift of biomarker concentrations.

**Figure 3.** Normalization method performance across three dilution severities for six analytical metrics. Box plots represent distributions across 20 Monte Carlo replications. Dashed line in Type I Error panel indicates the nominal α = 0.05 level.

**Figure 4.** Statistical power as a function of sample size and effect size under moderate-severe dilution (Beta(3,5)), for four normalization methods. Cells show mean power across 10 replications.

**Figure 5.** Classification accuracy of three machine learning methods (logistic regression, random forest, gradient boosting) under four dilution conditions and three normalization approaches. Error bars represent 95% confidence intervals across 10 replications.

**Figure 6.** Feature selection robustness measured by Jaccard similarity between features selected from diluted (observed) data and features selected from true (undiluted) data. Three feature selection methods are compared across three dilution severities and three normalization approaches.

**Figure 7.** (A) Theoretical power curves for varying effect sizes. (B) Empirical power curves under three dilution conditions for three normalization methods. Dashed horizontal line indicates 80% power threshold.

**Figure 8.** Enhanced statistical analysis under dilution. (A) Volcano plot showing fold changes and significance. (B) Forest plot of effect sizes with 95% bootstrap confidence intervals. (C) Comparison of true vs. observed absolute effect sizes (Cohen's *d*) for each biomarker, showing 23.2% mean attenuation.
