# Practical Task 1: Industry Classification from Business-Description Embeddings

Notes and findings for `task1.1.ipynb`. All numbers come from the notebook run (train/validation split 90/10,
`random_state=42`).

## Mandatory part

### Data and label encoding
- 31,706 companies, each with a 768-dimensional business-description embedding and one of 25 industries.
- Train / validation: 28,535 / 3,171 companies. All validation classes also appear in the training set.

### Class distribution
The classes are strongly imbalanced. Materials makes up about 25% of all companies, while the smallest industries
(e.g. Consumer Staples Distribution & Retail, Household & Personal Products) are below 1%. The largest class is 47
times the size of the smallest (8,394 vs. 178 companies).

### PCA
- About 4,200 companies share one identical embedding (probably an empty description). These and the 122 most
  distant points were removed before the PCA; 24,224 companies remain.
- The first two components explain only a small part of the variance, so the plot is a rough 2-D view of a
  768-dimensional space.
- Even so, industries occupy different regions (e.g. Pharmaceuticals and Banks sit at opposite ends), while related
  industries overlap (e.g. Semiconductors and Software).
- There is a clear signal for industry classification, but neighbouring industries will be confused.

### Logistic Regression vs. kNN

| Model | Accuracy | F1 weighted | F1 macro |
|---|---|---|---|
| Majority-class baseline | 0.249 | 0.099 | 0.016 |
| Logistic Regression (C = 4) | 0.777 | 0.773 | 0.710 |
| kNN (k = 15, cosine) | 0.775 | 0.770 | 0.699 |

- **Accuracy** = correctly classified companies / all companies.
- **Weighted F1** weights each industry's F1 by its size. It is the main metric because the classes are imbalanced.
- **Macro F1** gives every industry the same weight, so it shows how well the small industries are handled.
- Both models perform well (≈ 78% accuracy vs. 25% for the baseline). Macro F1 is lower than weighted F1, so small
  industries are classified worse than large ones.

### Errors vs. class size
- Larger classes generally get higher F1. Materials (largest) scores highest and Consumer Staples Distribution &
  Retail (smallest) lowest.
- The relationship is not consistent: Telecommunication Services reaches a similar F1 to Capital Goods with about
  14 times fewer training examples.
- Well-identified industries (F1 > 0.8): Materials, Pharmaceuticals/Biotechnology, Food/Beverage/Tobacco, Banks,
  Transportation.

### Which industries are similar?
Two industries count as similar if the model often confuses them. Their confusion rates are turned into a distance
and the industries are clustered hierarchically. The most similar pairs:
- Capital Goods ↔ Automobiles & Components
- Technology Hardware & Equipment ↔ Semiconductors & Semiconductor Equipment
- Software & Services ↔ Commercial & Professional Services
- Pharmaceuticals & Biotechnology ↔ Household & Personal Products
- Real Estate Management & Development ↔ Equity Real Estate Investment Trusts

### Confusion matrices
Both models show similar patterns, with most predictions on the diagonal. Some off-diagonal errors differ between
Logistic Regression and kNN, so each model also makes unique mistakes.

### Do both models misclassify the same examples?
Mostly yes. Neither model is clearly better overall, but each fixes some of the other's errors, and which one is
better depends on the industry. Combining them therefore helps, which is the starting point of the optional part.

## Optional part: Confidence Weighted Prediction
### Financial data

**Cleaning**
- ±∞ ratios (zero denominator) → NaN, with a flag column (zero revenue is informative, e.g. pre-revenue biotechs).
- Outliers → winsorised at the 1% / 99% quantiles.
- Missing values → kept as NaN (gradient boosting handles them).
- Duplicate ids → first non-missing value per id.

29,400 of 31,706 companies have financial data.

**Distributions across industries:** All three ratios differ significantly between industries (Kruskal–Wallis,
p ≈ 0; H = 2896 net profit margin, 1818 EBITDA margin, 6096 asset turnover). For example, banks and REITs have very
low asset turnover and retailers very high. The distributions overlap heavily, so financial data is only a weak,
complementary signal.

### Distributions across industries in all embedding columns

We ran the same Kruskal–Wallis test on each of the 768 embedding columns, with Benjamini–Hochberg and Bonferroni
correction. The effect size is ε² = H/(n−1), the share of rank variance explained by the industry. The ~4,200
identical placeholder embeddings are left out.

- All 768 columns differ significantly, even with Bonferroni. With ~24,000 companies this is expected, so the effect
  size matters more than the p-value.
- The median column has ε² = 0.19 (max 0.43). 282 columns separate industries better than the best financial ratio
  (asset turnover, ε² ≈ 0.22).
- The signal is spread over many columns. This is why the text embedding is the main signal and the financial
  data is only a correction.

### Embedding the financial data

Ratios → quantile transform + missing indicators, ±∞ flags, country one-hot, then standardised (50 columns). We
compared the embeddings by 5-fold CV accuracy, predicting the industry from financial data only:

| Embedding | CV accuracy |
|---|---|
| majority class | 26.6% |
| PCA, 2 components | 26.6% |
| PCA, 10 components | 31.3% |
| LDA, 10 components | 33.6% |
| all features | 34.3% |
| **gradient boosting probabilities** | **37.7%** |

PCA is a poor embedding here: the variance is spread evenly over the columns, and with 2 components it is no better
than always predicting *Materials*. The supervised embeddings are better. The gradient-boosting probabilities are
the best, so this model is the financial expert in the ensemble.

### Ensemble with calibrated confidence and p-values

1. **Embedding experts:**
   - **MLP ensemble:** 5 MLPs with 512 hidden units, trained from different random starts, with their probabilities
     averaged.
   - **kNN:** cosine distance, k = 15.
   - **LDA-kNN:** kNN in the 24-dimensional LDA space.

   Their probabilities are averaged with weights w.
2. **Financial expert:** gradient boosting on the cleaned ratios + country. It corrects the embedding prediction:
   log P = log P_emb + α (log P_fin − log P_prior).
3. **p-values (conformal):** p_k = the share of calibration companies whose true-class probability is ≤ P(k | x).
   Industries with p_k > 0.1 form a 90% prediction set.
4. **Confidence:** two features go into a logistic regression that outputs the probability the prediction is
   correct:
   - credibility (the largest p-value);
   - conformal confidence (1 − the second-largest p-value).

   A raw p-value is not a probability (used directly it scores ≈ 0.63), so it needs this step.
5. **Uncertain flag:** set when the 90% prediction set contains more than one industry.

`predict()` returns `(y_pred, confidence, p_values, uncertain)`. `y_pred` contains label-encoded industries, as in
the skeleton.

**Avoiding overfitting:** w, α and the calibrator are chosen on 5-fold out-of-fold predictions of the training data.
One half of those predictions gives the conformal scores and the other half fits the calibrator. The validation set
is only used for the final evaluation.

Chosen weights: MLP ensemble 0.5, kNN 0.3, LDA-kNN 0.2, α = 0.25.

### What we tried

Out-of-fold results on the training data:

| Model | Accuracy | Score |
|---|---|---|
| LR / kNN alone | 77.7% / 77.5% | – |
| Single MLP | 76.9% | – |
| LR + kNN (+ financial correction) | 78.9% | 0.686 |
| MLP ensemble alone | 78.4% | – |
| **MLP ensemble + kNN + LDA-kNN (+ financial correction)** | **79.4%** | **0.692** |

These changes moved the score by less than ±0.002, so they are not used:
- other outlier handling (winsorising at 5%/95% or 0.1%/99.9%, removing extreme values);
- removing embedding outliers or likely mislabelled companies;
- more models (kernel LR, linear SVM, random forest);
- 18 calibration methods (Platt, isotonic, beta, temperature, Venn-Abers, Dirichlet, top-label binning,
  meta-calibrators).

Balanced class weights lowered the score by 0.011.

### Confidence calibration

Confidence functions compared on the out-of-fold predictions:

| Confidence | Brier | Score |
|---|---|---|
| raw max probability | 0.1246 | 0.6921 |
| isotonic | 0.1248 | 0.6920 |
| **p-value calibrator** | **0.1246** | **0.6921** |

The p-value calibrator is as good as the best alternative. Brier = uncertainty − resolution + calibration error, and
calibrators only reduce the last term, which is already close to zero. So the score can only improve substantially
through higher accuracy: each percentage point of accuracy is worth about +0.012 score.

### Results (validation set)

| | Accuracy | F1 weighted | Brier | Score |
|---|---|---|---|---|
| LR baseline | – | – | – | 0.667 |
| **Ensemble** | **79.15%** | **0.789** | **0.127** | **0.689** |

- Bootstrap 95% CI of the score: 0.672–0.704, above the 66% threshold.
- The ensemble beats the LR baseline by +0.014 to +0.030 (paired bootstrap) and is better in every resample.
- Out-of-fold (0.692) and validation (0.689) scores are almost the same, so there is no sign of overfitting.

### Confidence vs. accuracy and uncertain predictions

- The confidence is well calibrated (the reliability curve is close to the diagonal), and accuracy rises with the
  p-value.
- The 90% prediction sets contain the true industry for 90.4% of companies, with 1.5 industries on average.

| | Share | Accuracy | Mean confidence |
|---|---|---|---|
| certain (one industry in the 90% set) | 61% | 93.0% | 0.92 |
| uncertain (two or more) | 39% | 57.9% | 0.59 |

### Training time

About 3.5 minutes on the training split and 3.5 minutes on all data, well below the 45-minute limit.