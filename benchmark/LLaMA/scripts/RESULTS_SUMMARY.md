# Results Summary: No-Sampling vs Samples5

## Executive Summary

**Key Finding**: Both conditions achieve **37.5% accuracy (3/8 correct)**, but **Samples5 shows better entropy separation** between correct and incorrect answers, suggesting entropy-based action selection is working as intended.

---

## Detailed Findings

### 1. Accuracy: No Difference
- **No-Sampling**: 3/8 correct (37.5%)
- **Samples5**: 3/8 correct (37.5%)
- **Conclusion**: Sampling doesn't improve accuracy in this small sample

### 2. Predictive Entropy: Better Separation in Samples5

| Condition | Overall Mean | Correct Mean | Incorrect Mean | Gap |
|-----------|-------------|-------------|----------------|-----|
| No-Sampling | 5.29 | 3.08 | 6.61 | 3.53 |
| Samples5 | 11.19 | 7.63 | 13.21 | **5.58** |

**Key Insight**: 
- Samples5 has larger gap between correct/incorrect (5.58 vs 3.53)
- This suggests entropy is a better predictor when using sampling
- Higher overall entropy in Samples5 is expected (sampling reveals uncertainty)

### 3. Semantic Entropy: Similar Performance

| Condition | Overall Mean | Correct Mean | Incorrect Mean | Gap |
|-----------|-------------|-------------|----------------|-----|
| No-Sampling | 1.30 | 0.82 | 1.71 | 0.89 |
| Samples5 | 1.30 | 0.99 | 1.53 | 0.54 |

**Key Insight**:
- Both conditions show good separation
- No-Sampling has slightly better separation (0.89 vs 0.54)
- Semantic entropy is more stable across conditions

### 4. Confidence (Answer Probability): Better in No-Sampling

| Condition | Correct Avg Prob | Incorrect Avg Prob | Ratio |
|-----------|------------------|---------------------|-------|
| No-Sampling | **0.72** | 0.16 | **4.5x** |
| Samples5 | 0.35 | 0.16 | 2.2x |

**Key Insight**:
- No-Sampling shows much better confidence separation (4.5x vs 2.2x)
- Correct answers in No-Sampling have higher confidence (0.72 vs 0.35)
- Samples5's lower confidence may be due to sampling diversity

### 5. Action Entropy: Lower in Samples5 (As Expected)

| Condition | Mean Action Entropy |
|-----------|---------------------|
| No-Sampling | 5.23 |
| Samples5 | **3.18** |

**Key Insight**:
- Samples5 selects lower-entropy actions (by design)
- This confirms entropy-based selection is working
- Lower entropy doesn't necessarily mean better accuracy

### 6. Step-wise Entropy: Similar

| Condition | Mean Step Entropy |
|-----------|-------------------|
| No-Sampling | 4.85 |
| Samples5 | 5.12 |

**Key Insight**:
- Similar uncertainty during reasoning steps
- Both conditions show similar reasoning patterns

---

## Interpretation

### What Works Well

1. **Entropy-based action selection**: Samples5 successfully selects lower-entropy actions
2. **Entropy separation**: Both conditions show entropy can distinguish correct/incorrect
3. **Semantic entropy**: More stable and reliable than predictive entropy

### What Needs Improvement

1. **Accuracy**: No improvement from sampling (but small sample size)
2. **Confidence**: Samples5 has lower confidence scores (may be due to sampling)
3. **Computational cost**: 5x more model calls for Samples5

### Key Questions

1. **Why doesn't lower action entropy improve accuracy?**
   - Lower entropy = more confident, but not necessarily correct
   - Need to analyze action quality, not just confidence

2. **Why is confidence lower in Samples5?**
   - Sampling generates diverse outputs, reducing average probability
   - But entropy separation is better, suggesting better uncertainty estimation

3. **Is the entropy separation useful?**
   - Yes! Can use entropy to identify uncertain cases
   - Better separation in Samples5 suggests it's more informative

---

## Recommendations

### Use Sampling When:
- ✅ You need better uncertainty estimation
- ✅ You want to identify uncertain cases
- ✅ You're building uncertainty-aware systems
- ✅ Computational cost is acceptable

### Use No-Sampling When:
- ✅ Computational cost is a concern
- ✅ You need higher confidence scores
- ✅ Accuracy is the only concern
- ✅ You want simpler, faster inference

### For Future Analysis:
1. **Larger sample size** (30-50+ questions) for statistical significance
2. **Per-question analysis** to identify which questions benefit from sampling
3. **Error analysis** to understand why incorrect answers have high entropy
4. **Action quality analysis** to see if lower entropy correlates with better tool usage

---

## Statistical Note

**Important**: With only 8 questions, these results are **not statistically significant**. The patterns are suggestive but need larger sample sizes to confirm.

For publication-quality analysis:
- Need at least 30-50 questions per condition
- Use statistical tests (t-test, Mann-Whitney U) to confirm differences
- Report confidence intervals
- Consider effect sizes, not just p-values

---

## Visualizations Generated

The analysis script generates 7 plots:

1. **Accuracy Comparison**: Bar chart showing accuracy for both conditions
2. **Entropy Metrics Distribution**: Box plots of all entropy metrics
3. **Entropy by Correctness**: Box plots split by correct/incorrect
4. **Confidence Metrics**: Box plots of answer probabilities (log scale)
5. **Entropy vs Correctness**: Scatter plot showing entropy for each question
6. **Step Entropy Distribution**: Histogram of step-wise entropies
7. **Action Entropy Comparison**: Box plot of selected action entropy

All plots are saved as high-resolution PNG files (300 DPI) in the `plots/` directory.

