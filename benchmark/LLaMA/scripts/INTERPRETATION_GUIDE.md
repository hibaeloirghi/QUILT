# Interpretation Guide: Accuracy and Entropy Analysis

## Overview

This analysis compares two experimental conditions:
- **No-Sampling**: Standard ReAct agent without action sampling
- **Samples5**: ReAct agent with 5 action samples per step, selecting the action with lowest entropy

## Key Metrics Explained

### 1. Accuracy Metrics

**What it measures**: The proportion of questions answered correctly.

**How to interpret**:
- Higher accuracy = better performance
- If both conditions have similar accuracy, sampling doesn't improve correctness
- If sampling improves accuracy, it suggests entropy-based selection helps

**From your results**:
- Both conditions: **37.5% accuracy (3/8 correct)**
- **Interpretation**: No difference in overall accuracy. Sampling doesn't improve correctness in this small sample.

---

### 2. Predictive Entropy H(Y|Z,x)

**What it measures**: Uncertainty in the model's answer predictions, given the tool outputs Z and question x.

**Formula**: H(Y|Z,x) = -1/N × Σ log p(y_i | Z, x)

**How to interpret**:
- **Lower entropy** = Model is more confident/consistent in its answers
- **Higher entropy** = Model is uncertain/inconsistent across samples
- **Correct answers** should ideally have lower entropy (model is confident when right)
- **Incorrect answers** may have higher entropy (model is uncertain when wrong)

**From your results**:
- **No-Sampling**: Mean = 5.29 (Correct: 3.08, Incorrect: 6.61)
- **Samples5**: Mean = 11.19 (Correct: 7.63, Incorrect: 13.21)

**Interpretation**:
- Samples5 has **higher entropy overall**, suggesting more uncertainty
- However, this is expected because sampling generates diverse outputs
- The key question: Do correct answers have lower entropy than incorrect ones?
  - **No-Sampling**: Correct (3.08) < Incorrect (6.61) ✓ Good separation
  - **Samples5**: Correct (7.63) < Incorrect (13.21) ✓ Good separation
- **Conclusion**: Entropy can distinguish correct from incorrect answers in both conditions, but Samples5 shows better separation (larger gap).

---

### 3. Semantic Entropy H_c(Y|Z,x)

**What it measures**: Uncertainty in the *meaning* of answers (clusters semantically similar answers together).

**How to interpret**:
- **Lower semantic entropy** = Answers are semantically similar (even if worded differently)
- **Higher semantic entropy** = Answers have different meanings
- This is more robust than predictive entropy because it groups equivalent answers

**From your results**:
- **No-Sampling**: Mean = 1.30 (Correct: 0.82, Incorrect: 1.71)
- **Samples5**: Mean = 1.30 (Correct: 0.99, Incorrect: 1.53)

**Interpretation**:
- Both conditions have similar semantic entropy
- Correct answers have lower semantic entropy (more consistent meaning)
- **Conclusion**: Semantic entropy is a good indicator of correctness, and both conditions perform similarly.

---

### 4. STA (Structured Task-Aware) Entropy

**What it measures**: Total uncertainty including both answer uncertainty and tool uncertainty.

**Formulas**:
- **STA-Predictive**: H(Y|Z,x) + H(Z|a) = Predictive entropy + Tool entropy
- **STA-Semantic**: H_c(Y|Z,x) + H(Z|a) = Semantic entropy + Tool entropy

**How to interpret**:
- Captures uncertainty from both the model's predictions AND tool outputs
- Higher STA = More uncertainty overall
- In your case, tool entropy is 0 (deterministic tools), so STA = answer entropy

**From your results**:
- Same as predictive/semantic entropy (no tool entropy)

---

### 5. Confidence Metrics (Answer Probability)

**What it measures**: The probability assigned to the generated answer sequence.

**Metrics**:
- **Average Probability**: Mean probability across all answer samples
- **Minimum Probability**: Lowest probability among samples
- **Maximum Probability**: Highest probability among samples

**How to interpret**:
- **Higher probability** = Model is more confident in the answer
- **Correct answers** should ideally have higher probability
- **Incorrect answers** may have lower probability

**From your results**:
- **No-Sampling (Correct)**: Avg prob = 0.72, Min = 0.28, Max = 0.94
- **No-Sampling (Incorrect)**: Avg prob = 0.16, Min = 0.00001, Max = 0.51
- **Samples5 (Correct)**: Avg prob = 0.35, Min = 0.0002, Max = 0.63
- **Samples5 (Incorrect)**: Avg prob = 0.16, Min = 0.0000001, Max = 0.50

**Interpretation**:
- **No-Sampling** shows better confidence separation: Correct (0.72) >> Incorrect (0.16)
- **Samples5** shows weaker separation: Correct (0.35) ≈ Incorrect (0.16)
- **Conclusion**: No-Sampling's confidence better predicts correctness. Samples5's lower confidence may be due to sampling diversity.

---

### 6. Step-wise Entropy

**What it measures**: Entropy at each reasoning step (Thought/Action).

**How to interpret**:
- **Lower step entropy** = Model is confident at that step
- **Higher step entropy** = Model is uncertain at that step
- Can identify which steps are most uncertain

**From your results**:
- **No-Sampling**: Mean = 4.85
- **Samples5**: Mean = 5.12

**Interpretation**:
- Similar step-wise entropy
- **Conclusion**: Both conditions show similar uncertainty during reasoning steps.

---

### 7. Action Entropy (Selected)

**What it measures**: Entropy of the selected action (lowest entropy among 5 samples in Samples5).

**How to interpret**:
- **Lower action entropy** = Selected action is more confident
- **Samples5** should have lower action entropy (by design - selects lowest)
- **No-Sampling** has no selection, so entropy reflects single generation

**From your results**:
- **No-Sampling**: Mean = 5.23
- **Samples5**: Mean = 3.18

**Interpretation**:
- **Samples5 has lower action entropy** ✓ This is expected - it selects the most confident action
- **Conclusion**: Entropy-based action selection is working as intended.

---

## Key Findings Summary

### 1. **Accuracy**: No difference (37.5% both)
   - Sampling doesn't improve correctness in this small sample

### 2. **Predictive Entropy**: Better separation in Samples5
   - Correct (7.63) vs Incorrect (13.21) - larger gap than No-Sampling
   - Suggests entropy is a better predictor when using sampling

### 3. **Semantic Entropy**: Similar performance
   - Both conditions show good separation between correct/incorrect

### 4. **Confidence**: Better in No-Sampling
   - No-Sampling shows clearer confidence separation
   - Samples5's lower confidence may be due to sampling diversity

### 5. **Action Selection**: Working as intended
   - Samples5 selects lower-entropy actions (3.18 vs 5.23)
   - This is the expected behavior

---

## Recommendations

### When to Use Sampling:
1. **If you want better uncertainty estimation** - Sampling provides more reliable entropy estimates
2. **If you want to select more confident actions** - Lower action entropy suggests better choices
3. **If you're building uncertainty-aware systems** - Better entropy separation helps identify uncertain cases

### When Not to Use Sampling:
1. **If computational cost is a concern** - Sampling requires 5x more model calls
2. **If you need higher confidence scores** - No-Sampling shows clearer confidence separation
3. **If accuracy is the only concern** - No improvement in this small sample

### Future Analysis:
1. **Larger sample size** - 8 questions is small; need more data for statistical significance
2. **Per-question analysis** - Some questions may benefit more from sampling
3. **Error analysis** - Understand why incorrect answers have high entropy
4. **Action quality** - Does lower action entropy correlate with better tool usage?

---

## Understanding the Plots

### Plot 1: Accuracy Comparison
- Simple bar chart showing accuracy for both conditions
- **Look for**: Clear difference in heights

### Plot 2: Entropy Metrics Distribution
- Box plots showing distribution of entropy values
- **Look for**: Differences in median, spread, outliers
- **Key insight**: Samples5 has higher entropy (more uncertainty) but better separation

### Plot 3: Entropy by Correctness
- Box plots split by correct/incorrect
- **Look for**: Clear separation between correct (lower) and incorrect (higher)
- **Key insight**: Both conditions show separation, Samples5 has larger gap

### Plot 4: Confidence Metrics
- Box plots of answer probabilities (log scale)
- **Look for**: Higher probabilities for correct answers
- **Key insight**: No-Sampling shows better confidence separation

### Plot 5: Entropy vs Correctness Scatter
- Scatter plot showing entropy for each question
- **Look for**: Clustering of correct (top) vs incorrect (bottom)
- **Key insight**: Visual confirmation of entropy separation

### Plot 6: Step Entropy Distribution
- Histogram of entropy at each reasoning step
- **Look for**: Differences in uncertainty during reasoning
- **Key insight**: Similar distributions suggest similar reasoning patterns

### Plot 7: Action Entropy Comparison
- Box plot of selected action entropy
- **Look for**: Lower entropy in Samples5 (by design)
- **Key insight**: Confirms entropy-based selection is working

---

## Statistical Significance

**Note**: With only 8 questions, these results are not statistically significant. The patterns are suggestive but need larger sample sizes to confirm.

**For publication/analysis**:
- Need at least 30-50 questions per condition
- Use statistical tests (t-test, Mann-Whitney U) to confirm differences
- Report confidence intervals for accuracy
- Consider effect sizes, not just p-values

---

## Questions to Explore Further

1. **Why does sampling increase overall entropy but improve separation?**
   - Sampling reveals uncertainty that deterministic generation hides
   - Better separation suggests entropy is more informative with sampling

2. **Why doesn't lower action entropy improve accuracy?**
   - Lower entropy actions may be more confident but not necessarily correct
   - Need to analyze action quality, not just confidence

3. **Can we use entropy to predict correctness?**
   - Yes, but threshold needs tuning
   - Semantic entropy may be more reliable than predictive entropy

4. **What about the computational cost?**
   - 5x more model calls for Samples5
   - Need to weigh benefits vs costs

