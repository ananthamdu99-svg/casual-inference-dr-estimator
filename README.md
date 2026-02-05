# Doubly Robust Estimator for Causal Inference

This project demonstrates estimation of Average Treatment Effect (ATE) and
Conditional Average Treatment Effect (CATE) using a Doubly Robust (DR) estimator.

## Concepts Covered
- Treatment effect heterogeneity
- Naive ATE vs Doubly Robust ATE
- Propensity score modeling
- Outcome regression

## Technologies
- Python
- NumPy
- Pandas
- Scikit-learn

## How to Run
```bash
python dr_estimator.py
Sample Output
TRUE ATE: 3.32
Baseline ATE: 4.55
DR ATE: 3.38
Baseline Bias: 1.23
DR Bias: 0.07
CATE (age < 30): 6.03
CATE (age ≥ 30): 2.09

## Author
Ananthalakshmi Ponnaiah
MCA Graduate



