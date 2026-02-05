import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression

np.random.seed(2)

n = 2000
age = np.random.randint(20, 50, n)
experience = np.random.randint(0, 15, n)

true_te = np.where(age < 30, 6, 2)
TRUE_ATE = true_te.mean()

logit = -3 + 0.08 * age + 0.2 * experience
ps_true = 1 / (1 + np.exp(-logit))
treatment = np.random.binomial(1, ps_true)

outcome = 30 + 0.5 * experience + treatment * true_te + np.random.normal(0, 1, n)

data = pd.DataFrame({
    "age": age,
    "experience": experience,
    "treatment": treatment,
    "outcome": outcome
})

baseline_ate = (
    data[data.treatment == 1].outcome.mean()
    - data[data.treatment == 0].outcome.mean()
)
baseline_bias = abs(baseline_ate - TRUE_ATE)

ps_model = LinearRegression()  # WRONG on purpose
ps_model.fit(data[['age','experience']], treatment)
ps = ps_model.predict(data[['age','experience']])
ps = np.clip(ps, 0.05, 0.95)


out_model = LinearRegression()
out_model.fit(data[['age','experience','treatment']], outcome)

X1 = data[['age','experience']].copy()
X1['treatment'] = 1
X0 = data[['age','experience']].copy()
X0['treatment'] = 0

mu1 = out_model.predict(X1)
mu0 = out_model.predict(X0)

T = treatment
Y = outcome

dr_scores = (
    mu1 - mu0
    + (T/ps)*(Y - mu1)
    - ((1-T)/(1-ps))*(Y - mu0)
)

dr_ate = dr_scores.mean()
dr_bias = abs(dr_ate - TRUE_ATE)

cate_young = dr_scores[age < 30].mean()
cate_old = dr_scores[age >= 30].mean()

print("TRUE ATE:", round(TRUE_ATE,2))
print("Baseline ATE:", round(baseline_ate,2))
print("DR ATE:", round(dr_ate,2))
print("Baseline Bias:", round(baseline_bias,2))
print("DR Bias:", round(dr_bias,2))
print("CATE (age < 30):", round(cate_young,2))
print("CATE (age >= 30):", round(cate_old,2))
