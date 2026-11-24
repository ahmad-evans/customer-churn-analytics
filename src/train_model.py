"""
Trains and compares churn prediction models, then scores every customer.

    python src/train_model.py

Outputs:
    outputs/model_comparison.csv   cross-validated scores for each model
    outputs/metrics.json           final test-set results for the chosen model
    outputs/feature_importance.csv what drives churn, ranked
    outputs/customer_risk_scores.csv  churn probability + risk tier per active customer
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
SEED = 42

# ---------------------------------------------------------------- 1. Load + clean
df = pd.read_csv(ROOT / "data" / "telco_customer_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)  # 11 new customers, not billed yet
df["Churned"] = (df["Churn"] == "Yes").astype(int)

# ---------------------------------------------------------------- 2. Feature engineering
addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
df["NumAddons"] = (df[addons] == "Yes").sum(axis=1)
df["AutoPay"] = df["PaymentMethod"].str.contains("automatic").astype(int)
df["AvgMonthlySpend"] = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"])
df["NewCustomer"] = (df["tenure"] < 12).astype(int)

numeric = ["tenure", "MonthlyCharges", "TotalCharges", "NumAddons", "AvgMonthlySpend", "SeniorCitizen", "AutoPay", "NewCustomer"]
categorical = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
               "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
               "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
X, y = df[numeric + categorical], df["Churned"]

# 80/20 split, stratified so both sets keep the same ~26.5% churn rate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

prep = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical),
])

# ---------------------------------------------------------------- 3. Compare models with 5-fold CV
# Models are left unweighted so their probabilities stay realistic (needed for revenue-at-risk).
# The smaller churn class is handled later by tuning the decision threshold instead of using 0.5.
# ROC-AUC is the main comparison metric because it doesn't depend on any threshold.
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_depth=4,
                                                        random_state=SEED),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rows = []
for name, model in models.items():
    pipe = Pipeline([("prep", prep), ("model", model)])
    s = cross_validate(pipe, X_train, y_train, cv=cv, scoring=["roc_auc", "neg_brier_score"])
    rows.append({"model": name,
                 "roc_auc": s["test_roc_auc"].mean(), "roc_auc_std": s["test_roc_auc"].std(),
                 "brier_score": -s["test_neg_brier_score"].mean()})
comparison = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).round(3)
comparison.to_csv(OUT / "model_comparison.csv", index=False)
print("\nCross-validated comparison (training data only):\n", comparison.to_string(index=False))

# ---------------------------------------------------------------- 4. Pick a decision threshold (training data only)
best_name = comparison.iloc[0]["model"]
best_pipe = Pipeline([("prep", prep), ("model", models[best_name])])
# Out-of-fold probabilities: each training customer is scored by a model that never saw them
oof = cross_val_predict(best_pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
prec, rec, thr = precision_recall_curve(y_train, oof)
f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
threshold = float(thr[f1s.argmax()])  # threshold with the best balance of catching churners vs. false alarms
print(f"\nChosen decision threshold: {threshold:.3f} (default would be 0.5)")

# ---------------------------------------------------------------- 5. Fit the best model, test ONCE on held-out data
best = best_pipe.fit(X_train, y_train)
proba = best.predict_proba(X_test)[:, 1]
pred = (proba >= threshold).astype(int)
pred_default = (proba >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

# Baseline to beat: "everyone on a month-to-month contract will churn"
baseline = (X_test["Contract"] == "Month-to-month").astype(int)

# Lift: how much more churn is packed into the top 20% riskiest customers vs. random
top20 = proba >= np.quantile(proba, 0.8)
lift = y_test[top20].mean() / y_test.mean()

metrics = {
    "chosen_model": best_name,
    "test_customers": int(len(y_test)),
    "decision_threshold": round(threshold, 3),
    "roc_auc": round(roc_auc_score(y_test, proba), 3),
    "accuracy": round(accuracy_score(y_test, pred), 3),
    "recall": round(recall_score(y_test, pred), 3),
    "precision": round(precision_score(y_test, pred), 3),
    "f1": round(f1_score(y_test, pred), 3),
    "brier_score": round(brier_score_loss(y_test, proba), 3),
    "at_default_0.5_threshold": {"recall": round(recall_score(y_test, pred_default), 3),
                                 "precision": round(precision_score(y_test, pred_default), 3),
                                 "f1": round(f1_score(y_test, pred_default), 3)},
    "confusion_matrix": {"true_neg": int(tn), "false_pos": int(fp), "false_neg": int(fn), "true_pos": int(tp)},
    "baseline_month_to_month_rule": {
        "recall": round(recall_score(y_test, baseline), 3),
        "precision": round(precision_score(y_test, baseline), 3),
        "f1": round(f1_score(y_test, baseline), 3)},
    "top20pct_churn_rate": round(float(y_test[top20].mean()), 3),
    "overall_test_churn_rate": round(float(y_test.mean()), 3),
    "top20pct_lift": round(float(lift), 2),
}
print("\nHeld-out test results:\n", json.dumps(metrics, indent=2))

# ---------------------------------------------------------------- 6. What drives churn? (permutation importance)
# Shuffle one column at a time and see how much ROC-AUC drops. Bigger drop = more important.
imp = permutation_importance(best, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=SEED, n_jobs=-1)
importance = (pd.DataFrame({"feature": X.columns, "importance": imp.importances_mean, "std": imp.importances_std})
              .sort_values("importance", ascending=False).round(4))
importance.to_csv(OUT / "feature_importance.csv", index=False)
print("\nTop drivers:\n", importance.head(8).to_string(index=False))

# ---------------------------------------------------------------- 7. Score every customer
# Out-of-fold scoring over the full dataset: no customer is scored by a model that trained on them.
df["ChurnProbability"] = cross_val_predict(best_pipe, X, y, cv=cv, method="predict_proba")[:, 1].round(3)
# Tiers: Low = under the decision threshold, Medium = threshold up to 50%, High = 50%+ chance of leaving
df["RiskTier"] = pd.cut(df["ChurnProbability"], [-0.01, threshold, 0.5, 1.0], labels=["Low", "Medium", "High"])
# Keep only customers still active: they're the ones a retention team can still save
active = df[df["Churned"] == 0].copy()
active["MonthlyRevenueAtRisk"] = (active["MonthlyCharges"] * active["ChurnProbability"]).round(2)

cols = ["customerID", "Contract", "tenure", "InternetService", "TechSupport", "PaymentMethod",
        "MonthlyCharges", "NumAddons", "ChurnProbability", "RiskTier", "MonthlyRevenueAtRisk"]
active[cols].sort_values("ChurnProbability", ascending=False).to_csv(OUT / "customer_risk_scores.csv", index=False)

tiers = active.groupby("RiskTier", observed=True).agg(customers=("customerID", "count"),
                                                      monthly_revenue=("MonthlyCharges", "sum"))
metrics["active_customers"] = int(len(active))
metrics["risk_tiers"] = {t: {"customers": int(r.customers), "monthly_revenue": round(float(r.monthly_revenue), 2)}
                         for t, r in tiers.iterrows()}
metrics["expected_monthly_revenue_at_risk"] = round(float(active["MonthlyRevenueAtRisk"].sum()), 2)
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
print("\nActive customers by risk tier:\n", tiers.round(2))
print("Expected monthly revenue at risk: $", metrics["expected_monthly_revenue_at_risk"])
