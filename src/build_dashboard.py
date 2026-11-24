"""
Builds an interactive HTML dashboard from the data and model outputs.

    python src/build_dashboard.py      ->  dashboards/churn_dashboard.html
Run src/train_model.py first.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
BLUE, RED, GRAY = "#2E75B6", "#C0504D", "#A6A6A6"

df = pd.read_csv(ROOT / "data" / "telco_customer_churn.csv")
df["Churned"] = (df["Churn"] == "Yes").astype(int)
df["TenureGroup"] = pd.cut(df["tenure"], [-1, 11, 23, 47, 100],
                           labels=["0-11 mo", "12-23 mo", "24-47 mo", "48+ mo"])
metrics = json.loads((OUT / "metrics.json").read_text())
importance = pd.read_csv(OUT / "feature_importance.csv").head(8).iloc[::-1]
scores = pd.read_csv(OUT / "customer_risk_scores.csv")


def rate(col):
    r = df.groupby(col, observed=True)["Churned"].mean().mul(100).round(1)
    return r.sort_values(ascending=False) if col != "TenureGroup" else r


fig = make_subplots(
    rows=3, cols=2, vertical_spacing=0.12, horizontal_spacing=0.14,
    subplot_titles=("Churn rate by contract (%)", "Churn rate by tenure (%)",
                    "Churn rate by payment method (%)", "Top churn drivers (model)",
                    "Active customers by risk tier", "Monthly revenue by risk tier ($)"))

for (col, r, c) in [("Contract", 1, 1), ("TenureGroup", 1, 2), ("PaymentMethod", 2, 1)]:
    s = rate(col)
    fig.add_bar(x=s.index.astype(str), y=s.values, marker_color=BLUE, text=s.values,
                textposition="outside", showlegend=False, row=r, col=c)

fig.add_bar(x=importance["importance"], y=importance["feature"], orientation="h",
            marker_color=BLUE, showlegend=False, row=2, col=2)

tiers = scores.groupby("RiskTier").agg(customers=("customerID", "count"), revenue=("MonthlyCharges", "sum"))
tiers = tiers.reindex(["High", "Medium", "Low"])
colors = [RED, "#E6A23C", GRAY]
fig.add_bar(x=tiers.index, y=tiers["customers"], marker_color=colors, text=tiers["customers"],
            textposition="outside", showlegend=False, row=3, col=1)
fig.add_bar(x=tiers.index, y=tiers["revenue"].round(0), marker_color=colors,
            text=[f"${v:,.0f}" for v in tiers["revenue"]], textposition="outside", showlegend=False, row=3, col=2)

for r in (1, 2, 3):
    for c in (1, 2):
        fig.update_yaxes(rangemode="tozero", automargin=True, row=r, col=c)
fig.update_yaxes(range=[0, 60], row=1, col=1)
fig.update_yaxes(range=[0, 60], row=1, col=2)
fig.update_yaxes(range=[0, 55], row=2, col=1)
fig.update_layout(height=1150, template="simple_white", margin=dict(t=40, l=40, r=30, b=40),
                  font=dict(family="Arial", size=12))

m = metrics
kpis = [
    ("Overall churn rate", f"{df['Churned'].mean() * 100:.1f}%"),
    ("Model ROC-AUC (test)", f"{m['roc_auc']:.2f}"),
    ("Churners caught (recall)", f"{m['recall'] * 100:.0f}%"),
    ("Top-20% lift", f"{m['top20pct_lift']:.1f}x"),
    ("High-risk active customers", f"{m['risk_tiers']['High']['customers']:,}"),
]
kpi_html = "".join(f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>' for l, v in kpis)

html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Customer Churn Dashboard</title>
<style>
 body{{font-family:Arial,sans-serif;margin:0;background:#f5f7fa;color:#222}}
 header{{background:#1F4E79;color:#fff;padding:20px 32px}}
 header h1{{margin:0;font-size:24px}} header p{{margin:6px 0 0;opacity:.85}}
 .kpis{{display:flex;flex-wrap:wrap;gap:12px;padding:20px 32px}}
 .kpi{{background:#fff;border-radius:8px;padding:14px 18px;flex:1;min-width:150px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
 .kpi .v{{font-size:26px;font-weight:bold;color:#1F4E79}} .kpi .l{{font-size:13px;color:#666;margin-top:4px}}
 .card{{background:#fff;margin:0 32px 32px;border-radius:8px;padding:12px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
</style></head><body>
<header><h1>Customer Churn Dashboard</h1>
<p>IBM Telco dataset, 7,043 customers. Model: {m['chosen_model']}, scored on a held-out test set of {m['test_customers']:,} customers.</p></header>
<div class="kpis">{kpi_html}</div>
<div class="card">{fig.to_html(full_html=False, include_plotlyjs=True)}</div>
</body></html>"""
(ROOT / "dashboards").mkdir(exist_ok=True)
(ROOT / "dashboards" / "churn_dashboard.html").write_text(html)
print("Wrote dashboards/churn_dashboard.html")
