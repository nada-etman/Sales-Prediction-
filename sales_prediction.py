"""
CodeAlpha - Data Science Internship
Task 4: Sales Prediction using Python
-----------------------------------------
Predict product sales based on advertising spend across TV, Radio and
Newspaper, and analyze how each platform impacts sales outcomes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------
df = pd.read_csv("Advertising.csv", index_col=0)
print("Dataset shape:", df.shape)
print(df.head())
print(df.describe())
print(df.isna().sum())

# ---------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------
plt.figure(figsize=(5, 4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Between Advertising Spend and Sales")
plt.tight_layout()
plt.savefig("sales_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["TV", "Radio", "Newspaper"]):
    sns.regplot(x=col, y="Sales", data=df, ax=ax,
                scatter_kws={"alpha": 0.6}, line_kws={"color": "red"})
    ax.set_title(f"{col} Spend vs Sales")
plt.tight_layout()
plt.savefig("sales_vs_advertising_platforms.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 4. Train and compare models
# ---------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    results[name] = {"R2": r2, "MAE": mae, "RMSE": rmse}
    print(f"\n{name}:")
    print(f"  R2 Score: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")

# ---------------------------------------------------------
# 5. Linear regression coefficients (impact of each platform)
# ---------------------------------------------------------
lin_model = models["Linear Regression"]
coefs = pd.Series(lin_model.coef_, index=X.columns).sort_values(ascending=False)
print("\nLinear Regression Coefficients (impact per unit spend):")
print(coefs)
print(f"Intercept: {lin_model.intercept_:.4f}")

plt.figure(figsize=(5, 4))
coefs.plot(kind="bar", color="purple")
plt.title("Impact of Advertising Platform on Sales (Linear Coefficients)")
plt.ylabel("Coefficient")
plt.tight_layout()
plt.savefig("sales_platform_impact.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# 6. Best model - actual vs predicted plot
# ---------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["R2"])
best_model = models[best_name]
best_preds = best_model.predict(X_test)
print(f"\nBest model: {best_name} (R2={results[best_name]['R2']:.4f})")

plt.figure(figsize=(6, 6))
plt.scatter(y_test, best_preds, alpha=0.6, color="darkblue")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title(f"Actual vs Predicted Sales ({best_name})")
plt.tight_layout()
plt.savefig("sales_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n=== Model Comparison Summary ===")
for name, m in results.items():
    print(f"{name}: R2={m['R2']:.4f}, MAE={m['MAE']:.4f}, RMSE={m['RMSE']:.4f}")
