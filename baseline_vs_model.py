"""
What actually sets the price in this dataset — and what the model adds on top.

The original write-up said this project predicts used EV prices "from battery
capacity and vehicle specifications". That is half right, and the half it gets
wrong is the interesting one.

Price in this dataset is set almost entirely by which model the car is. Twenty-
one model names span 18 to 158 million won, while prices *within* a model vary
by a median of 2.8. Looking up the mean price of the car's model — no learning
at all — already explains 98.8% of the variance.

So the honest way to report this project is as a ladder, not a single number:

    mean price of every car           RMSE 36.6   (predict the global mean)
    mean price of that model          RMSE  4.09  (a 21-row lookup table)
    RandomForest on every feature     RMSE  1.53  (what this repo builds)

The specification variables do matter, but at the second decimal place: they
explain variation *within* a model, not *across* models. That is where battery
capacity, warranty and mileage earn their place, and this script measures it.

Run:  python baseline_vs_model.py
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import LabelEncoder

SEED = 42
TARGET = "가격(백만원)"
NUMERIC = ["주행거리(km)", "보증기간(년)", "연식(년)"]
CATEGORICAL = ["제조사", "차량상태", "구동방식", "사고이력"]


def load():
    """Same preprocessing as main.ipynb, kept deliberately identical."""
    raw = pd.read_csv("train.csv")
    df = raw.drop(columns=["ID"]).copy()

    missing_battery = int(df["배터리용량"].isna().sum())

    df[NUMERIC] = KNNImputer(n_neighbors=5).fit_transform(df[NUMERIC])
    df[CATEGORICAL] = SimpleImputer(strategy="most_frequent").fit_transform(df[CATEGORICAL])
    df["배터리용량"] = df["배터리용량"].fillna(df["배터리용량"].median())

    model_names = df["모델"].astype(str)
    for column in ["제조사", "모델", "차량상태", "구동방식", "사고이력"]:
        df[column] = LabelEncoder().fit_transform(df[column].astype(str))

    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    return X, y, model_names, missing_battery


def cv_rmse(X, y, columns):
    scores = -cross_val_score(
        RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
        X[columns], y,
        cv=KFold(5, shuffle=True, random_state=SEED),
        scoring="neg_root_mean_squared_error",
    )
    return scores.mean(), scores.std()


def lookup_rmse(y, keys):
    """Out-of-fold group-mean prediction: a lookup table, scored honestly."""
    predictions = np.zeros(len(y))
    for train_index, test_index in KFold(5, shuffle=True, random_state=SEED).split(y):
        means = y.iloc[train_index].groupby(keys.iloc[train_index]).mean()
        fallback = y.iloc[train_index].mean()
        predictions[test_index] = keys.iloc[test_index].map(means).fillna(fallback).values
    return float(np.sqrt(np.mean((y.values - predictions) ** 2)))


def plot(table, ladder, y):
    """Left: price is set by model name. Right: what each step of the ladder buys."""
    fig, (ax_models, ax_ladder) = plt.subplots(
        1, 2, figsize=(13.5, 6), gridspec_kw={"width_ratios": [1.35, 1]}
    )

    positions = np.arange(len(table))
    ax_models.barh(positions, table["mean"], xerr=table["std"], height=0.68,
                   color="#2f6f8f", ecolor="#a8453f", capsize=3)
    ax_models.set_yticks(positions)
    ax_models.set_yticklabels(table.index, fontsize=9)
    ax_models.set_xlabel("price (million won)")
    ax_models.set_title("Price by model name\n"
                        "red bars are the spread within each model", fontsize=11)
    ax_models.grid(axis="x", alpha=0.3)
    ax_models.set_axisbelow(True)

    labels = [label for label, _ in ladder]
    values = [value for _, value in ladder]
    colours = ["#b0b7bf", "#a8632a", "#a8632a", "#0e7c86"]
    bars = ax_ladder.barh(np.arange(len(values)), values, height=0.6, color=colours)
    ax_ladder.set_yticks(np.arange(len(values)))
    ax_ladder.set_yticklabels(labels, fontsize=9)
    ax_ladder.invert_yaxis()
    ax_ladder.set_xlabel("5-fold CV RMSE (million won, lower is better)")
    ax_ladder.set_title("What each step buys\n"
                        "most of the way there without any learning", fontsize=11)
    ax_ladder.grid(axis="x", alpha=0.3)
    ax_ladder.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax_ladder.text(bar.get_width() + 0.6, bar.get_y() + bar.get_height() / 2,
                       f"{value:.2f}", va="center", fontsize=9.5)
    ax_ladder.set_xlim(0, max(values) * 1.16)

    fig.tight_layout()
    out = pathlib.Path(__file__).parent / "baseline_vs_model.png"
    fig.savefig(out, dpi=130)
    print(f"\nFigure written to {out.name}")


def main():
    X, y, model_names, missing_battery = load()
    print(f"{len(y):,} cars, {X.shape[1]} features, "
          f"price {y.mean():.1f} +/- {y.std():.1f} million won")
    print(f"battery capacity missing for {missing_battery:,} rows "
          f"({missing_battery / len(y):.0%}), median-imputed\n")

    print("The ladder  (5-fold CV RMSE, million won)")
    baseline = float(y.std())
    ladder = [("predict the global mean", baseline)]
    print(f"  {'global mean':38s} {baseline:6.2f}")
    for keys, label, short in (
        (X['제조사'].astype(str), "manufacturer lookup table", "manufacturer lookup"),
        (model_names, "model-name lookup table", "model-name lookup"),
    ):
        value = lookup_rmse(y, keys)
        ladder.append((short, value))
        print(f"  {label:38s} {value:6.2f}   R2 = {1 - (value / baseline) ** 2:.4f}")
    for columns, label in (
        (["모델"], "RandomForest, model name only"),
        ([c for c in X.columns if c != "모델"], "RandomForest, everything except model"),
        (list(X.columns), "RandomForest, all features"),
    ):
        mean, std = cv_rmse(X, y, columns)
        if label.endswith("all features"):
            ladder.append(("RandomForest, all features", mean))
        print(f"  {label:38s} {mean:6.2f} +/- {std:.2f}   R2 = {1 - (mean / baseline) ** 2:.4f}")

    print("\nPrice by model")
    grouped = y.groupby(model_names)
    table = pd.DataFrame({
        "n": grouped.size(), "mean": grouped.mean(), "std": grouped.std(),
    }).sort_values("mean")
    table["spread %"] = 100 * table["std"] / table["mean"]
    print(table.round(2).to_string())
    print(f"\n  within-model spread, median {table['std'].median():.2f} "
          f"against an overall spread of {y.std():.2f}")
    print("  Knowing the model name pins the price to within a few million won.")

    print("\nWhat explains the rest")
    residual = y - grouped.transform("mean")
    others = [c for c in X.columns if c != "모델"]
    predicted = cross_val_predict(
        RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
        X[others], residual, cv=KFold(5, shuffle=True, random_state=SEED),
    )
    residual_rmse = float(np.sqrt(np.mean((residual - predicted) ** 2)))
    residual_r2 = 1 - np.sum((residual - predicted) ** 2) / np.sum((residual - residual.mean()) ** 2)
    print(f"  residual spread after removing the model mean: {residual.std():.2f}")
    print(f"  the remaining features predict it to RMSE {residual_rmse:.2f}  (R2 = {residual_r2:.3f})")
    print("  So the specification variables are real — they just operate inside a")
    print("  model, not across models. That is the claim the README should make.")

    print("\nPermutation importance (increase in RMSE when a feature is shuffled)")
    forest = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1).fit(X, y)
    importance = permutation_importance(
        forest, X, y, n_repeats=5, random_state=SEED, n_jobs=-1,
        scoring="neg_root_mean_squared_error",
    )
    for index in importance.importances_mean.argsort()[::-1]:
        print(f"  {X.columns[index]:12s} {importance.importances_mean[index]:7.2f} "
              f"+/- {importance.importances_std[index]:.2f}")

    print("\nA caveat about the data")
    print("  Accident history and model year come out near zero. In real used-car")
    print("  pricing those two dominate — age drives depreciation and a recorded")
    print("  accident cuts resale value sharply. Their absence here is a property")
    print("  of a synthetic competition dataset, not a finding about used EVs, and")
    print("  no conclusion about the real market should be drawn from this model.")

    plot(table, ladder, y)


if __name__ == "__main__":
    main()
