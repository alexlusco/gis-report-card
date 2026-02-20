import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(exist_ok=True)

CSV_PATH = OUT_DIR / "synthetic_gis_survey.csv"
FTE_BUNDLE_PATH = OUT_DIR / "fte_bootstrap_bundle.joblib"
SHARE_BUNDLE_PATH = OUT_DIR / "share_bootstrap_bundle.joblib"

RANDOM_SEED = 42

PROVINCES_TERRITORIES = [
    "BC", "AB", "SK", "MB", "ON", "QC", "NB", "NS", "PE", "NL", "YT", "NT", "NU"
]

# Dropdown bins (labels must match app.py)
POP_BINS = [
    ("< 25k",        0,         25_000),
    ("25k–99k",      25_000,    100_000),
    ("100k–249k",    100_000,   250_000),
    ("250k–999k",    250_000, 1_000_000),
    ("1M+",        1_000_000, 10_000_000),
]

BUDGET_BINS = [
    ("< $250M",        0,              250_000_000),
    ("$250M–$999M",    250_000_000,   1_000_000_000),
    ("$1B–$4.9B",      1_000_000_000, 5_000_000_000),
    ("$5B+",           5_000_000_000, 100_000_000_000),
]


def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def sample_from_bins(rng, bins, n, probs=None):
    labels = [b[0] for b in bins]
    chosen = rng.choice(labels, size=n, p=probs, replace=True)

    values = np.zeros(n, dtype=float)
    for i, lab in enumerate(chosen):
        lo, hi = next((b[1], b[2]) for b in bins if b[0] == lab)
        # For open-ended-ish top bins, sample between lo and 2*lo as a proxy
        if hi >= 10_000_000 or hi >= 100_000_000_000:
            hi2 = max(lo * 2, lo + 1)
            values[i] = rng.uniform(lo, hi2)
        else:
            values[i] = rng.uniform(lo, hi)
    return chosen, values


def generate_synthetic_surveys(n=50, seed=RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    province = rng.choice(PROVINCES_TERRITORIES, size=n, replace=True)

    # Bin probabilities (more small/medium)
    pop_probs = np.array([0.20, 0.35, 0.25, 0.17, 0.03])
    pop_bin, population = sample_from_bins(rng, POP_BINS, n, probs=pop_probs)

    bud_probs = np.array([0.25, 0.40, 0.25, 0.10])
    budget_bin, municipal_budget = sample_from_bins(rng, BUDGET_BINS, n, probs=bud_probs)

    # Controls/predictors
    annual_retention_rate = clamp(rng.normal(0.88, 0.06, size=n), 0.60, 0.99)
    training_hours = clamp(rng.normal(18, 7, size=n), 0, 60)

    has_gis_strategy = rng.binomial(1, p=0.65, size=n).astype(int)

    tickets_per_month = clamp(rng.lognormal(mean=np.log(180), sigma=0.7, size=n), 10, 2500)
    avg_years_experience = clamp(rng.normal(7.0, 3.0, size=n), 0.5, 25)

    pct_reactive = clamp(rng.normal(0.55, 0.18, size=n), 0.05, 0.95)
    num_domains = rng.integers(2, 13, size=n)

    # Province effect (small)
    prov_eff = {p: rng.normal(0.0, 0.10) for p in PROVINCES_TERRITORIES}

    pop_score = pd.Categorical(pop_bin, categories=[b[0] for b in POP_BINS], ordered=True).codes
    bud_score = pd.Categorical(budget_bin, categories=[b[0] for b in BUDGET_BINS], ordered=True).codes

    # Outcome 1: GIS FTE
    latent_log_fte = (
        0.8
        + 0.35 * pop_score
        + 0.20 * bud_score
        + 0.35 * np.log(tickets_per_month)
        + 0.08 * num_domains
        + 0.10 * has_gis_strategy
        + 0.03 * training_hours
        + 0.06 * avg_years_experience
        - 0.90 * (1 - annual_retention_rate)
        - 0.35 * pct_reactive
        + np.array([prov_eff[p] for p in province])
        + rng.normal(0, 0.35, size=n)
    )
    gis_fte = np.exp(latent_log_fte)
    gis_fte = clamp(gis_fte, 1, 300)

    # Outcome 2: GIS spend share (% of total municipal budget)
    # Goal: wide spread between ~0.5% and 3.0%, without piling up at the top.
    #
    # Approach:
    # 1) Build a "mean" spend share (mu) from predictors (kept in 0..1)
    # 2) Sample around that mean with a Beta distribution (adds variability)
    # 3) Map into [0.5, 3.0] percent

    # Build a predictor-driven mean in percent units (roughly 0.6% .. 2.6%)
    base_pct = (
        0.9
        + 0.22 * np.log(gis_fte)
        + 0.25 * has_gis_strategy
        + 0.015 * training_hours
        + 0.04 * num_domains
        + 0.06 * np.log(tickets_per_month)
        - 0.18 * bud_score
        + rng.normal(0, 0.35, size=n)  # municipality-to-municipality spread
    )
    base_pct = clamp(base_pct, 0.5, 3.0)

    # Convert to 0..1 mean for Beta on the [0.5, 3.0] scale
    mu = (base_pct - 0.5) / (3.0 - 0.5)          # 0..1
    mu = clamp(mu, 0.02, 0.98)                   # avoid edge issues

    # Concentration: smaller => more variability. Tune this (4–12) to taste.
    k = 6.0

    a = mu * k
    b = (1 - mu) * k

    draw01 = rng.beta(a, b)                      # 0..1 sample around mean
    gis_spend_share_percent = 0.5 + draw01 * (3.0 - 0.5)

    df = pd.DataFrame({
        "client_id": [f"C{str(i+1).zfill(3)}" for i in range(n)],

        "province_territory": province,
        "population_range": pop_bin,
        "municipal_budget_range": budget_bin,

        "annual_retention_rate": np.round(annual_retention_rate, 3),
        "avg_training_hours_per_employee": np.round(training_hours, 1),
        "has_gis_strategy": has_gis_strategy,
        "tickets_per_month": np.round(tickets_per_month, 0).astype(int),
        "avg_years_experience": np.round(avg_years_experience, 1),
        "pct_work_reactive": np.round(pct_reactive, 3),
        "num_domains_served": num_domains,

        "gis_fte": np.round(gis_fte, 1),
        "gis_spend_share_percent": np.round(gis_spend_share_percent, 3),

        # Optional raw numeric for sanity checks (not used in model)
        "population_numeric_sim": np.round(population).astype(int),
        "municipal_budget_numeric_sim": np.round(municipal_budget).astype(np.int64),
    })

    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = ["province_territory", "population_range", "municipal_budget_range", "has_gis_strategy"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def pick_ridge_alpha(X, y, seed=RANDOM_SEED) -> float:
    alphas = np.logspace(-2, 3, 25)  # 0.01 .. 1000
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    best_alpha = float(alphas[0])
    best_mse = np.inf

    for a in alphas:
        mses = []
        for tr, te in kf.split(X):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr, yte = y.iloc[tr], y.iloc[te]

            model = Pipeline(steps=[
                ("pre", build_preprocessor(Xtr)),
                ("ridge", Ridge(alpha=float(a), random_state=seed)),
            ])
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            mses.append(np.mean((yte - pred) ** 2))

        avg_mse = float(np.mean(mses))
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_alpha = float(a)

    return best_alpha


def fit_bootstrap_bundle(X, y, alpha, n_boot=400, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    n = len(X)

    bootstrap_models = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Xb = X.iloc[idx].copy()
        yb = y.iloc[idx].copy()

        pipe = Pipeline(steps=[
            ("pre", build_preprocessor(Xb)),
            ("ridge", Ridge(alpha=float(alpha), random_state=seed)),
        ])
        pipe.fit(Xb, yb)
        bootstrap_models.append(pipe)

    central = Pipeline(steps=[
        ("pre", build_preprocessor(X)),
        ("ridge", Ridge(alpha=float(alpha), random_state=seed)),
    ])
    central.fit(X, y)

    return {
        "alpha": float(alpha),
        "central_model": central,
        "bootstrap_models": bootstrap_models,
        "feature_columns": list(X.columns),
        "q_lo": 0.25,
        "q_hi": 0.75,
        "q_med": 0.50,
    }


def main():
    df = generate_synthetic_surveys(n=50, seed=RANDOM_SEED)
    df.to_csv(CSV_PATH, index=False)
    print(f"Wrote: {CSV_PATH}")

    feature_cols = [
        "province_territory",
        "population_range",
        "municipal_budget_range",
        "annual_retention_rate",
        "avg_training_hours_per_employee",
        "has_gis_strategy",
        "tickets_per_month",
        "avg_years_experience",
        "pct_work_reactive",
        "num_domains_served",
    ]
    X = df[feature_cols]

    y_fte = df["gis_fte"]
    alpha_fte = pick_ridge_alpha(X, y_fte)
    fte_bundle = fit_bootstrap_bundle(X, y_fte, alpha=alpha_fte, n_boot=400)
    joblib.dump(fte_bundle, FTE_BUNDLE_PATH)
    print(f"Saved: {FTE_BUNDLE_PATH} (alpha={alpha_fte:.4g}, boot=400)")

    y_share = df["gis_spend_share_percent"]
    alpha_share = pick_ridge_alpha(X, y_share)
    share_bundle = fit_bootstrap_bundle(X, y_share, alpha=alpha_share, n_boot=400)
    joblib.dump(share_bundle, SHARE_BUNDLE_PATH)
    print(f"Saved: {SHARE_BUNDLE_PATH} (alpha={alpha_share:.4g}, boot=400)")

    print("\nSample:")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()