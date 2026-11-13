"""
Core simulation for the workshop.

Setting
-------
A corpus of N firm-quarter disclosure snippets. A latent binary concept
D* ("this filing discusses funding stress") moves an outcome y (change in
CDS spread, bps). We cannot read 4,000 filings, so we have an LLM label
them. We also hand-label a small random set of filings as ground truth.

    y = alpha + beta * D* + gamma * x + eps,     beta = 2.0

Two prompts produce two different labels of the same concept. The question
is what happens to beta-hat, and what to do about it.

Design note: the hand-labeled set is drawn independently of the main corpus
(as in the prediction-powered-inference setup), which keeps the variance
algebra clean -- the two terms simply add.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- parameters
N_CORPUS = 4000      # LLM-labeled corpus
N_LABELED = 200      # hand-labeled validation set
BETA_TRUE = 2.0
GAMMA_TRUE = 1.0
ALPHA_TRUE = 0.5
SIGMA = 3.0
P_D = 0.30

# prompt error profiles: (sensitivity, specificity)
PROMPTS = {
    "A (broad)":  (0.95, 0.80),   # flags almost every true case, over-flags
    "B (strict)": (0.65, 0.98),   # almost never over-flags, misses a third
}
PCOLS = {"A (broad)": "d_A", "B (strict)": "d_B"}


# ------------------------------------------------------------------ the DGP
def draw(rng, n, differential=False):
    x = rng.normal(size=n)
    d_true = (rng.random(n) < P_D).astype(int)
    y = ALPHA_TRUE + BETA_TRUE * d_true + GAMMA_TRUE * x + rng.normal(0, SIGMA, n)

    out = {"x": x, "d_true": d_true, "y": y}
    for name, (sens, spec) in PROMPTS.items():
        if differential:
            # the LLM over-flags exactly the documents whose outcome was bad:
            # error rates now depend on y, not just on D*
            shift = 0.12 * np.sign(y - np.median(y))
            s_eff = np.clip(sens + shift, 0.01, 0.99)
            p_eff = np.clip(spec - shift, 0.01, 0.99)
        else:
            s_eff = np.full(n, sens)
            p_eff = np.full(n, spec)
        u = rng.random(n)
        out[PCOLS[name]] = np.where(d_true == 1, u < s_eff, u > p_eff).astype(int)
    return pd.DataFrame(out)


# ------------------------------------------- OLS with influence functions
def ols_if(y, d, x):
    """
    Regress y on [1, d, x]. Return the coefficient on d and the influence
    function for that coefficient, so that beta ~= mean(psi) + beta and
    var(beta_hat) = var(psi) / n  (heteroskedasticity-robust).
    """
    n = len(y)
    Z = np.column_stack([np.ones(n), d, x]).astype(float)
    ZtZ = Z.T @ Z
    beta = np.linalg.solve(ZtZ, Z.T @ y)
    resid = y - Z @ beta
    Sinv = np.linalg.inv(ZtZ / n)
    psi = (Z * resid[:, None]) @ Sinv[:, 1]       # influence fn for coef on d
    return float(beta[1]), psi


def fit(corpus, labeled, dcol, z=1.96):
    """Naive, labeled-only, and rectified estimates with robust CIs."""
    b_naive, psi_naive = ols_if(corpus["y"].values, corpus[dcol].values,
                                corpus["x"].values)
    se_naive = psi_naive.std(ddof=1) / np.sqrt(len(corpus))

    b_lab, psi_lab = ols_if(labeled["y"].values, labeled["d_true"].values,
                            labeled["x"].values)
    se_lab = psi_lab.std(ddof=1) / np.sqrt(len(labeled))

    # the rectifier: how far the LLM label pushes beta, measured where we
    # can see both the LLM label and the truth
    b_lab_llm, psi_lab_llm = ols_if(labeled["y"].values, labeled[dcol].values,
                                    labeled["x"].values)
    rect = b_lab - b_lab_llm
    se_rect_term = (psi_lab - psi_lab_llm).std(ddof=1) / np.sqrt(len(labeled))

    b_rect = b_naive + rect
    se_b_rect = np.sqrt(se_naive ** 2 + se_rect_term ** 2)

    res = {
        "naive":        (b_naive, se_naive),
        "labeled_only": (b_lab,   se_lab),
        "rectified":    (b_rect,  se_b_rect),
    }
    return {k: {"beta": b, "se": s, "lo": b - z * s, "hi": b + z * s}
            for k, (b, s) in res.items()}


# ------------------------------------------------------------- headline run
def headline(seed=20261102, differential=False):
    rng = np.random.default_rng(seed)
    corpus = draw(rng, N_CORPUS, differential)
    labeled = draw(rng, N_LABELED, differential)
    rows = []
    for name, dcol in PCOLS.items():
        for est, v in fit(corpus, labeled, dcol).items():
            rows.append({"prompt": name, "estimator": est, **v,
                         "width": v["hi"] - v["lo"],
                         "covers": v["lo"] <= BETA_TRUE <= v["hi"]})
    return corpus, labeled, pd.DataFrame(rows)


# ---------------------------------------------------------- monte carlo
def monte_carlo(reps=2000, differential=False, seed=7):
    rng = np.random.default_rng(seed)
    acc = {(p, e): {"b": [], "cov": 0}
           for p in PCOLS for e in ("naive", "labeled_only", "rectified")}
    for _ in range(reps):
        corpus = draw(rng, N_CORPUS, differential)
        labeled = draw(rng, N_LABELED, differential)
        for p, dcol in PCOLS.items():
            for e, v in fit(corpus, labeled, dcol).items():
                acc[(p, e)]["b"].append(v["beta"])
                acc[(p, e)]["cov"] += int(v["lo"] <= BETA_TRUE <= v["hi"])
    rows = []
    for (p, e), a in acc.items():
        b = np.array(a["b"])
        rows.append({"prompt": p, "estimator": e,
                     "mean_beta": b.mean(),
                     "bias": b.mean() - BETA_TRUE,
                     "rmse": np.sqrt(np.mean((b - BETA_TRUE) ** 2)),
                     "coverage": a["cov"] / reps})
    return pd.DataFrame(rows)


# ------------------------------------------- how many documents to label?
def labeling_curve(sizes=(25, 50, 100, 200, 400, 800, 1600), reps=400, seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    for n_lab in sizes:
        w_r, w_l = [], []
        for _ in range(reps):
            corpus = draw(rng, N_CORPUS)
            labeled = draw(rng, n_lab)
            f = fit(corpus, labeled, "d_A")
            w_r.append(f["rectified"]["hi"] - f["rectified"]["lo"])
            w_l.append(f["labeled_only"]["hi"] - f["labeled_only"]["lo"])
        rows.append({"n_labeled": n_lab,
                     "width_rectified": np.mean(w_r),
                     "width_labeled_only": np.mean(w_l)})
    return pd.DataFrame(rows)


def quality_sweep(accuracies=(0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
                  reps=400, seed=13):
    """
    How much precision does rectifying actually buy, as a function of label
    quality? Sensitivity = specificity = acc, so the label gets steadily
    better. Compares the rectified CI against simply using the 200
    hand-labeled documents alone.
    """
    global PROMPTS, PCOLS
    keep_p, keep_c = PROMPTS, PCOLS
    rows = []
    rng = np.random.default_rng(seed)
    for acc in accuracies:
        PROMPTS = {"Q": (acc, acc)}
        PCOLS = {"Q": "d_Q"}
        w_r, w_l = [], []
        for _ in range(reps):
            corpus = draw(rng, N_CORPUS)
            labeled = draw(rng, N_LABELED)
            f = fit(corpus, labeled, "d_Q")
            w_r.append(f["rectified"]["hi"] - f["rectified"]["lo"])
            w_l.append(f["labeled_only"]["hi"] - f["labeled_only"]["lo"])
        rows.append({"label_accuracy": acc,
                     "width_rectified": np.mean(w_r),
                     "width_labeled_only": np.mean(w_l),
                     "pct_narrower": 100 * (1 - np.mean(w_r) / np.mean(w_l))})
    PROMPTS, PCOLS = keep_p, keep_c
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:9.3f}")

    corpus, labeled, tab = headline()

    print("\n=== how good are the labels? (full corpus, vs ground truth) ===")
    for name, dcol in PCOLS.items():
        sens = corpus.loc[corpus.d_true == 1, dcol].mean()
        spec = 1 - corpus.loc[corpus.d_true == 0, dcol].mean()
        agree = (corpus[dcol] == corpus.d_true).mean()
        print(f"  {name:12s} flagged={corpus[dcol].mean():.3f}  "
              f"sens={sens:.3f}  spec={spec:.3f}  raw agreement={agree:.3f}")
    print(f"  {'ground truth':12s} share  ={corpus.d_true.mean():.3f}")

    print("\n=== headline estimates (beta_true = 2.000) ===")
    print(tab.to_string(index=False))

    print("\n=== monte carlo, 2000 reps, nondifferential LLM errors ===")
    print(monte_carlo().to_string(index=False))

    print("\n=== monte carlo, 2000 reps, DIFFERENTIAL LLM errors ===")
    print(monte_carlo(differential=True, seed=99).to_string(index=False))

    print("\n=== how many documents do you need to hand-label? ===")
    print(labeling_curve().to_string(index=False))
