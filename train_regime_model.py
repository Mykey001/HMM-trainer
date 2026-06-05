"""
Market Regime Model — Training & Evaluation Pipeline
=====================================================
Trains multiple regime detection models, evaluates them rigorously,
and presents results for the user to choose the best configuration.

Models trained:
  1. GMM with 4 components (fixed — backward compatible)
  2. GMM with BIC-optimal components (3–8 sweep)
  3. HMM with 4 components (fixed)
  4. HMM with BIC-optimal components (3–8 sweep)
  5. Bayesian GMM (auto-selects component count)

Evaluation:
  - Clustering metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin)
  - Regime interpretability (mean returns/volatility per regime)
  - Regime stability (mean duration, transition frequency)
  - Supervised metrics vs rule-based ground truth (accuracy, precision, recall, F1)
  - Visualizations (regime overlay, confusion matrix, transition matrix)

Data: XAUUSD M5, Sep 2018 – Apr 2026 (~536K bars)
Split: Train 2018–2024 / Test 2025–2026

Author: Retrained pipeline — June 2026
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
from scipy.optimize import linear_sum_assignment

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMM = True
except ImportError:
    HAS_HMM = False
    print("[WARNING] hmmlearn not installed. HMM models will be skipped.")
    print("         Install with: pip install hmmlearn")

from feature_engine import compute_all_features, get_feature_matrix, FEATURE_NAMES

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(
    SCRIPT_DIR, "XAUUSD_M5_201809211615_202604301140.csv"
)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "evaluation_results")

# Train/test temporal split
TRAIN_END_DATE = "2024-12-31"    # Train: 2018–2024
TEST_START_DATE = "2025-01-01"   # Test: 2025–2026

# Model sweep range
N_COMPONENTS_RANGE = range(3, 9)  # 3, 4, 5, 6, 7, 8
N_RANDOM_STARTS = 10              # Multiple random initializations
MAX_ITER = 500                    # Max EM iterations

# Ground truth parameters
FORWARD_WINDOW = 12   # 1-hour forward return (12 x M5)
RETURN_UPPER_PCT = 70  # Percentile for "bullish" threshold
RETURN_LOWER_PCT = 30  # Percentile for "bearish" threshold
VOL_MEDIAN_PCT = 50    # Percentile for volatility split


# ============================================================
# DATA LOADING
# ============================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Load and parse the MT5 CSV data file.

    Expected format: tab-separated with columns:
        <DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>
    """
    print(f"Loading data from: {filepath}")
    t0 = time.perf_counter()

    df = pd.read_csv(
        filepath,
        sep="\t",
        header=0,
        names=["date", "time", "open", "high", "low", "close",
               "tickvol", "vol", "spread"],
        dtype={
            "open": np.float64,
            "high": np.float64,
            "low": np.float64,
            "close": np.float64,
            "tickvol": np.float64,
            "vol": np.float64,
            "spread": np.float64,
        },
    )

    # Parse datetime
    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        format="%Y.%m.%d %H:%M:%S",
    )
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)

    elapsed = time.perf_counter() - t0
    print(f"  Loaded {len(df):,} bars in {elapsed:.1f}s")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")

    return df


# ============================================================
# GROUND TRUTH LABELING
# ============================================================

def create_ground_truth(df: pd.DataFrame,
                        forward_window: int = FORWARD_WINDOW) -> np.ndarray:
    """Create rule-based ground truth regime labels.

    Uses forward returns (direction) + trailing volatility (magnitude)
    to classify each bar into one of 4 regimes:
        0: Low Volatility Bullish
        1: Neutral Consolidation
        2: High Volatility Bearish
        3: Low Volatility Bearish

    These labels are ONLY for evaluation — never used in training.
    Forward-looking nature prevents any data leakage into unsupervised models.
    """
    close = df["close"].values
    n = len(close)

    # Forward return (1-hour)
    fwd_ret = np.full(n, np.nan)
    fwd_ret[:n - forward_window] = (
        close[forward_window:] / close[:n - forward_window] - 1.0
    )

    # Trailing 1-day realized volatility
    log_ret = np.log(close[1:] / close[:-1])
    log_ret = np.concatenate(([np.nan], log_ret))
    trail_vol = pd.Series(log_ret).rolling(288, min_periods=100).std().values

    # Compute thresholds from valid data
    valid_mask = ~(np.isnan(fwd_ret) | np.isnan(trail_vol))
    valid_fwd = fwd_ret[valid_mask]
    valid_vol = trail_vol[valid_mask]

    ret_upper = np.percentile(valid_fwd, RETURN_UPPER_PCT)
    ret_lower = np.percentile(valid_fwd, RETURN_LOWER_PCT)
    vol_median = np.percentile(valid_vol, VOL_MEDIAN_PCT)

    # Assign labels
    labels = np.full(n, -1, dtype=np.int32)

    for i in range(n):
        if np.isnan(fwd_ret[i]) or np.isnan(trail_vol[i]):
            continue

        if fwd_ret[i] > ret_upper:
            if trail_vol[i] < vol_median:
                labels[i] = 0  # Low Vol Bullish
            else:
                labels[i] = 0  # High Vol Bullish → group with Bullish
        elif fwd_ret[i] < ret_lower:
            if trail_vol[i] >= vol_median:
                labels[i] = 2  # High Vol Bearish
            else:
                labels[i] = 3  # Low Vol Bearish
        else:
            labels[i] = 1  # Neutral Consolidation

    return labels


# ============================================================
# MODEL TRAINING
# ============================================================

def train_gmm(X: np.ndarray, n_components: int,
              random_state: int = 42) -> GaussianMixture:
    """Train a Gaussian Mixture Model."""
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        n_init=N_RANDOM_STARTS,
        max_iter=MAX_ITER,
        random_state=random_state,
        tol=1e-4,
    )
    gmm.fit(X)
    return gmm


def train_hmm(X: np.ndarray, n_components: int,
              random_state: int = 42) -> "GaussianHMM":
    """Train a Gaussian Hidden Markov Model."""
    if not HAS_HMM:
        return None

    best_model = None
    best_score = -np.inf

    for seed in range(N_RANDOM_STARTS):
        try:
            model = GaussianHMM(
                n_components=n_components,
                covariance_type="full",
                n_iter=MAX_ITER,
                random_state=random_state + seed,
                tol=1e-4,
            )
            model.fit(X)
            score = model.score(X)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue

    return best_model


def train_dpgmm(X: np.ndarray, max_components: int = 10,
                random_state: int = 42) -> BayesianGaussianMixture:
    """Train a Dirichlet Process Gaussian Mixture Model.

    Automatically selects the optimal number of components.
    """
    dpgmm = BayesianGaussianMixture(
        n_components=max_components,
        covariance_type="full",
        max_iter=MAX_ITER,
        random_state=random_state,
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=1.0,
        tol=1e-4,
    )
    dpgmm.fit(X)
    return dpgmm


# ============================================================
# REGIME LABELING (Auto-assign names from cluster statistics)
# ============================================================

REGIME_TEMPLATES = {
    "low_vol_bullish":   {"id": 0, "name": "Low Volatility Bullish",   "color": "#2ecc71"},
    "neutral":           {"id": 1, "name": "Neutral Consolidation",    "color": "#f39c12"},
    "high_vol_bearish":  {"id": 2, "name": "High Volatility Bearish",  "color": "#e74c3c"},
    "low_vol_bearish":   {"id": 3, "name": "Low Volatility Bearish",   "color": "#9b59b6"},
    "high_vol_bullish":  {"id": 4, "name": "High Volatility Bullish",  "color": "#3498db"},
    "low_vol_neutral":   {"id": 5, "name": "Low Volatility Neutral",   "color": "#1abc9c"},
    "high_vol_neutral":  {"id": 6, "name": "High Volatility Neutral",  "color": "#e67e22"},
    "extreme_bearish":   {"id": 7, "name": "Extreme Volatility Crisis", "color": "#c0392b"},
}


def auto_label_regimes(X: np.ndarray, labels: np.ndarray,
                       df: pd.DataFrame, feature_names: list) -> dict:
    """Auto-assign meaningful regime names based on cluster statistics.

    Analyzes mean returns and volatility within each cluster to
    determine which market regime it represents.

    Returns:
        Dict mapping cluster_id -> {"name": str, "color": str, ...}
    """
    n_clusters = len(np.unique(labels[labels >= 0]))

    # Compute per-cluster statistics
    log_returns = np.log(
        df["close"].values[1:] / df["close"].values[:-1]
    )
    log_returns = np.concatenate(([0], log_returns))

    cluster_stats = []
    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() == 0:
            cluster_stats.append({"return": 0, "vol": 0, "count": 0})
            continue

        cluster_rets = log_returns[mask]
        cluster_stats.append({
            "return": np.mean(cluster_rets),
            "vol": np.std(cluster_rets),
            "count": int(mask.sum()),
        })

    # Sort clusters by return (ascending)
    sorted_by_return = sorted(range(n_clusters),
                              key=lambda k: cluster_stats[k]["return"])

    # Get median volatility across clusters
    vols = [cluster_stats[k]["vol"] for k in range(n_clusters)]
    vol_median = np.median(vols)

    # Assign labels
    regime_map = {}
    for rank, k in enumerate(sorted_by_return):
        stats = cluster_stats[k]
        is_high_vol = stats["vol"] > vol_median
        frac = rank / max(n_clusters - 1, 1)  # 0 = most bearish, 1 = most bullish

        if n_clusters <= 4:
            if frac <= 0.25:
                template = ("high_vol_bearish" if is_high_vol
                            else "low_vol_bearish")
            elif frac >= 0.75:
                template = ("high_vol_bullish" if is_high_vol
                            else "low_vol_bullish")
            else:
                template = "neutral"
        else:
            if frac <= 0.2:
                template = ("extreme_bearish" if is_high_vol
                            else "low_vol_bearish")
            elif frac <= 0.4:
                template = ("high_vol_bearish" if is_high_vol
                            else "low_vol_bearish")
            elif frac >= 0.8:
                template = ("high_vol_bullish" if is_high_vol
                            else "low_vol_bullish")
            elif frac >= 0.6:
                template = ("low_vol_bullish" if not is_high_vol
                            else "high_vol_bullish")
            else:
                template = ("high_vol_neutral" if is_high_vol
                            else "low_vol_neutral")

        info = REGIME_TEMPLATES[template].copy()
        info["cluster_id"] = k
        info["mean_return"] = float(stats["return"])
        info["mean_volatility"] = float(stats["vol"])
        info["bar_count"] = stats["count"]
        regime_map[int(k)] = info

    return regime_map


# ============================================================
# EVALUATION METRICS
# ============================================================

def compute_clustering_metrics(X: np.ndarray,
                               labels: np.ndarray) -> dict:
    """Compute internal clustering quality metrics."""
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return {"silhouette": -1, "calinski_harabasz": 0, "davies_bouldin": 999}

    # Sample for silhouette (full dataset is too slow)
    n_sample = min(50000, len(X))
    idx = np.random.choice(len(X), n_sample, replace=False)

    return {
        "silhouette": float(silhouette_score(X[idx], labels[idx])),
        "calinski_harabasz": float(calinski_harabasz_score(X[idx], labels[idx])),
        "davies_bouldin": float(davies_bouldin_score(X[idx], labels[idx])),
    }


def compute_regime_stability(labels: np.ndarray) -> dict:
    """Compute regime stability metrics."""
    # Count regime transitions
    transitions = np.sum(labels[1:] != labels[:-1])
    transition_rate = transitions / len(labels)

    # Compute regime durations
    durations = []
    current_regime = labels[0]
    current_duration = 1

    for i in range(1, len(labels)):
        if labels[i] == current_regime:
            current_duration += 1
        else:
            durations.append(current_duration)
            current_regime = labels[i]
            current_duration = 1
    durations.append(current_duration)

    durations = np.array(durations)

    return {
        "total_transitions": int(transitions),
        "transition_rate": float(transition_rate),
        "mean_duration_bars": float(np.mean(durations)),
        "median_duration_bars": float(np.median(durations)),
        "mean_duration_hours": float(np.mean(durations) * 5 / 60),
        "median_duration_hours": float(np.median(durations) * 5 / 60),
        "min_duration_bars": int(np.min(durations)),
        "max_duration_bars": int(np.max(durations)),
    }


def compute_supervised_metrics(pred_labels: np.ndarray,
                               gt_labels: np.ndarray,
                               n_pred_clusters: int) -> dict:
    """Compute supervised metrics using Hungarian algorithm for label matching.

    Maps predicted cluster IDs to ground truth labels optimally.
    """
    # Filter valid ground truth
    valid_mask = gt_labels >= 0
    pred_valid = pred_labels[valid_mask]
    gt_valid = gt_labels[valid_mask]

    if len(pred_valid) == 0:
        return {"accuracy": 0, "precision_macro": 0,
                "recall_macro": 0, "f1_macro": 0}

    n_gt_classes = len(np.unique(gt_valid))

    # Build cost matrix for Hungarian algorithm
    cost = np.zeros((n_pred_clusters, n_gt_classes), dtype=np.int64)
    gt_unique = np.unique(gt_valid)

    for i in range(n_pred_clusters):
        for j, gt_class in enumerate(gt_unique):
            cost[i, j] = -np.sum((pred_valid == i) & (gt_valid == gt_class))

    # Optimal matching
    row_ind, col_ind = linear_sum_assignment(cost)

    # Create mapped predictions
    label_map = {}
    for r, c in zip(row_ind, col_ind):
        label_map[r] = gt_unique[c]

    # Map predictions
    mapped_pred = np.array([
        label_map.get(p, -1) for p in pred_valid
    ])

    # Filter out unmapped predictions
    mapped_mask = mapped_pred >= 0
    mapped_pred = mapped_pred[mapped_mask]
    mapped_gt = gt_valid[mapped_mask]

    if len(mapped_pred) == 0:
        return {"accuracy": 0, "precision_macro": 0,
                "recall_macro": 0, "f1_macro": 0, "label_map": label_map}

    acc = accuracy_score(mapped_gt, mapped_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        mapped_gt, mapped_pred, average="macro", zero_division=0
    )

    # Per-class metrics
    prec_per, rec_per, f1_per, sup_per = precision_recall_fscore_support(
        mapped_gt, mapped_pred, average=None, zero_division=0
    )

    conf_mat = confusion_matrix(mapped_gt, mapped_pred)

    return {
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "precision_per_class": prec_per.tolist(),
        "recall_per_class": rec_per.tolist(),
        "f1_per_class": f1_per.tolist(),
        "support_per_class": sup_per.tolist(),
        "confusion_matrix": conf_mat.tolist(),
        "label_map": {int(k): int(v) for k, v in label_map.items()},
    }


def compute_economic_metrics(labels: np.ndarray,
                             df: pd.DataFrame) -> dict:
    """Compute economic interpretability metrics.

    Checks if regimes actually predict different forward returns
    (the ultimate test of a regime model).
    """
    close = df["close"].values
    n = len(close)

    # 1-hour forward return
    fwd_ret = np.full(n, np.nan)
    fwd_ret[:n - 12] = close[12:] / close[:n - 12] - 1.0

    results = {}
    for regime_id in np.unique(labels):
        mask = (labels == regime_id) & (~np.isnan(fwd_ret))
        if mask.sum() == 0:
            continue

        rets = fwd_ret[mask]
        results[int(regime_id)] = {
            "mean_fwd_return_bps": float(np.mean(rets) * 10000),
            "median_fwd_return_bps": float(np.median(rets) * 10000),
            "std_fwd_return_bps": float(np.std(rets) * 10000),
            "win_rate_pct": float(np.mean(rets > 0) * 100),
            "bar_count": int(mask.sum()),
        }

    return results


# ============================================================
# VISUALIZATION
# ============================================================

def plot_regime_overlay(df: pd.DataFrame, labels: np.ndarray,
                        regime_map: dict, title: str,
                        filepath: str, n_bars: int = 5000):
    """Plot price chart with regime-colored background."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                    gridspec_kw={"height_ratios": [3, 1]})

    # Use last n_bars for visibility
    start = max(0, len(df) - n_bars)
    plot_df = df.iloc[start:]
    plot_labels = labels[start:]
    plot_idx = range(len(plot_df))

    # Price chart with regime backgrounds
    ax1.plot(plot_idx, plot_df["close"].values,
             color="white", linewidth=0.5, alpha=0.9)

    # Color backgrounds
    for regime_id, info in regime_map.items():
        mask = plot_labels == regime_id
        if mask.any():
            for i in range(len(mask)):
                if mask[i]:
                    ax1.axvspan(i - 0.5, i + 0.5, alpha=0.3,
                                color=info["color"], linewidth=0)

    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.set_ylabel("Price (XAUUSD)")
    ax1.set_facecolor("#1a1a2e")
    ax1.grid(True, alpha=0.2)

    # Legend
    legend_elements = []
    for regime_id in sorted(regime_map.keys()):
        info = regime_map[regime_id]
        from matplotlib.patches import Patch
        legend_elements.append(
            Patch(facecolor=info["color"], alpha=0.5,
                  label=info["name"])
        )
    ax1.legend(handles=legend_elements, loc="upper left",
               fontsize=8, framealpha=0.8)

    # Regime timeline
    unique_regimes = sorted(regime_map.keys())
    colors = [regime_map[r]["color"] for r in unique_regimes]

    for i, regime_id in enumerate(unique_regimes):
        mask = plot_labels == regime_id
        if mask.any():
            ax2.fill_between(plot_idx, 0, 1,
                             where=mask, alpha=0.7,
                             color=regime_map[regime_id]["color"],
                             label=regime_map[regime_id]["name"])

    ax2.set_ylabel("Regime")
    ax2.set_xlabel(f"Bar Index (last {n_bars} bars)")
    ax2.set_facecolor("#1a1a2e")
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])

    fig.patch.set_facecolor("#0d1117")
    ax1.tick_params(colors="white")
    ax2.tick_params(colors="white")
    ax1.xaxis.label.set_color("white")
    ax1.yaxis.label.set_color("white")
    ax2.xaxis.label.set_color("white")
    ax2.yaxis.label.set_color("white")
    ax1.title.set_color("white")

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {filepath}")


def plot_confusion_matrix(conf_mat: np.ndarray, regime_names: list,
                          title: str, filepath: str):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))

    if HAS_SEABORN:
        sns.heatmap(conf_mat, annot=True, fmt="d", cmap="YlOrRd",
                    xticklabels=regime_names,
                    yticklabels=regime_names, ax=ax)
    else:
        im = ax.imshow(conf_mat, cmap="YlOrRd")
        for i in range(conf_mat.shape[0]):
            for j in range(conf_mat.shape[1]):
                ax.text(j, i, str(conf_mat[i, j]),
                        ha="center", va="center")
        ax.set_xticks(range(len(regime_names)))
        ax.set_yticks(range(len(regime_names)))
        ax.set_xticklabels(regime_names, rotation=45, ha="right")
        ax.set_yticklabels(regime_names)
        plt.colorbar(im)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


def plot_transition_matrix(labels: np.ndarray, regime_map: dict,
                           title: str, filepath: str):
    """Plot regime transition probability matrix."""
    n_regimes = len(regime_map)
    trans_count = np.zeros((n_regimes, n_regimes), dtype=np.int64)

    for i in range(len(labels) - 1):
        trans_count[labels[i], labels[i + 1]] += 1

    # Normalize to probabilities
    row_sums = trans_count.sum(axis=1, keepdims=True)
    trans_prob = np.where(row_sums > 0, trans_count / row_sums, 0)

    names = [regime_map[k]["name"] for k in sorted(regime_map.keys())]

    fig, ax = plt.subplots(figsize=(8, 6))

    if HAS_SEABORN:
        sns.heatmap(trans_prob, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=names, yticklabels=names, ax=ax,
                    vmin=0, vmax=1)
    else:
        im = ax.imshow(trans_prob, cmap="Blues", vmin=0, vmax=1)
        for i in range(trans_prob.shape[0]):
            for j in range(trans_prob.shape[1]):
                ax.text(j, i, f"{trans_prob[i, j]:.2f}",
                        ha="center", va="center")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticklabels(names)
        plt.colorbar(im)

    ax.set_xlabel("To Regime")
    ax.set_ylabel("From Regime")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


def plot_bic_comparison(bic_scores: dict, filepath: str):
    """Plot BIC scores for different component counts."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, scores in bic_scores.items():
        ks = sorted(scores.keys())
        vals = [scores[k] for k in ks]
        ax.plot(ks, vals, "o-", label=model_name, linewidth=2, markersize=8)
        best_k = ks[np.argmin(vals)]
        best_val = min(vals)
        ax.annotate(f"Best: k={best_k}", xy=(best_k, best_val),
                    fontsize=9, fontweight="bold",
                    textcoords="offset points", xytext=(10, 10),
                    arrowprops=dict(arrowstyle="->", color="red"))

    ax.set_xlabel("Number of Components (k)")
    ax.set_ylabel("BIC Score (lower is better)")
    ax.set_title("Model Selection: BIC Score vs Number of Components")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


def plot_model_comparison(results: dict, filepath: str):
    """Plot model comparison bar chart."""
    model_names = list(results.keys())
    metrics = ["accuracy", "precision_macro", "recall_macro",
               "f1_macro", "silhouette"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 6))

    for i, metric in enumerate(metrics):
        values = []
        for name in model_names:
            r = results[name]
            if metric == "silhouette":
                values.append(r.get("clustering", {}).get("silhouette", 0))
            else:
                values.append(r.get("supervised", {}).get(metric, 0))

        bars = axes[i].bar(range(len(model_names)), values,
                           color=["#3498db", "#2ecc71", "#e74c3c",
                                  "#9b59b6", "#f39c12"][:len(model_names)])

        axes[i].set_title(metric.replace("_", " ").title(), fontsize=11)
        axes[i].set_xticks(range(len(model_names)))
        axes[i].set_xticklabels(
            [n.replace(" ", "\n") for n in model_names],
            fontsize=8
        )
        axes[i].set_ylim(0, max(max(values) * 1.2, 0.1))

        for j, v in enumerate(values):
            axes[i].text(j, v + 0.01, f"{v:.3f}",
                         ha="center", fontsize=8, fontweight="bold")

    fig.suptitle("Model Comparison — All Metrics", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():
    print("=" * 70)
    print("  MARKET REGIME MODEL — TRAINING & EVALUATION PIPELINE")
    print("  " + "=" * 66)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --------------------------------------------------------
    # STEP 1: Load Data
    # --------------------------------------------------------
    print("[STEP 1/7] Loading data...")
    df_raw = load_data(DATA_FILE)
    print()

    # --------------------------------------------------------
    # STEP 2: Feature Engineering
    # --------------------------------------------------------
    print("[STEP 2/7] Computing 14 enhanced features (C-optimized NumPy)...")
    t0 = time.perf_counter()
    df_feat = compute_all_features(df_raw)

    # Drop NaN warmup rows
    X_all, df_valid = get_feature_matrix(df_feat, drop_na=True)

    elapsed = time.perf_counter() - t0
    print(f"  Computed {len(FEATURE_NAMES)} features in {elapsed:.1f}s")
    print(f"  Valid rows: {len(df_valid):,} (dropped {len(df_raw) - len(df_valid):,} warmup rows)")
    print()

    # --------------------------------------------------------
    # STEP 3: Train / Test Split
    # --------------------------------------------------------
    print("[STEP 3/7] Splitting train/test (temporal)...")

    train_mask = df_valid.index <= TRAIN_END_DATE
    test_mask = df_valid.index >= TEST_START_DATE

    df_train = df_valid[train_mask]
    df_test = df_valid[test_mask]

    X_train = df_train[FEATURE_NAMES].values
    X_test = df_test[FEATURE_NAMES].values

    print(f"  Train: {len(df_train):,} bars "
          f"({df_train.index[0].date()} to {df_train.index[-1].date()})")
    print(f"  Test:  {len(df_test):,} bars "
          f"({df_test.index[0].date()} to {df_test.index[-1].date()})")
    print()

    # --------------------------------------------------------
    # STEP 4: Scale Features
    # --------------------------------------------------------
    print("[STEP 4/7] Scaling features (RobustScaler)...")

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"  Scaler fitted on {len(X_train):,} training samples")
    print()

    # --------------------------------------------------------
    # STEP 5: Create Ground Truth for Test Set
    # --------------------------------------------------------
    print("[STEP 5/7] Creating ground truth labels for evaluation...")

    gt_labels_all = create_ground_truth(df_valid)
    gt_test = gt_labels_all[test_mask]

    valid_gt_count = np.sum(gt_test >= 0)
    print(f"  Ground truth labels: {valid_gt_count:,} valid out of {len(gt_test):,} test bars")
    for regime_id in sorted(np.unique(gt_test[gt_test >= 0])):
        count = np.sum(gt_test == regime_id)
        print(f"    Regime {regime_id}: {count:,} bars ({count / valid_gt_count * 100:.1f}%)")
    print()

    # --------------------------------------------------------
    # STEP 6: Train All Models
    # --------------------------------------------------------
    print("[STEP 6/7] Training models...")
    print("-" * 50)

    all_results = {}
    bic_scores = {"GMM": {}}
    if HAS_HMM:
        bic_scores["HMM"] = {}

    # --- 6a: GMM BIC Sweep ---
    print("\n  [6a] GMM — BIC sweep (k=3..8)...")
    best_gmm_bic = np.inf
    best_gmm_k = 4

    for k in N_COMPONENTS_RANGE:
        t0 = time.perf_counter()
        gmm_k = train_gmm(X_train_scaled, k)
        elapsed = time.perf_counter() - t0

        bic = gmm_k.bic(X_train_scaled)
        aic = gmm_k.aic(X_train_scaled)
        bic_scores["GMM"][k] = bic

        converged = "✓" if gmm_k.converged_ else "✗"
        print(f"    k={k}: BIC={bic:,.0f}  AIC={aic:,.0f}  "
              f"converged={converged}  time={elapsed:.1f}s")

        if bic < best_gmm_bic:
            best_gmm_bic = bic
            best_gmm_k = k
            best_gmm_model = gmm_k

    print(f"    → Best GMM: k={best_gmm_k} (BIC={best_gmm_bic:,.0f})")

    # --- 6b: GMM Fixed k=4 ---
    print("\n  [6b] GMM — Fixed k=4...")
    t0 = time.perf_counter()
    gmm_4 = train_gmm(X_train_scaled, 4)
    elapsed = time.perf_counter() - t0
    print(f"    BIC={gmm_4.bic(X_train_scaled):,.0f}  time={elapsed:.1f}s")

    # --- 6c: HMM BIC Sweep ---
    if HAS_HMM:
        print("\n  [6c] HMM — BIC sweep (k=3..8)...")
        best_hmm_score = -np.inf
        best_hmm_k = 4

        for k in N_COMPONENTS_RANGE:
            t0 = time.perf_counter()
            hmm_k = train_hmm(X_train_scaled, k)
            elapsed = time.perf_counter() - t0

            if hmm_k is not None:
                score = hmm_k.score(X_train_scaled)
                bic_approx = -2 * score * len(X_train_scaled) + \
                    (k * k + 2 * k * X_train_scaled.shape[1] - 1) * \
                    np.log(len(X_train_scaled))
                bic_scores["HMM"][k] = bic_approx

                print(f"    k={k}: LogLik={score:.2f}  "
                      f"BIC≈{bic_approx:,.0f}  time={elapsed:.1f}s")

                if score > best_hmm_score:
                    best_hmm_score = score
                    best_hmm_k = k
                    best_hmm_model = hmm_k
            else:
                print(f"    k={k}: FAILED")

        print(f"    → Best HMM: k={best_hmm_k} (LogLik={best_hmm_score:.2f})")

        # --- 6d: HMM Fixed k=4 ---
        print("\n  [6d] HMM — Fixed k=4...")
        t0 = time.perf_counter()
        hmm_4 = train_hmm(X_train_scaled, 4)
        elapsed = time.perf_counter() - t0
        if hmm_4 is not None:
            print(f"    LogLik={hmm_4.score(X_train_scaled):.2f}  time={elapsed:.1f}s")

    # --- 6e: Bayesian GMM ---
    print("\n  [6e] Bayesian GMM (Dirichlet Process)...")
    t0 = time.perf_counter()
    dpgmm = train_dpgmm(X_train_scaled, max_components=10)
    elapsed = time.perf_counter() - t0

    effective_k = np.sum(dpgmm.weights_ > 0.01)
    print(f"    Effective components: {effective_k} (of 10 max)")
    print(f"    Weights: {np.round(dpgmm.weights_[dpgmm.weights_ > 0.01], 3)}")
    print(f"    time={elapsed:.1f}s")

    print()

    # --------------------------------------------------------
    # STEP 7: Evaluate All Models on Test Set
    # --------------------------------------------------------
    print("[STEP 7/7] Evaluating models on test set...")
    print("-" * 50)

    models_to_eval = {
        f"GMM (k=4)": ("gmm", gmm_4, 4),
        f"GMM (k={best_gmm_k} BIC)": ("gmm", best_gmm_model, best_gmm_k),
    }

    if HAS_HMM:
        if hmm_4 is not None:
            models_to_eval[f"HMM (k=4)"] = ("hmm", hmm_4, 4)
        if best_hmm_model is not None:
            models_to_eval[f"HMM (k={best_hmm_k} BIC)"] = (
                "hmm", best_hmm_model, best_hmm_k
            )

    models_to_eval["Bayesian GMM"] = ("gmm", dpgmm, int(effective_k))

    for model_name, (model_type, model, n_comp) in models_to_eval.items():
        print(f"\n  Evaluating: {model_name}")

        # Predict
        if model_type == "hmm":
            test_labels = model.predict(X_test_scaled)
            train_labels = model.predict(X_train_scaled)
        else:
            test_labels = model.predict(X_test_scaled)
            train_labels = model.predict(X_train_scaled)

        # Auto-label regimes
        regime_map = auto_label_regimes(
            X_train_scaled, train_labels, df_train, FEATURE_NAMES
        )

        # Clustering metrics
        cluster_metrics = compute_clustering_metrics(X_test_scaled, test_labels)
        print(f"    Silhouette:        {cluster_metrics['silhouette']:.4f}")
        print(f"    Calinski-Harabasz: {cluster_metrics['calinski_harabasz']:.1f}")
        print(f"    Davies-Bouldin:    {cluster_metrics['davies_bouldin']:.4f}")

        # Stability metrics
        stability = compute_regime_stability(test_labels)
        print(f"    Mean duration:     {stability['mean_duration_bars']:.1f} bars "
              f"({stability['mean_duration_hours']:.1f} hours)")
        print(f"    Transition rate:   {stability['transition_rate']:.4f}")

        # Supervised metrics
        supervised = compute_supervised_metrics(test_labels, gt_test, n_comp)
        print(f"    Accuracy:          {supervised['accuracy']:.4f}")
        print(f"    Precision (macro): {supervised['precision_macro']:.4f}")
        print(f"    Recall (macro):    {supervised['recall_macro']:.4f}")
        print(f"    F1 (macro):        {supervised['f1_macro']:.4f}")

        # Economic metrics
        economic = compute_economic_metrics(test_labels, df_test)
        for rid, econ in economic.items():
            rname = regime_map.get(rid, {}).get("name", f"Regime {rid}")
            print(f"    {rname}: fwd_ret={econ['mean_fwd_return_bps']:+.2f}bps "
                  f"win_rate={econ['win_rate_pct']:.1f}%")

        # Store results
        all_results[model_name] = {
            "model_type": model_type,
            "n_components": n_comp,
            "clustering": cluster_metrics,
            "stability": stability,
            "supervised": supervised,
            "economic": economic,
            "regime_map": {str(k): v for k, v in regime_map.items()},
        }

        # Generate visualizations
        safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        plot_regime_overlay(
            df_test, test_labels, regime_map,
            f"Regime Overlay — {model_name}",
            os.path.join(RESULTS_DIR, f"regime_overlay_{safe_name}.png"),
            n_bars=min(5000, len(df_test)),
        )

        plot_transition_matrix(
            test_labels, regime_map,
            f"Transition Matrix — {model_name}",
            os.path.join(RESULTS_DIR, f"transition_matrix_{safe_name}.png"),
        )

        if "confusion_matrix" in supervised:
            gt_names = ["LowVol Bull", "Neutral", "HighVol Bear", "LowVol Bear"]
            plot_confusion_matrix(
                np.array(supervised["confusion_matrix"]),
                gt_names[:len(supervised["confusion_matrix"])],
                f"Confusion Matrix — {model_name}",
                os.path.join(RESULTS_DIR, f"confusion_matrix_{safe_name}.png"),
            )

    # BIC comparison chart
    plot_bic_comparison(bic_scores, os.path.join(RESULTS_DIR, "bic_comparison.png"))

    # Model comparison chart
    plot_model_comparison(all_results, os.path.join(RESULTS_DIR, "model_comparison.png"))

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SAVING RESULTS")
    print("=" * 70)

    # Save metrics summary
    metrics_path = os.path.join(RESULTS_DIR, "metrics_summary.json")
    # Convert numpy types for JSON
    def convert_numpy(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    clean_results = json.loads(
        json.dumps(all_results, default=convert_numpy)
    )
    with open(metrics_path, "w") as f:
        json.dump(clean_results, f, indent=2)
    print(f"  Metrics: {metrics_path}")

    # Determine best model
    best_model_name = max(
        all_results,
        key=lambda k: all_results[k]["supervised"]["f1_macro"]
    )
    best_info = all_results[best_model_name]
    best_model_obj = dict(models_to_eval)[best_model_name][1]
    best_model_type = dict(models_to_eval)[best_model_name][0]

    print(f"\n  ★ Best Model: {best_model_name}")
    print(f"    F1={best_info['supervised']['f1_macro']:.4f}  "
          f"Acc={best_info['supervised']['accuracy']:.4f}  "
          f"Silhouette={best_info['clustering']['silhouette']:.4f}")

    # Save best model
    model_path = os.path.join(SCRIPT_DIR, "market_regime_model.pkl")
    joblib.dump(best_model_obj, model_path)
    print(f"  Model saved: {model_path}")

    # Save scaler
    scaler_path = os.path.join(SCRIPT_DIR, "regime_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved: {scaler_path}")

    # Save metadata
    metadata = {
        "model_name": best_model_name,
        "model_type": best_model_type,
        "n_components": best_info["n_components"],
        "features": FEATURE_NAMES,
        "regime_map": best_info["regime_map"],
        "train_date_range": f"{df_train.index[0]} to {df_train.index[-1]}",
        "test_date_range": f"{df_test.index[0]} to {df_test.index[-1]}",
        "train_bars": len(df_train),
        "test_bars": len(df_test),
        "metrics": {
            "accuracy": best_info["supervised"]["accuracy"],
            "precision_macro": best_info["supervised"]["precision_macro"],
            "recall_macro": best_info["supervised"]["recall_macro"],
            "f1_macro": best_info["supervised"]["f1_macro"],
            "silhouette": best_info["clustering"]["silhouette"],
        },
        "trained_at": datetime.now().isoformat(),
    }
    metadata_path = os.path.join(SCRIPT_DIR, "regime_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=convert_numpy)
    print(f"  Metadata saved: {metadata_path}")

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON SUMMARY")
    print("=" * 70)
    print()
    print(f"  {'Model':<25} {'k':>3} {'Accuracy':>9} {'Precision':>10} "
          f"{'Recall':>8} {'F1':>8} {'Silhouette':>11} {'Avg Duration':>13}")
    print("  " + "-" * 98)

    for name, res in all_results.items():
        s = res["supervised"]
        c = res["clustering"]
        st = res["stability"]
        print(f"  {name:<25} {res['n_components']:>3} "
              f"{s['accuracy']:>9.4f} {s['precision_macro']:>10.4f} "
              f"{s['recall_macro']:>8.4f} {s['f1_macro']:>8.4f} "
              f"{c['silhouette']:>11.4f} "
              f"{st['mean_duration_hours']:>10.1f}hrs")

    print()
    print(f"  ★ RECOMMENDED: {best_model_name}")
    print(f"    Accuracy:  {best_info['supervised']['accuracy']:.2%}")
    print(f"    Precision: {best_info['supervised']['precision_macro']:.2%}")
    print(f"    Recall:    {best_info['supervised']['recall_macro']:.2%}")
    print(f"    F1 Score:  {best_info['supervised']['f1_macro']:.2%}")
    print()
    print(f"  Results saved to: {RESULTS_DIR}")
    print(f"  Model saved to:   {model_path}")
    print(f"  Metadata:         {metadata_path}")
    print()
    print("  Pipeline complete. Review the charts in evaluation_results/")
    print("  to make your final decision.")
    print("=" * 70)


if __name__ == "__main__":
    main()
