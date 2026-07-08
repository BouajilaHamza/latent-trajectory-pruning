# src/multi_layer_probe.py
"""Multi-layer probing for Task 1.3: Layer-wise Conviction Analysis.

Loads multi-layer traces from the extractor and trains a separate logistic
regression probe per layer. Reports AUC-ROC and accuracy per layer to
identify the earliest layer that reliably predicts trajectory failure.
"""

import os
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score


def train_multi_layer_probes(
    data_path: str = "data/multi_layer_traces.pt",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[int, dict]:
    """Trains and evaluates a logistic regression probe per layer.

    Args:
        data_path: Path to the multi-layer traces file.
        test_size: Fraction of data for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Dict mapping layer_idx -> {"accuracy": float, "auc_roc": float, "n_samples": int}
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file {data_path} not found. Run multi_layer_extractor.py first."
        )

    data = torch.load(data_path, weights_only=False)
    results = {}
    layers = sorted(data.keys())

    print(f"Training probes for {len(layers)} layers...")
    print(f"{'Layer':>6} | {'N':>6} | {'Accuracy':>8} | {'AUC-ROC':>8} | {'Signal'}")
    print("-" * 55)

    for layer_idx in layers:
        X = data[layer_idx]["X"].numpy()
        y = data[layer_idx]["y"].numpy()

        n_pos = y.sum()
        n_neg = len(y) - n_pos

        # Need both classes to compute AUC
        if n_pos == 0 or n_neg == 0:
            print(
                f"{layer_idx:>6} | {len(y):>6} | {'N/A':>8} | {'N/A':>8} | "
                f"Single class only (pos={n_pos}, neg={n_neg})"
            )
            results[layer_idx] = {
                "accuracy": None,
                "auc_roc": None,
                "n_samples": len(y),
                "skipped": True,
            }
            continue

        # Stratified split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        except ValueError:
            # Too few samples for stratified split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

        clf = LogisticRegression(
            max_iter=1000, random_state=random_state, solver="lbfgs"
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)

        # AUC-ROC needs both classes in test set
        if len(set(y_test)) > 1:
            auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            auc = None

        # Signal strength indicator
        if auc is not None:
            if auc >= 0.80:
                signal = "★★★ STRONG"
            elif auc >= 0.65:
                signal = "★★  MODERATE"
            elif auc >= 0.55:
                signal = "★   WEAK"
            else:
                signal = "·   NOISE"
        else:
            signal = "?   UNKNOWN"

        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"{layer_idx:>6} | {len(y):>6} | {acc:>8.4f} | {auc_str:>8} | {signal}")

        results[layer_idx] = {
            "accuracy": acc,
            "auc_roc": auc,
            "n_samples": len(y),
            "skipped": False,
        }

    # Summary
    valid_results = {k: v for k, v in results.items() if v.get("auc_roc") is not None}
    if valid_results:
        best_layer = max(valid_results, key=lambda k: valid_results[k]["auc_roc"])
        best_auc = valid_results[best_layer]["auc_roc"]

        # Find earliest layer with AUC within 5% of best
        threshold = best_auc * 0.95
        earliest_good = min(
            (k for k, v in valid_results.items() if v["auc_roc"] >= threshold),
            default=best_layer,
        )

        print("\n" + "=" * 55)
        print(f"Best layer:     {best_layer} (AUC={best_auc:.4f})")
        print(
            f"Earliest strong: {earliest_good} "
            f"(AUC={valid_results[earliest_good]['auc_roc']:.4f}, "
            f"within 5% of best)"
        )
        print(
            f"Implication:    Signal emerges by layer {earliest_good}/"
            f"{max(layers)} ({earliest_good/max(layers)*100:.0f}% depth)"
        )
    else:
        print("\nNo valid AUC-ROC scores computed. Need more data with both classes.")

    return results


if __name__ == "__main__":
    train_multi_layer_probes("data/multi_layer_traces.pt")
