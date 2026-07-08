import os
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report


def train_and_evaluate_probe(data_path: str):
    """Trains a logistic regression probe on extracted hidden states to predict trajectory success."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file {data_path} not found. Run extractor.py first."
        )

    data = torch.load(data_path)
    X = data["X"].numpy()
    y = data["y"].numpy()

    print(f"Loaded dataset: X shape {X.shape}, y shape {y.shape}")

    # Split data (80% train, 20% test)
    # We use tokens as independent samples here, which is a simplification.
    # A more rigorous approach would be splitting by question.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Logistic Regression probe...")
    # L2 regularization, max_iter increased for convergence
    C_values = [0.01, 0.1, 1.0, 10.0]
    results = {}

    for C_val in C_values:
        print(f"\nTraining Logistic Regression probe with C={C_val}...")
        clf = LogisticRegression(C=C_val, max_iter=1000, random_state=42, solver='liblinear') # Added solver for explicit L2
        clf.fit(X_train, y_train)

        y_train_pred = clf.predict(X_train)
        y_train_prob = clf.predict_proba(X_train)[:, 1]
        train_acc = accuracy_score(y_train, y_train_pred)
        train_auc = roc_auc_score(y_train, y_train_prob)
        
        y_test_pred = clf.predict(X_test)
        y_test_prob = clf.predict_proba(X_test)[:, 1]
        test_acc = accuracy_score(y_test, y_test_pred)
        test_auc = roc_auc_score(y_test, y_test_prob)

        results[C_val] = {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "test_auc_roc": test_auc,
        }
        
        print(f"--- Results for C={C_val} ---")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Testing Accuracy: {test_acc:.4f}")
        print(f"Testing AUC-ROC:  {test_auc:.4f}")
    
    # Store results for reporting
    global experiment_results
    experiment_results = results
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n--- Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC-ROC:  {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return clf


if __name__ == "__main__":
    train_and_evaluate_probe("data/traces.pt")
