import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve, average_precision_score

RANDOM_STATE = 42
OUTPUT_PNG = "pr_curve.png"
OUTPUT_PDF = "pr_curve.pdf"

MODEL_COLORS = {"LR": "#1f77b4", "DT": "#d62728",
                "RF": "#2ca02c", "SVM": "#ff7f0e"}
MODEL_LABELS = {"LR": "LR", "DT": "DT",
                "RF": "RF", "SVM": "SVM"}


def build_models():
    return {
        "LR": LogisticRegression(C=0.1, penalty="l2",
                                 random_state=RANDOM_STATE, max_iter=2000),
        "DT": DecisionTreeClassifier(max_depth=None, min_samples_split=10,
                                     criterion="gini", random_state=RANDOM_STATE),
        "RF": RandomForestClassifier(n_estimators=200, max_depth=None,
                                     min_samples_split=5,
                                     random_state=RANDOM_STATE, n_jobs=-1),
        "SVM": SVC(C=10, kernel="rbf", gamma="scale",
                   random_state=RANDOM_STATE, probability=True),
    }


def main():
    df_trn = pd.read_csv("train.csv")
    df_tst = pd.read_csv("test.csv")
    feat_cols = [c for c in df_trn.columns
                 if c not in ["ecg_id", "ground_truth"]]

    y_trn = df_trn["ground_truth"].values
    y_tst = df_tst["ground_truth"].values
    baseline = y_tst.mean()

    sc = StandardScaler()
    X_trn = sc.fit_transform(df_trn[feat_cols].values)
    X_tst = sc.transform(df_tst[feat_cols].values)

    results = []
    for name, clf in build_models().items():
        clf.fit(X_trn, y_trn)
        y_prob = clf.predict_proba(X_tst)[:, 1]
        prec, rec, _ = precision_recall_curve(y_tst, y_prob)
        ap = average_precision_score(y_tst, y_prob)
        results.append((name, rec, prec, ap))

    results.sort(key=lambda r: -r[3])

    fig, ax = plt.subplots(figsize=(7, 6.5))

    for name, rec, prec, ap in results:
        ax.plot(rec, prec, linewidth=2.2, color=MODEL_COLORS[name],
                label=f"{MODEL_LABELS[name]} (AUPRC = {ap:.4f})")

    ax.axhline(y=baseline, color="k", linestyle="--", linewidth=1, alpha=0.4,
               label=f"Baseline ({baseline:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("PR-Curves on all features (Test Set)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_PDF, bbox_inches="tight")


if __name__ == "__main__":
    main()