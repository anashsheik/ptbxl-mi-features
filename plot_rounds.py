import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_CSV = "ml_results_individual_rounds.csv"
OUTPUT_PNG = "round_comparison.png"
OUTPUT_PDF = "round_comparison.pdf"

MODEL_ORDER = ["LR", "DT", "RF", "SVM"]
MODEL_LABELS = {"LR": "Logistic Regression", "DT": "Decision Tree",
                "RF": "Random Forest", "SVM": "SVM"}
MODEL_COLORS = {"LR": "#1f77b4", "DT": "#d62728",
                "RF": "#2ca02c", "SVM": "#ff7f0e"}

GROUP_ORDER = ["R1: Baseline", "R2: Heart rate", "R3: QRS morph.",
               "R4: ST/T-wave", "R5: P-wave/QT", "R6: All features"]
GROUP_COUNTS = {"R1: Baseline": 15, "R2: Heart rate": 7,
                "R3: QRS morph.": 60, "R4: ST/T-wave": 36,
                "R5: P-wave/QT": 24, "R6: All features": 142}


def main():
    df = pd.read_csv(INPUT_CSV)
    piv = df.pivot(index="Round", columns="Model", values="Test_F1")
    piv = piv.reindex(GROUP_ORDER)[MODEL_ORDER]

    best_r6 = piv.loc["R6: All features"].max()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    n = len(GROUP_ORDER)
    bw = 0.18
    x = np.arange(n)

    for i, m in enumerate(MODEL_ORDER):
        ax.bar(x + i * bw, piv[m].values, bw,
               label=MODEL_LABELS[m], color=MODEL_COLORS[m],
               edgecolor="white", linewidth=0.5)

    ax.axhline(y=best_r6, color="black", linestyle="--",
               linewidth=1, alpha=0.6)
    ax.text(n - 0.4, best_r6 + 0.012, f"R6 best: {best_r6:.4f}",
            fontsize=9, ha="right", style="italic")

    labels = [f"{g}\n({GROUP_COUNTS[g]})" for g in GROUP_ORDER]
    ax.set_xticks(x + 1.5 * bw)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("F1-score (Test-set)", fontsize=12)
    ax.set_title("F1-score per individual feature-group",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_ylim(0, 0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_PDF, bbox_inches="tight")


if __name__ == "__main__":
    main()
