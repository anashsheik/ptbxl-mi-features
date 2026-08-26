import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
CV_FOLDS = 5
METRICS = ["Accuracy", "Precision", "Recall", "F1", "MCC", "AUROC", "AUPRC"]


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "MCC":       matthews_corrcoef(y_true, y_pred),
        "AUROC":     roc_auc_score(y_true, y_prob),
        "AUPRC":     average_precision_score(y_true, y_prob),
    }


def get_round_label(feature, r1, r2, r3, r4, r5):
    if feature in r1: return "R1: Baseline"
    if feature in r2: return "R2: Heart rate"
    if feature in r3: return "R3: QRS morph."
    if feature in r4: return "R4: ST/T-wave"
    if feature in r5: return "R5: P-wave/QT"
    return "?"


def define_rounds(feat_cols):
    r1 = [c for c in feat_cols if c in ["age", "sex", "weight"]
          or c.startswith("axis_") or c.endswith("_flag")]
    r2 = [c for c in feat_cols if c in ["HR_mean", "HR_std", "RR_mean",
          "RR_std", "RR_min", "RR_max", "QTc"]]
    r3 = [c for c in feat_cols if any(c.endswith(s) for s in
          ["_R_amplitude", "_Q_amplitude", "_S_amplitude",
           "_QRS_duration", "_R_peak_time"])]
    r4 = [c for c in feat_cols if any(c.endswith(s) for s in
          ["_ST_level", "_T_amplitude", "_T_duration"])]
    r5 = [c for c in feat_cols if any(c.endswith(s) for s in
          ["_P_amplitude", "_QT_interval"])]
    r6 = r1 + r2 + r3 + r4 + r5
    r7 = r1 + r3
    r8 = r1 + r4
    return r1, r2, r3, r4, r5, r6, r7, r8


def define_models():
    return {
        "LR": {
            "clf": LogisticRegression(random_state=RANDOM_STATE, max_iter=2000),
            "grid": {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"]},
        },
        "DT": {
            "clf": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "grid": {"max_depth": [3, 5, 10, None],
                     "min_samples_split": [2, 5, 10],
                     "criterion": ["gini", "entropy"]},
        },
        "RF": {
            "clf": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            "grid": {"n_estimators": [100, 200],
                     "max_depth": [5, 10, None],
                     "min_samples_split": [2, 5]},
        },
        "SVM": {
            "clf": SVC(random_state=RANDOM_STATE, probability=True),
            "grid": {"C": [0.1, 1, 10], "kernel": ["rbf"],
                     "gamma": ["scale", "auto"]},
        },
    }


def main():
    df_trn = pd.read_csv("train.csv")
    df_val = pd.read_csv("val.csv")
    df_tst = pd.read_csv("test.csv")

    feat_cols = [c for c in df_trn.columns if c not in ["ecg_id", "ground_truth"]]
    y_trn = df_trn["ground_truth"].values
    y_val = df_val["ground_truth"].values
    y_tst = df_tst["ground_truth"].values

    r1, r2, r3, r4, r5, r6, r7, r8 = define_rounds(feat_cols)

    rounds = {
        "R1: Baseline":              r1,
        "R2: Heart rate":            r2,
        "R3: QRS morph.":            r3,
        "R4: ST/T-wave":             r4,
        "R5: P-wave/QT":             r5,
        "R6: All features":          r6,
        "R7: Baseline + QRS morph.": r7,
        "R8: Baseline + ST/T-wave":  r8,
    }

    models = define_models()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print("\nFeature rounds:")
    for name, cols in rounds.items():
        print(f"  {name}: {len(cols)} features")
    n_exp = len(rounds) * len(models)
    print(f"\n{len(rounds)} rounds x {len(models)} models = {n_exp} experiments\n")

    all_results = []
    trained_models = {}

    for rname, rcols in rounds.items():
        scaler = StandardScaler()
        Xtrn = scaler.fit_transform(df_trn[rcols].values)
        Xval = scaler.transform(df_val[rcols].values)
        Xtst = scaler.transform(df_tst[rcols].values)

        print(f"  {rname} ({len(rcols)} features)")

        for mname, cfg in models.items():
            gs = GridSearchCV(cfg["clf"], cfg["grid"], cv=cv,
                              scoring="f1", n_jobs=-1, refit=True)
            gs.fit(Xtrn, y_trn)
            best = gs.best_estimator_

            fold_metrics = {m: [] for m in METRICS}
            for f_trn, f_val in cv.split(Xtrn, y_trn):
                clone = best.__class__(**best.get_params())
                clone.fit(Xtrn[f_trn], y_trn[f_trn])
                yp = clone.predict(Xtrn[f_val])
                ypr = clone.predict_proba(Xtrn[f_val])[:, 1]
                for m, v in compute_metrics(y_trn[f_val], yp, ypr).items():
                    fold_metrics[m].append(v)

            val_m = compute_metrics(y_val, best.predict(Xval),
                                    best.predict_proba(Xval)[:, 1])
            tst_m = compute_metrics(y_tst, best.predict(Xtst),
                                    best.predict_proba(Xtst)[:, 1])

            row = {
                "Round": rname, "N_features": len(rcols), "Model": mname,
                "Best_params": str(gs.best_params_),
            }
            for m in METRICS:
                row[f"CV_{m}_mean"] = round(np.mean(fold_metrics[m]), 4)
                row[f"CV_{m}_std"]  = round(np.std(fold_metrics[m]), 4)
                row[f"Val_{m}"]     = round(val_m[m], 4)
                row[f"Test_{m}"]    = round(tst_m[m], 4)
            all_results.append(row)

            trained_models[(rname, mname)] = {
                "estimator": best,
                "Xtst": Xtst,
                "features": rcols,
            }

            print(f"    {mname:3s}  "
                  f"CV F1={row['CV_F1_mean']:.4f}+-{row['CV_F1_std']:.4f}  "
                  f"Val={row['Val_F1']:.4f}  "
                  f"Test F1={row['Test_F1']:.4f}  "
                  f"AUROC={row['Test_AUROC']:.4f}")

        print()

    df_results = pd.DataFrame(all_results)
    df_results.to_csv("ml_results_individual_rounds.csv", index=False)

    print("\nTest F1 summary:")
    piv = df_results.pivot(index="Round", columns="Model", values="Test_F1")
    piv = piv.reindex(rounds.keys())[["LR", "DT", "RF", "SVM"]]
    print(piv.to_string(float_format="%.4f"))

    print("\nTest AUROC summary:")
    piv2 = df_results.pivot(index="Round", columns="Model", values="Test_AUROC")
    piv2 = piv2.reindex(rounds.keys())[["LR", "DT", "RF", "SVM"]]
    print(piv2.to_string(float_format="%.4f"))

    print("\nRanking (best F1 per round):")
    for rname in rounds.keys():
        sub = df_results[df_results["Round"] == rname]
        b = sub.loc[sub["Test_F1"].idxmax()]
        print(f"  {rname:30s}  {b['Model']:3s}  "
              f"F1={b['Test_F1']:.4f}  AUROC={b['Test_AUROC']:.4f}")

    best_row = df_results.loc[df_results["Test_F1"].idxmax()]
    best_round = best_row["Round"]
    best_model_name = best_row["Model"]

    print(f"\nBest model overall: {best_model_name} on {best_round} "
          f"(Test F1 = {best_row['Test_F1']:.4f})")

    best = trained_models[(best_round, best_model_name)]
    best_est = best["estimator"]
    best_Xtst = best["Xtst"]
    best_features = best["features"]

    y_pred_best = best_est.predict(best_Xtst)
    cm = confusion_matrix(y_tst, y_pred_best)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion matrix ({best_model_name} on {best_round})")
    print(f"  TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"  Precision={tp/(tp+fp):.4f}")
    print(f"  Recall={tp/(tp+fn):.4f}")
    print(f"  F1={2*tp/(2*tp+fp+fn):.4f}")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["Normal", "MI"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title(f"{best_model_name}, {best_round} "
                 f"({len(best_features)} features) — Test set", fontsize=13)
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    color=color, fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=200, bbox_inches="tight")

    print(f"\nPermutation feature importance ({best_model_name}, {best_round})")
    baseline_f1 = f1_score(y_tst, best_est.predict(best_Xtst))
    print(f"  Baseline test F1 = {baseline_f1:.4f}")
    print("  Running 10 repeats")

    result = permutation_importance(
        best_est, best_Xtst, y_tst,
        n_repeats=10, random_state=RANDOM_STATE,
        scoring="f1", n_jobs=-1,
    )

    imp_df = pd.DataFrame({
        "feature": best_features,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
        "round": [get_round_label(f, r1, r2, r3, r4, r5) for f in best_features],
    }).sort_values("importance_mean", ascending=False)

    out_csv = f"permutation_importance_{best_model_name.lower()}.csv"
    imp_df.to_csv(out_csv, index=False)
    print(f"{out_csv}")

    print(f"\n{'Rank':>4} {'Feature':25s} {'Round':18s} {'Mean':>8s} {'Std':>8s}")
    for i, (_, row) in enumerate(imp_df.head(20).iterrows(), 1):
        print(f"{i:4d} {row['feature']:25s} {row['round']:18s} "
              f"{row['importance_mean']:8.4f} {row['importance_std']:8.4f}")

    print("\nRound-level permutation importance:")
    for rnd in ["R1: Baseline", "R2: Heart rate", "R3: QRS morph.",
                "R4: ST/T-wave", "R5: P-wave/QT"]:
        sub = imp_df[imp_df["round"] == rnd]
        if len(sub) == 0:
            continue
        total = sub["importance_mean"].sum()
        top = sub.iloc[0]
        print(f"  {rnd:18s}  total={total:+.4f}  "
              f"top={top['feature']} ({top['importance_mean']:.4f})")


if __name__ == "__main__":
    main()
