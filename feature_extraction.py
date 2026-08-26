import os
import ast
import time
import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk
import warnings
warnings.filterwarnings("ignore")

# Set the path to your local PTB-XL copy, either by editing the default below
# or by setting the PTBXL_ROOT environment variable before running:
#   export PTBXL_ROOT=/path/to/ptb-xl-1.0.3
PTBXL_ROOT = os.environ.get(
    "PTBXL_ROOT",
    "path/to/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
)
METADATA_CSV = f"{PTBXL_ROOT}/ptbxl_database.csv"
SAMPLING_RATE = 500
DELINEATION_METHOD = "prominence"

OUTPUT_TRAIN = "train.csv"
OUTPUT_VAL   = "val.csv"
OUTPUT_TEST  = "test.csv"

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

MI_CODES = {"IMI", "AMI", "ASMI", "ALMI", "ILMI", "LMI", "IPLMI", "IPMI", "PMI"}


def has_mi(scp_str):
    try:
        d = ast.literal_eval(scp_str)
        return int(any(k in MI_CODES for k in d.keys()))
    except Exception:
        return 0


def is_norm(scp_str):
    try:
        d = ast.literal_eval(scp_str)
        return int("NORM" in d and d["NORM"] >= 50)
    except Exception:
        return 0

def samples_to_ms(n_samples, sr):
    return (n_samples / sr) * 1000.0


def extract_lead_features(ecg_signal, sr, lead_name):
    prefix = lead_name
    features = {}

    try:
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sr, method="neurokit")
        _, rpeaks_info = nk.ecg_peaks(ecg_cleaned, sampling_rate=sr)
        r_peaks = rpeaks_info["ECG_R_Peaks"]

        if len(r_peaks) < 2:
            return {}

        _, waves = nk.ecg_delineate(
            ecg_cleaned, r_peaks,
            sampling_rate=sr,
            method=DELINEATION_METHOD,
            show=False
        )

        n_beats = len(r_peaks)
        r_amps, q_amps, s_amps = [], [], []
        qrs_durs, r_peak_times = [], []
        st_levels = []
        t_amps, t_durs = [], []
        p_amps = []
        qt_intervals = []

        for i in range(n_beats):
            r_idx = r_peaks[i]
            r_amps.append(ecg_cleaned[r_idx])

            q_idx = waves["ECG_Q_Peaks"][i] if i < len(waves["ECG_Q_Peaks"]) else np.nan
            q_amps.append(ecg_cleaned[int(q_idx)] if not np.isnan(q_idx) else np.nan)

            s_idx = waves["ECG_S_Peaks"][i] if i < len(waves["ECG_S_Peaks"]) else np.nan
            s_amps.append(ecg_cleaned[int(s_idx)] if not np.isnan(s_idx) else np.nan)

            t_idx = waves["ECG_T_Peaks"][i] if i < len(waves["ECG_T_Peaks"]) else np.nan
            t_on  = waves["ECG_T_Onsets"][i] if i < len(waves["ECG_T_Onsets"]) else np.nan
            t_off = waves["ECG_T_Offsets"][i] if i < len(waves["ECG_T_Offsets"]) else np.nan
            t_amps.append(ecg_cleaned[int(t_idx)] if not np.isnan(t_idx) else np.nan)
            if not np.isnan(t_on) and not np.isnan(t_off):
                t_durs.append(samples_to_ms(int(t_off) - int(t_on), sr))
            else:
                t_durs.append(np.nan)

            p_idx = waves["ECG_P_Peaks"][i] if i < len(waves["ECG_P_Peaks"]) else np.nan
            p_amps.append(ecg_cleaned[int(p_idx)] if not np.isnan(p_idx) else np.nan)

            r_on  = waves["ECG_R_Onsets"][i] if i < len(waves["ECG_R_Onsets"]) else np.nan
            r_off = waves["ECG_R_Offsets"][i] if i < len(waves["ECG_R_Offsets"]) else np.nan

            if not np.isnan(r_on) and not np.isnan(r_off):
                qrs_durs.append(samples_to_ms(int(r_off) - int(r_on), sr))
            else:
                qrs_durs.append(np.nan)

            if not np.isnan(r_on):
                r_peak_times.append(samples_to_ms(r_idx - int(r_on), sr))
            else:
                r_peak_times.append(np.nan)

            if not np.isnan(r_off):
                j60_idx = int(r_off) + int(0.060 * sr)
                if j60_idx < len(ecg_cleaned):
                    st_levels.append(ecg_cleaned[j60_idx])
                else:
                    st_levels.append(np.nan)
            else:
                st_levels.append(np.nan)

            if not np.isnan(r_on) and not np.isnan(t_off):
                qt_intervals.append(samples_to_ms(int(t_off) - int(r_on), sr))
            else:
                qt_intervals.append(np.nan)

        features[f"{prefix}_R_amplitude"]    = np.nanmedian(r_amps)
        features[f"{prefix}_Q_amplitude"]    = np.nanmedian(q_amps)
        features[f"{prefix}_S_amplitude"]    = np.nanmedian(s_amps)
        features[f"{prefix}_QRS_duration"]   = np.nanmedian(qrs_durs)
        features[f"{prefix}_R_peak_time"]    = np.nanmedian(r_peak_times)
        features[f"{prefix}_ST_level"]       = np.nanmedian(st_levels)
        features[f"{prefix}_T_amplitude"]    = np.nanmedian(t_amps)
        features[f"{prefix}_T_duration"]     = np.nanmedian(t_durs)
        features[f"{prefix}_P_amplitude"]    = np.nanmedian(p_amps)
        features[f"{prefix}_QT_interval"]    = np.nanmedian(qt_intervals)

    except Exception:
        return {}

    return features


def extract_global_features(ecg_cleaned_lead2, r_peaks, sr, qt_intervals_lead2):
    features = {}
    if len(r_peaks) < 2:
        return {}

    rr_intervals = np.diff(r_peaks) / sr * 1000.0
    hr_values = 60000.0 / rr_intervals

    features["HR_mean"] = np.mean(hr_values)
    features["HR_std"]  = np.std(hr_values)
    features["RR_mean"] = np.mean(rr_intervals)
    features["RR_std"]  = np.std(rr_intervals)
    features["RR_min"]  = np.min(rr_intervals)
    features["RR_max"]  = np.max(rr_intervals)

    qt_vals = [v for v in qt_intervals_lead2 if not np.isnan(v)]
    if len(qt_vals) > 0 and np.mean(rr_intervals) > 0:
        features["QTc"] = np.median(qt_vals) / np.sqrt(np.mean(rr_intervals) / 1000.0)
    else:
        features["QTc"] = np.nan

    return features


def extract_baseline_features(row):
    features = {}
    features["age"]    = row.get("age", np.nan)
    features["sex"]    = row.get("sex", np.nan)
    features["weight"] = row.get("weight", np.nan)

    axis_val = row.get("heart_axis", np.nan)
    for cat in ["MID", "LAD", "ALAD", "RAD", "ARAD", "AXL", "AXR"]:
        features[f"axis_{cat}"] = int(axis_val == cat) if not pd.isna(axis_val) else 0
    features["axis_UNKNOWN"] = int(pd.isna(axis_val))

    for col in ["baseline_drift", "static_noise", "burst_noise", "electrodes_problems"]:
        features[f"{col}_flag"] = int(pd.notna(row.get(col, np.nan)))

    return features


def process_record(row, ptbxl_root):
    ecg_id = int(row["ecg_id"])
    record_path = os.path.join(ptbxl_root, row["filename_hr"])

    try:
        record = wfdb.rdrecord(record_path)
        signals = record.p_signal
    except Exception:
        return None

    if signals is None or signals.shape[1] != 12:
        return None

    all_features = {"ecg_id": ecg_id}
    all_features.update(extract_baseline_features(row))

    all_lead_features = {}
    lead2_rpeaks = None
    lead2_cleaned = None
    lead2_qt = []

    for lead_idx, lead_name in enumerate(LEAD_NAMES):
        lead_feats = extract_lead_features(signals[:, lead_idx], SAMPLING_RATE, lead_name)
        if not lead_feats:
            return None
        all_lead_features.update(lead_feats)

        if lead_name == "II":
            try:
                ecg_cleaned = nk.ecg_clean(signals[:, lead_idx], sampling_rate=SAMPLING_RATE)
                _, rpeaks_info = nk.ecg_peaks(ecg_cleaned, sampling_rate=SAMPLING_RATE)
                lead2_rpeaks = rpeaks_info["ECG_R_Peaks"]
                lead2_cleaned = ecg_cleaned
                _, waves = nk.ecg_delineate(ecg_cleaned, lead2_rpeaks,
                    sampling_rate=SAMPLING_RATE, method=DELINEATION_METHOD, show=False)
                lead2_qt = []
                for i in range(len(lead2_rpeaks)):
                    r_on  = waves["ECG_R_Onsets"][i] if i < len(waves["ECG_R_Onsets"]) else np.nan
                    t_off = waves["ECG_T_Offsets"][i] if i < len(waves["ECG_T_Offsets"]) else np.nan
                    if not np.isnan(r_on) and not np.isnan(t_off):
                        lead2_qt.append(samples_to_ms(int(t_off) - int(r_on), SAMPLING_RATE))
                    else:
                        lead2_qt.append(np.nan)
            except Exception:
                lead2_rpeaks = None

    if lead2_rpeaks is not None and lead2_cleaned is not None:
        global_feats = extract_global_features(lead2_cleaned, lead2_rpeaks, SAMPLING_RATE, lead2_qt)
        if not global_feats:
            return None
        all_features.update(global_feats)
    else:
        return None

    all_features.update(all_lead_features)
    return all_features


def main():
    print(f"Feature extraction: {SAMPLING_RATE} Hz, {DELINEATION_METHOD} delineation")

    print("\nLoading data")
    df = pd.read_csv(METADATA_CSV)
    print(f"  Total records: {len(df)}")

    df["label_MI"]   = df["scp_codes"].apply(has_mi)
    df["label_NORM"] = df["scp_codes"].apply(is_norm)
    df_binary = df[(df["label_MI"] == 1) | (df["label_NORM"] == 1)].copy()
    df_binary = df_binary[~((df_binary["label_MI"] == 1) & (df_binary["label_NORM"] == 1))].copy()
    df_binary["target"] = df_binary["label_MI"]
    print(f"  MI vs NORM: {len(df_binary)} (MI={int(df_binary['target'].sum())}, NORM={int((df_binary['target']==0).sum())})")

    n_bad_age = (df_binary["age"] > 100).sum()
    df_binary = df_binary[df_binary["age"] <= 100].copy()
    print(f"  After age filter: {len(df_binary)} ({n_bad_age} removed)")

    print(f"\nExtracting features ({len(df_binary)} records)")
    all_rows, targets, folds = [], [], []
    n_ok, n_fail = 0, 0
    t0 = time.time()

    for idx, (_, row) in enumerate(df_binary.iterrows()):
        if (idx + 1) % 500 == 0:
            print(f"  {idx+1}/{len(df_binary)} ({n_ok} ok, {n_fail} skip)")

        result = process_record(row, PTBXL_ROOT)
        if result is not None:
            all_rows.append(result)
            targets.append(float(row["target"]))
            folds.append(int(row["strat_fold"]))
            n_ok += 1
        else:
            n_fail += 1

    print(f"  Done: {n_ok} ok, {n_fail} failed")

    print("\n Records dropped")
    df_feat = pd.DataFrame(all_rows)
    df_feat["strat_fold"] = folds
    df_feat["ground_truth"] = targets

    n_before = len(df_feat)
    df_feat = df_feat.dropna().reset_index(drop=True)
    print(f"  {n_before} __> {len(df_feat)} ({n_before - len(df_feat)} dropped)")

    feat_cols = [c for c in df_feat.columns if c not in ["ecg_id", "strat_fold", "ground_truth"]]
    col_order = ["ecg_id"] + feat_cols + ["ground_truth"]

    df_trn = df_feat[df_feat["strat_fold"].isin(range(1, 9))][col_order].reset_index(drop=True)
    df_val = df_feat[df_feat["strat_fold"] == 9][col_order].reset_index(drop=True)
    df_tst = df_feat[df_feat["strat_fold"] == 10][col_order].reset_index(drop=True)

    for name, s in [("Train (1-8)", df_trn), ("Val (9)", df_val), ("Test (10)", df_tst)]:
        mi = int((s["ground_truth"] == 1.0).sum())
        print(f"  {name:12s}: {len(s):5d} (MI={mi}, Normal={len(s)-mi}, ratio={mi/len(s):.3f})")

    df_trn.to_csv(OUTPUT_TRAIN, index=False)
    df_val.to_csv(OUTPUT_VAL, index=False)
    df_tst.to_csv(OUTPUT_TEST, index=False)

    print(f"\n{OUTPUT_TRAIN} ({len(df_trn)}), {OUTPUT_VAL} ({len(df_val)}), {OUTPUT_TEST} ({len(df_tst)})")


if __name__ == "__main__":
    main()
