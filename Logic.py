from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pywt
from scipy.signal import butter, filtfilt, iirnotch, find_peaks

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

FS = 1000

SEG_BEFORE = 100
SEG_AFTER = 200
SEGMENT_LEN = SEG_BEFORE + SEG_AFTER

WAVELETS = ["db1", "db2", "db4"]
WAVELET_LEVEL = 4
TEST_SIZE = 0.30
RANDOM_STATE = 42
IDENTIFY_THRESHOLD = 0.80


def bandpass_filter(sig, lowcut=0.5, highcut=40, fs=FS, order=4):
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, sig)


def preprocess_signal(raw, fs=FS):
    sig = np.asarray(raw, dtype=float)
    sig = np.nan_to_num(sig)
    sig = bandpass_filter(sig, fs=fs)
    sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
    return sig


def detect_rpeaks(sig, fs=FS):
    diff = np.diff(sig, prepend=sig[0])
    squared = diff ** 2
    win = max(1, int(0.15 * fs))
    integrated = np.convolve(squared, np.ones(win) / win, mode="same")
    threshold = np.mean(integrated) + 0.5 * np.std(integrated)
    peaks, _ = find_peaks(integrated, height=threshold, distance=int(0.3 * fs))
    return peaks.astype(int)


def extract_heartbeats(sig, fs=FS):
    r_peaks = detect_rpeaks(sig, fs=fs)
    beats = []
    for r in r_peaks:
        start = r - SEG_BEFORE
        end = r + SEG_AFTER
        if start >= 0 and end <= len(sig):
            beats.append(sig[start:end])
    return np.asarray(beats)


# file processing

def infer_numeric_columns(df):
    cols = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def load_csv_signal(file_path, has_header=True, target_column: Optional[str] = None,
                    target_column_index: Optional[int] = None):
    df = pd.read_csv(file_path, header=0 if has_header else None)

    if target_column is not None and target_column in df.columns:
        return df[target_column].to_numpy(dtype=float)

    if target_column_index is not None and 0 <= target_column_index < df.shape[1]:
        return df.iloc[:, target_column_index].to_numpy(dtype=float)

    best_col = None
    best_non_nan = -1
    for c in df.columns:
        series = pd.to_numeric(df[c], errors="coerce")
        count = int(series.notna().sum())
        if count > best_non_nan:
            best_non_nan = count
            best_col = c

    return (pd.to_numeric(df[best_col], errors="coerce").ffill().bfill().to_numpy(dtype=float))


def list_subject_files(data_root: str, subject_id: str):
    out = []
    for fname in sorted(os.listdir(data_root)):
        if os.path.splitext(fname)[0].startswith(subject_id):
            out.append(os.path.join(data_root, fname))
    return out


def load_subject_beats(data_root, subject_id, max_files: Optional[int] = None, has_header=True,
                       target_column: Optional[str] = None, target_column_index: Optional[int] = None, fs=FS):
    all_beats = []
    files = list_subject_files(data_root, subject_id)
    if max_files is not None:
        files = files[:max_files]

    for file_path in files:
        raw = load_csv_signal(
            file_path,
            has_header=has_header,
            target_column=target_column,
            target_column_index=target_column_index,
        )
        processed = preprocess_signal(raw, fs=fs)
        beats = extract_heartbeats(processed, fs=fs)
        if beats.shape[0] > 0:
            all_beats.append(beats)

    return np.vstack(all_beats)


def load_dataset(data_root, subject_ids, max_files_per_subject: Optional[int] = None, has_header=True,
                 target_column: Optional[str] = None, target_column_index: Optional[int] = None, fs=FS, log_fn=print):
    X_list, y_list = [], []
    for label, sid in enumerate(subject_ids):
        log_fn(f"Loading {sid} (label={label})")
        beats = load_subject_beats(
            data_root,
            sid,
            max_files=max_files_per_subject,
            has_header=has_header,
            target_column=target_column,
            target_column_index=target_column_index,
            fs=fs,
        )
        log_fn(f"  beats: {beats.shape[0]}")
        if beats.shape[0] > 0:
            X_list.append(beats)
            y_list.extend([label] * beats.shape[0])
    return np.vstack(X_list), np.asarray(y_list, dtype=int)


# feature extraction

def wavelet_features_single(segment, wavelet="db1", level=WAVELET_LEVEL):
    coeffs = pywt.wavedec(segment, wavelet, level=level)
    feats = []
    for c in coeffs:
        feats.extend([
            float(np.mean(c)),
            float(np.std(c)),
            float(np.sum(c ** 2)),
            float(np.mean(np.abs(c))),
            float(np.max(np.abs(c))),
            float(np.median(np.abs(c))),
        ])
    return np.asarray(feats, dtype=float)


def extract_features(X, wavelet):
    return np.asarray([wavelet_features_single(x, wavelet=wavelet) for x in X])


def extract_all_wavelets(X, log_fn=print):
    out = {}
    for w in WAVELETS:
        log_fn(f"Extracting wavelet features: {w}")
        F = extract_features(X, wavelet=w)
        log_fn(f"  shape = {F.shape}")
        out[w] = F
    return out


# classification

def get_classifiers():
    return {
        "SVM": [
            ("RBF C=1", SVC(kernel="rbf", C=1, gamma="scale", probability=True, random_state=RANDOM_STATE)),
            ("RBF C=10", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=RANDOM_STATE)),
            ("Poly d=3", SVC(kernel="poly", C=1, degree=3, probability=True, random_state=RANDOM_STATE)),
        ],
        "KNN": [
            ("k=3", KNeighborsClassifier(n_neighbors=3)),
            ("k=5", KNeighborsClassifier(n_neighbors=5)),
            ("k=7", KNeighborsClassifier(n_neighbors=7)),
        ],
        "RandomForest": [
            ("n=50", RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)),
            ("n=100", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
            ("n=200", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
        ],
    }


def train_evaluate_all(X, y, test_size=TEST_SIZE, log_fn=print):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sss.split(X, y))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}
    for clf_name, variants in get_classifiers().items():
        results[clf_name] = {}
        for param_label, clf in variants:
            clf.fit(X_train_s, y_train)
            y_pred = clf.predict(X_test_s)
            acc = accuracy_score(y_test, y_pred)
            results[clf_name][param_label] = {
                "accuracy": acc,
                "clf": clf,
                "scaler": scaler,
                "y_test": y_test,
                "y_pred": y_pred,
                "confusion": confusion_matrix(y_test, y_pred),
                "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
            }
            log_fn(f"  {clf_name} [{param_label}] acc={acc:.4f}")
    return results


def identify_subject(beats_features, clf, scaler, subject_ids, threshold=IDENTIFY_THRESHOLD):
    if beats_features.shape[0] == 0:
        return "Unknown", 0.0
    Xs = scaler.transform(beats_features)
    preds = clf.predict(Xs)
    votes = np.bincount(preds, minlength=len(subject_ids))
    best_label = int(np.argmax(votes))
    confidence = votes[best_label] / len(preds)
    if confidence >= threshold:
        return subject_ids[best_label], confidence
    return "Unknown", confidence


# train

def run_full_training(data_root, subject_ids, max_files_per_subject: Optional[int] = None, has_header=True,
                      target_column: Optional[str] = None, target_column_index: Optional[int] = None, fs=FS,
                      log_fn=print):
    log_fn("[1/4] Loading dataset")
    X, y = load_dataset(
        data_root,
        subject_ids,
        max_files_per_subject=max_files_per_subject,
        has_header=has_header,
        target_column=target_column,
        target_column_index=target_column_index,
        fs=fs,
        log_fn=log_fn,
    )
    log_fn(f"Total beats = {X.shape[0]}")
    log_fn("[2/4] Extracting wavelet features")
    wavelet_features = extract_all_wavelets(X, log_fn=log_fn)

    all_results = {}
    rows = []
    best_acc = -1
    best_model_info = None

    log_fn("[3/4] Training classifiers")
    for wv, F in wavelet_features.items():
        log_fn(f"Wavelet = {wv}")
        results = train_evaluate_all(F, y, log_fn=log_fn)
        all_results[wv] = results
        for clf_name, variants in results.items():
            for param, info in variants.items():
                rows.append({
                    "Wavelet": wv,
                    "Classifier": clf_name,
                    "Parameters": param,
                    "Accuracy (%)": round(info["accuracy"] * 100, 2),
                })
                if info["accuracy"] > best_acc:
                    best_acc = info["accuracy"]
                    best_model_info = (wv, clf_name, param, info)

    df = pd.DataFrame(rows).sort_values(["Wavelet", "Accuracy (%)"], ascending=[True, False])

    wv_best, clf_best, param_best, best_info = best_model_info

    log_fn("[4/4] Training finished")
    return all_results, df, {
        "patient_ids": subject_ids,
        "best_wavelet": wv_best,
        "best_clf": clf_best,
        "best_param": param_best,
        "accuracy": best_acc,
        "clf": best_info["clf"],
        "scaler": best_info["scaler"],
        "all_results_summary": df.to_dict("records"),
    }


def main():
    from gui import main as gui_main
    gui_main()


if __name__ == "__main__":
    main()
