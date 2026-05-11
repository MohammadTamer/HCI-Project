from __future__ import annotations
import os
import warnings
import numpy as np
import pandas as pd
import pywt
from scipy.fftpack import dct
import statsmodels.api as sm
from scipy.signal import butter, filtfilt, iirnotch, find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FS = 250

SEG_BEFORE = 100
SEG_AFTER = 200
SEGMENT_LEN = SEG_BEFORE + SEG_AFTER

WAVELETS = ["db1", "db2", "db4"]
WAVELET_LEVEL = 4
TEST_SIZE = 0.30
RANDOM_STATE = 42
IDENTIFY_THRESHOLD = 0.80

MODEL_STATE = {
    "clf": None,
    "scaler": None,
    "subject_ids": None,
    "wavelet": "db1",
}


# file processing

def load_csv_signal(file_path):
    return pd.read_csv(file_path).iloc[:, 0].to_numpy(dtype=float)


def load_dataset(data_root, subject_ids, fs=FS, log_fn=print):
    X_list = []
    y_list = []
    for label, sid in enumerate(subject_ids):
        log_fn(f"Loading {sid}")
        subject_beats = []
        files = sorted([
            os.path.join(data_root, f)
            for f in os.listdir(data_root)
            if f.startswith(sid)
        ])
        for file_path in files:
            raw = pd.read_csv(file_path).iloc[:, 0].to_numpy(dtype=float)
            processed = preprocess_signal(raw, fs)
            beats = extract_heartbeats(processed, fs)
            subject_beats.append(beats)
        subject_beats = np.vstack(subject_beats)
        X_list.append(subject_beats)
        y_list.extend([label] * subject_beats.shape[0])
    return np.vstack(X_list), np.array(y_list)


# data cleaning and beats extraction

def bandpass_filter(sig, lowcut=0.5, highcut=40, fs=FS, order=2):
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
    integrated = np.convolve(squared, np.ones(win) / win)
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


# feature extraction

def fiducial_features(segment, fs=FS):
    r_peaks = detect_rpeaks(segment, fs)
    qr_list = []
    rs_list = []
    slope_list = []
    window = int(0.06 * fs)
    for r in r_peaks:
        left = segment[r - window:r]
        right = segment[r:r + window]
        q = (r - window) + np.argmin(left)
        s = r + np.argmin(right)
        qr = r - q
        rs = s - r
        dt = max(s - q, 1)
        slope = (segment[s] - segment[q]) / dt
        qr_list.append(qr)
        rs_list.append(rs)
        slope_list.append(slope)

    return np.array([
        np.mean(qr_list),
        np.mean(rs_list),
        np.mean(slope_list),
    ])


def non_fiducial_features(segment, wavelet, level=WAVELET_LEVEL):
    coeffs = pywt.wavedec(segment, wavelet, level=level)
    wavelet_feats = []
    for c in coeffs:
        wavelet_feats.extend([
            np.mean(c),
            np.std(c),
            np.sum(c ** 2),
            np.mean(np.abs(c)),
            np.max(np.abs(c)),
            np.median(np.abs(c)),
        ])
    return np.concatenate([
        wavelet_feats,
    ])


def extract_features(X, wavelet):
    features = []
    for segment in X:
        fid_feats = fiducial_features(segment)
        nonfid_feats = non_fiducial_features(segment, wavelet=wavelet)
        combined = np.concatenate([fid_feats, nonfid_feats])
        features.append(combined)
    return np.asarray(features)


def extract_all_wavelets(X, log_fn=print):
    out = {}
    for w in WAVELETS:
        log_fn(f"Extracting wavelet features: {w}")
        F = extract_features(X, wavelet=w)
        log_fn(f"  shape = {F.shape}")
        out[w] = F
    return out


# training

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


def train_evaluate_all(X, y, log_fn=print):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE)
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


def run_full_training(data_root, subject_ids, fs=FS, log_fn=print):
    log_fn("[1/4] Loading dataset")
    X, y = load_dataset(data_root, subject_ids, fs=fs, log_fn=log_fn)
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

    MODEL_STATE["clf"] = best_info["clf"]
    MODEL_STATE["scaler"] = best_info["scaler"]
    MODEL_STATE["subject_ids"] = subject_ids
    MODEL_STATE["wavelet"] = wv_best

    log_fn("[4/4] Training finished")
    return df


# prediction

def predict_subject_from_file(csv_path, fs=FS):
    sig = load_csv_signal(csv_path)
    processed = preprocess_signal(sig, fs=fs)
    beats = extract_heartbeats(processed, fs=fs)
    wavelet_name = MODEL_STATE["wavelet"]
    feats = extract_features(beats, wavelet=wavelet_name)
    name, conf = identify_subject(
        feats,
        MODEL_STATE["clf"],
        MODEL_STATE["scaler"],
        MODEL_STATE["subject_ids"]
    )
    return {
        "signal": processed,
        "beats": beats,
        "name": name,
        "confidence": conf,
        "wavelet": wavelet_name,
    }


def identify_subject(beats_features, clf, scaler, subject_ids):
    preds = clf.predict(scaler.transform(beats_features))
    best_label = np.argmax(np.bincount(preds))
    confidence = np.mean(preds == best_label)
    if confidence >= 0.80:
        return subject_ids[best_label], confidence
    return "Unknown", confidence


def main():
    from gui import main as gui_main
    gui_main()


if __name__ == "__main__":
    main()
