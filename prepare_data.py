import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

def labels_are_numeric(y):
    try:
        np.asarray(y, dtype=np.int64)
        return True
    except (ValueError, TypeError):
        return False

def encode_labels(y):
    y = np.asarray(y)
    if labels_are_numeric(y):
        return y.astype(np.int64), None

    encoder = LabelEncoder()
    out = encoder.fit_transform(y.astype(str)).astype(np.int64)
    mapping = {str(c): int(i) for i, c in enumerate(encoder.classes_)}
    print(f"[prepare] encoded label column ({len(mapping)} classes)")
    return out, mapping

def resolve_benign_id(benign_class, benign_label, label_encode_map):
    """Return encoded class id for benign, or None to use frequency remapping."""
    if benign_label is not None:
        if label_encode_map is not None:
            key = str(benign_label)
            if key not in label_encode_map:
                raise ValueError(
                    f"--benign-label {benign_label!r} not in labels: "
                    f"{list(label_encode_map.keys())}"
                )
            return label_encode_map[key]
        try:
            return int(benign_label)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"--benign-label {benign_label!r} is not a valid class name or id"
            ) from exc
    if benign_class is not None:
        return int(benign_class)
    return None

def remap_labels_by_frequency(y):
    values, counts = np.unique(y, return_counts=True)
    order = values[np.argsort(-counts)]
    mapping = {int(v): i for i, v in enumerate(order)}
    out = np.array([mapping[int(v)] for v in y], dtype=np.int64)
    return out, mapping

def remap_benign_to_zero(y, benign_id):
    y = np.asarray(y, dtype=np.int64)
    benign_id = int(benign_id)
    others = sorted(int(c) for c in np.unique(y) if c != benign_id)
    mapping = {benign_id: 0}
    for i, c in enumerate(others, start=1):
        mapping[c] = i
    out = np.array([mapping[int(v)] for v in y], dtype=np.int64)
    return out, mapping

def remap_labels_for_cold(y, benign_id=None):
    if benign_id is not None:
        return remap_benign_to_zero(y, benign_id)
    return remap_labels_by_frequency(y)

def encode_and_scale_features(df, label_col, scaler="minmax"):
    data = df.copy()
    if label_col not in data.columns:
        raise KeyError(f"label column {label_col!r} not in CSV columns")

    num_cols = data.select_dtypes(include=["int", "float"]).columns.tolist()
    cat_cols = data.select_dtypes(include=["object", "string"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != label_col]

    if num_cols:
        if scaler == "minmax":
            scale = MinMaxScaler()
        elif scaler == "standard":
            scale = StandardScaler()
        else:
            raise ValueError(f"scaler must be minmax or standard, got {scaler!r}")
        data[num_cols] = scale.fit_transform(data[num_cols])

    for col in cat_cols:
        enc = LabelEncoder()
        data[col] = enc.fit_transform(data[col].astype(str))

    X = data.drop(columns=[label_col])
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = np.nan_to_num(X.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return X

def save_preprocessed_csv(path, X, y, label_col="Label"):
    out = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    out[label_col] = y
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out.to_csv(path, index=False)
    print(f"[prepare] saved CSV -> {path}  (rows={len(out)}, cols={len(out.columns)})")

def save_arrays(out_dir, X_train, y_train, X_test, y_test):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X_train.npy"), X_train.astype(np.float32))
    np.save(os.path.join(out_dir, "y_train.npy"), y_train.astype(np.int64))
    np.save(os.path.join(out_dir, "X_test.npy"), X_test.astype(np.float32))
    np.save(os.path.join(out_dir, "y_test.npy"), y_test.astype(np.int64))
    print(f"[prepare] wrote npy -> {out_dir}/")
    print(
        f"[prepare]   train={len(y_train)} test={len(y_test)} "
        f"features={X_train.shape[1]} classes={len(np.unique(y_train))}"
    )
    print(f"[prepare]   train class-0 count={int(np.sum(y_train == 0))}")

def prepare(args):
    df = pd.read_csv(args.csv)
    label_col = args.label_col
    
    # -----------------------------------------------------------------
    # NEW: Filter to select only samples where keep == False
    # -----------------------------------------------------------------
    if "keep" in df.columns:
        # Filter rows
        df = df[df["keep"] == False]
        
        # CRITICAL: Drop the 'keep' column so it doesn't get treated 
        # as a mathematical feature in encode_and_scale_features()
        df = df.drop(columns=["keep"])
        
        print(f"[prepare] Filtered CSV. Processing {len(df)} samples where keep == False")
    else:
        print("[prepare] WARNING: 'keep' column not found in CSV. Processing all rows.")
    # -----------------------------------------------------------------

    if args.skip_feature_preprocess:
        X = df.drop(columns=[label_col]).to_numpy(dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        print("[prepare] skipped feature encoding and scaling")
    else:
        X = encode_and_scale_features(df, label_col, scaler=args.scaler)

    y, label_encode_map = encode_labels(df[label_col].to_numpy())
    benign_id = resolve_benign_id(args.benign_class, args.benign_label, label_encode_map)

    if args.save_csv:
        save_preprocessed_csv(args.save_csv, X, y, label_col)

    if benign_id is not None:
        print(f"[prepare] benign class id (before remap): {benign_id}")
        y, mapping = remap_labels_for_cold(y, benign_id)
    else:
        print("[prepare] no benign specified; mapping majority class -> 0")
        y, mapping = remap_labels_for_cold(y, None)

    if label_encode_map is not None:
        print(f"[prepare] raw label -> id: {label_encode_map}")
    print(f"[prepare] class remap: {mapping}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y if len(np.unique(y)) > 1 else None,
    )
    save_arrays(args.out_dir, X_train, y_train, X_test, y_test)

def main():
    p = argparse.ArgumentParser(description="Prepare labelled CSV for CoLD (.npy)")
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--label-col", type=str, default="Label")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--skip-feature-preprocess", action="store_true",
        help="Skip feature encoding/scaling (features must already be numeric)",
    )
    p.add_argument("--scaler", choices=["minmax", "standard"], default="minmax")
    p.add_argument(
        "--benign-class", type=int, default=None,
        help="Class id to map to 0 (after label encoding, if any)",
    )
    p.add_argument(
        "--benign-label", type=str, default=None,
        help="Class name to map to 0 (for string labels in CSV)",
    )
    p.add_argument(
        "--save-csv", type=str, default=None,
        help="Write feature-processed CSV (labels before class-0 remap)",
    )
    prepare(p.parse_args())

if __name__ == "__main__":
    main()
















