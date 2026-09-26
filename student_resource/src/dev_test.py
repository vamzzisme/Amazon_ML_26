"""
Quick dev test: load ALL data but block/score only a small S1 subset.
This gives realistic blocking recall because all S2/S3 records are present.
"""
import os, sys, time
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import normalize_name, normalize_address, extract_street_number, extract_pin_zip
from blocking import run_blocking
from features import compute_features_batch, add_competition_features
from scorer import score_submission, load_ground_truth
from decoder import decode_predictions

BASE = os.path.dirname(os.path.dirname(__file__))
TRAIN_DIR = os.path.join(BASE, 'dataset', 'train')


def load_preprocess(path):
    print(f"Loading {os.path.basename(path)}...")
    df = pd.read_csv(path, sep='\t', dtype=str).fillna('')
    t0 = time.time()
    df['name_norm'] = df['business_name'].apply(normalize_name)
    df['addr_norm'] = df['business_address'].apply(normalize_address)
    df['street_num'] = df['addr_norm'].apply(extract_street_number)
    df['pin_zip'] = df['addr_norm'].apply(extract_pin_zip)
    print(f"  {len(df)} records preprocessed in {time.time()-t0:.1f}s")
    return df


def run_dev_test(n_s1=5000, country='US', top_k=15):
    """
    Take n_s1 random US S1 records, use ALL S2/S3 US records.
    Split those S1 into train (80%) and dev (20%).
    """
    t_start = time.time()

    # Load all data
    s1_all = load_preprocess(os.path.join(TRAIN_DIR, 'train_source1.tsv'))
    s2_all = load_preprocess(os.path.join(TRAIN_DIR, 'train_source2.tsv'))
    s3_all = load_preprocess(os.path.join(TRAIN_DIR, 'train_source3.tsv'))
    truth = load_ground_truth(os.path.join(TRAIN_DIR, 'train_ground_truth.tsv'))

    # Filter to one country to fit in memory
    s1_c = s1_all[s1_all['country'] == country].reset_index(drop=True)
    s23_c = pd.concat([
        s2_all[s2_all['country'] == country],
        s3_all[s3_all['country'] == country]
    ], ignore_index=True)
    del s1_all, s2_all, s3_all  # free RAM

    print(f"\n{country}: {len(s1_c)} S1, {len(s23_c)} S23")

    # Sample S1
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(s1_c), size=min(n_s1, len(s1_c)), replace=False)
    s1_sample = s1_c.iloc[sample_idx].reset_index(drop=True)

    # 80/20 split
    split = int(len(s1_sample) * 0.8)
    s1_train = s1_sample.iloc[:split].reset_index(drop=True)
    s1_dev = s1_sample.iloc[split:].reset_index(drop=True)

    truth_train = {eid: truth.get(eid, set()) for eid in s1_train['entity_id']}
    truth_dev = {eid: truth.get(eid, set()) for eid in s1_dev['entity_id']}

    print(f"Train S1: {len(s1_train)}, Dev S1: {len(s1_dev)}")

    # ── BLOCKING (train) ──
    print("\n=== BLOCKING (train) ===")
    cands_train = run_blocking(s1_train, s23_c, top_k=top_k)

    # Measure blocking recall
    s23_id_to_idx = {eid: i for i, eid in enumerate(s23_c['entity_id'].values)}
    hit, total = 0, 0
    for s1_idx in range(len(s1_train)):
        for m in truth_train.get(s1_train['entity_id'].iloc[s1_idx], set()):
            total += 1
            if m in s23_id_to_idx and s23_id_to_idx[m] in cands_train.get(s1_idx, set()):
                hit += 1
    print(f"Blocking recall (train): {hit}/{total} = {hit/max(total,1):.4f}")

    # ── TRAINING PAIRS ──
    pairs, labels = [], []
    for s1_idx, s23_set in cands_train.items():
        s1_eid = s1_train['entity_id'].iloc[s1_idx]
        true_m = truth_train.get(s1_eid, set())
        for s23_idx in s23_set:
            s23_eid = s23_c['entity_id'].iloc[s23_idx]
            pairs.append((s1_idx, s23_idx))
            labels.append(1 if s23_eid in true_m else 0)
        # Add missed true matches
        for tm in true_m:
            if tm in s23_id_to_idx and s23_id_to_idx[tm] not in s23_set:
                pairs.append((s1_idx, s23_id_to_idx[tm]))
                labels.append(1)

    labels = np.array(labels, dtype=np.int32)
    print(f"Pairs: {len(pairs)} ({labels.sum()} pos, {len(labels)-labels.sum()} neg)")

    # ── FEATURES ──
    print("\n=== FEATURES (train) ===")
    X, fn = compute_features_batch(pairs, s1_train, s23_c)
    X, fn = add_competition_features(X, pairs, s1_train, s23_c, fn)
    print(f"Shape: {X.shape}")

    # ── LIGHTGBM ──
    print("\n=== TRAINING ===")
    ds = lgb.Dataset(X, label=labels, feature_name=fn)
    params = {
        'objective': 'binary', 'metric': 'binary_logloss',
        'learning_rate': 0.05, 'num_leaves': 63,
        'min_child_samples': 50, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5,
        'scale_pos_weight': (len(labels) - labels.sum()) / max(labels.sum(), 1),
        'verbose': -1, 'n_jobs': -1,
    }
    model = lgb.train(params, ds, num_boost_round=300)

    imp = model.feature_importance(importance_type='gain')
    for name, score in sorted(zip(fn, imp), key=lambda x: -x[1])[:8]:
        print(f"  {name:25s} {score:.0f}")

    # ── DEV EVAL ──
    print("\n=== BLOCKING (dev) ===")
    cands_dev = run_blocking(s1_dev, s23_c, top_k=top_k)

    hit, total = 0, 0
    for s1_idx in range(len(s1_dev)):
        for m in truth_dev.get(s1_dev['entity_id'].iloc[s1_idx], set()):
            total += 1
            if m in s23_id_to_idx and s23_id_to_idx[m] in cands_dev.get(s1_idx, set()):
                hit += 1
    print(f"Blocking recall (dev): {hit}/{total} = {hit/max(total,1):.4f}")

    dev_pairs = []
    for s1_idx, s23_set in cands_dev.items():
        for s23_idx in s23_set:
            dev_pairs.append((s1_idx, s23_idx))

    print(f"\n=== FEATURES (dev) ===")
    X_dev, fn_dev = compute_features_batch(dev_pairs, s1_dev, s23_c)
    X_dev, fn_dev = add_competition_features(X_dev, dev_pairs, s1_dev, s23_c, fn_dev)

    probs = model.predict(X_dev)

    results = decode_predictions(
        dev_pairs, probs,
        s1_dev['entity_id'].values, s23_c['entity_id'].values,
        len(s1_dev)
    )

    pred_dict = {k: set(v) for k, v in results.items()}
    f05 = score_submission(pred_dict, truth_dev)

    print(f"\n{'='*50}")
    print(f"DEV F0.5: {f05:.6f}")
    print(f"{'='*50}")
    print(f"Total time: {time.time()-t_start:.1f}s")

    return model, f05


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--n-s1', type=int, default=5000)
    p.add_argument('--country', default='US')
    p.add_argument('--top-k', type=int, default=15)
    args = p.parse_args()
    run_dev_test(n_s1=args.n_s1, country=args.country, top_k=args.top_k)
