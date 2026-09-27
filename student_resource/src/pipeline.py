"""
Main pipeline: preprocess → block → features → train → decode → output.
Run on dev subset first to validate, then full test.

Usage:
  python3 src/pipeline.py --mode dev      # validate on train dev split
  python3 src/pipeline.py --mode test     # generate test submission
"""
import argparse
import time
import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import normalize_name, normalize_address, extract_street_number, extract_pin_zip
from blocking import run_blocking
from features import compute_features_batch, add_competition_features
from scorer import score_submission, load_ground_truth, make_dev_split
from decoder import decode_predictions

BASE = os.path.dirname(os.path.dirname(__file__))
TRAIN_DIR = os.path.join(BASE, 'dataset', 'train')
TEST_DIR = os.path.join(BASE, 'dataset', 'test')
OUTPUT_DIR = os.path.join(BASE, 'output')


def load_and_preprocess(source_path, limit=None):
    """Load a source TSV file and add normalized columns."""
    print(f"Loading {os.path.basename(source_path)}...")
    df = pd.read_csv(source_path, sep='\t', dtype=str, nrows=limit)
    df = df.fillna('')
    
    print(f"  Preprocessing {len(df)} records...")
    t0 = time.time()
    df['name_norm'] = df['business_name'].apply(normalize_name)
    df['addr_norm'] = df['business_address'].apply(normalize_address)
    df['street_num'] = df['addr_norm'].apply(extract_street_number)
    df['pin_zip'] = df['addr_norm'].apply(extract_pin_zip)
    print(f"  Done in {time.time()-t0:.1f}s")
    return df


def build_training_pairs(candidates, s1_df, s23_df, truth_dict, max_neg_ratio=5):
    """
    Build (positive, negative) pairs from blocking candidates + ground truth.
    """
    s1_ids = s1_df['entity_id'].values
    s23_ids = s23_df['entity_id'].values
    
    # Build reverse lookup: s23_entity_id → s23_idx
    s23_id_to_idx = {eid: i for i, eid in enumerate(s23_ids)}
    
    pairs = []
    labels = []
    
    for s1_idx, s23_idx_set in candidates.items():
        s1_eid = s1_ids[s1_idx]
        true_matches = truth_dict.get(s1_eid, set())
        
        import random
        pos_list = []
        neg_list = []
        
        # Separate positives and negatives
        for s23_idx in s23_idx_set:
            s23_eid = s23_ids[s23_idx]
            if s23_eid in true_matches:
                pos_list.append(s23_idx)
            else:
                neg_list.append(s23_idx)
                
        # Subsample negatives to respect max_neg_ratio
        if max_neg_ratio is not None:
            max_n = max(len(pos_list) * max_neg_ratio, 5)
            if len(neg_list) > max_n:
                neg_list = random.sample(neg_list, max_n)
                
        for s23_idx in pos_list:
            pairs.append((s1_idx, s23_idx))
            labels.append(1)
        for s23_idx in neg_list:
            pairs.append((s1_idx, s23_idx))
            labels.append(0)
        
        # Also add ground truth positives that blocking may have missed
        for true_eid in true_matches:
            if true_eid in s23_id_to_idx:
                true_idx = s23_id_to_idx[true_eid]
                if true_idx not in s23_idx_set:
                    pairs.append((s1_idx, true_idx))
                    labels.append(1)
    
    return pairs, np.array(labels, dtype=np.int32)


def write_output(results, output_path):
    """Write matching_results.tsv."""
    with open(output_path, 'w') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for s1_id in sorted(results.keys()):
            matched = ','.join(results[s1_id])
            f.write(f'{s1_id}\t{matched}\n')
    print(f"Written {output_path} ({len(results)} rows)")


def write_candidates(candidates, s1_ids, s23_ids, output_path):
    """Write candidate_pairs.tsv."""
    with open(output_path, 'w') as f:
        f.write('source1_entity_id\tcandidate_entity_ids\n')
        all_s1 = set(range(len(s1_ids)))
        covered_s1 = set(candidates.keys())
        
        for s1_idx in sorted(covered_s1):
            s1_eid = s1_ids[s1_idx]
            cand_eids = [s23_ids[i] for i in candidates[s1_idx]]
            f.write(f'{s1_eid}\t{",".join(cand_eids)}\n')
        
        # S1s with no candidates
        for s1_idx in sorted(all_s1 - covered_s1):
            f.write(f'{s1_ids[s1_idx]}\t\n')
    print(f"Written {output_path}")


def run_pipeline(mode='dev', sample_limit=None, top_k=20):
    """Run the full pipeline."""
    t_start = time.time()
    
    # ── Load data ──
    s1_train = load_and_preprocess(os.path.join(TRAIN_DIR, 'train_source1.tsv'), limit=sample_limit)
    s2_train = load_and_preprocess(os.path.join(TRAIN_DIR, 'train_source2.tsv'), limit=sample_limit)
    s3_train = load_and_preprocess(os.path.join(TRAIN_DIR, 'train_source3.tsv'), limit=sample_limit)
    
    # Combine S2 + S3
    s23_train = pd.concat([s2_train, s3_train], ignore_index=True)
    print(f"\nCombined S23 train: {len(s23_train)} records")
    
    truth = load_ground_truth(os.path.join(TRAIN_DIR, 'train_ground_truth.tsv'))
    
    # ── Dev split ──
    if mode == 'dev':
        train_ids, dev_ids = make_dev_split(s1_train, truth)
        print(f"Dev split: {len(train_ids)} train, {len(dev_ids)} dev S1 entities")
        
        # For blocking + features, use only dev S1s (faster iteration)
        dev_mask = s1_train['entity_id'].isin(dev_ids)
        s1_dev = s1_train[dev_mask].reset_index(drop=True)
        s1_model = s1_train[~dev_mask].reset_index(drop=True)
        truth_model = {k: v for k, v in truth.items() if k in train_ids}
        truth_dev = {k: v for k, v in truth.items() if k in dev_ids}
    else:
        s1_model = s1_train
        truth_model = truth
    
    # ── Blocking on training data (for model training) ──
    print("\n=== BLOCKING (train) ===")
    candidates_train = run_blocking(s1_model, s23_train, top_k=top_k)
    
    # Measure blocking recall
    if mode == 'dev' or True:
        blocked_recall = 0
        total_true = 0
        s23_id_to_idx = {eid: i for i, eid in enumerate(s23_train['entity_id'].values)}
        s1_model_ids = s1_model['entity_id'].values
        
        for s1_idx, s23_set in candidates_train.items():
            s1_eid = s1_model_ids[s1_idx]
            true_matches = truth_model.get(s1_eid, set())
            for tm in true_matches:
                total_true += 1
                if tm in s23_id_to_idx and s23_id_to_idx[tm] in s23_set:
                    blocked_recall += 1
        
        print(f"Blocking recall: {blocked_recall}/{total_true} = {blocked_recall/max(total_true,1):.4f}")
    
    # ── Build training pairs ──
    print("\n=== BUILDING PAIRS ===")
    pairs_train, labels_train = build_training_pairs(candidates_train, s1_model, s23_train, truth_model)
    print(f"Training pairs: {len(pairs_train)} ({labels_train.sum()} positive, {len(labels_train)-labels_train.sum()} negative)")
    
    # ── Feature engineering ──
    print("\n=== FEATURES (train) ===")
    t0 = time.time()
    X_train, feature_names = compute_features_batch(pairs_train, s1_model, s23_train)
    X_train, feature_names = add_competition_features(X_train, pairs_train, s1_model, s23_train, feature_names)
    print(f"Features computed in {time.time()-t0:.1f}s, shape: {X_train.shape}")
    
    # ── Train LightGBM ──
    print("\n=== TRAINING MODELS (ENSEMBLE) ===")
    
    # Hard-negative mining: give higher weight to false positives with high name similarity
    # We will approximate this by giving extra weight to negatives
    base_weight = (len(labels_train) - labels_train.sum()) / max(labels_train.sum(), 1)
    sample_weights = np.ones(len(labels_train))
    sample_weights[labels_train == 1] = base_weight
    
    train_data_lgb = lgb.Dataset(X_train, label=labels_train, feature_name=feature_names, weight=sample_weights)
    
    params_lgb = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': -1,
        'min_child_samples': 100,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1,
    }
    
    print("Training LightGBM...")
    model_lgb = lgb.train(params_lgb, train_data_lgb, num_boost_round=300)
    
    # Feature importance
    imp = model_lgb.feature_importance(importance_type='gain')
    print("LightGBM Top 10 Features:")
    for name, score in sorted(zip(feature_names, imp), key=lambda x: -x[1])[:10]:
        print(f"  {name:25s} {score:.0f}")
        
    print("Training XGBoost...")
    model_xgb = xgb.XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05, 
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1
    )
    model_xgb.fit(X_train, labels_train, sample_weight=sample_weights)
    
    print("Training CatBoost...")
    model_cat = CatBoostClassifier(
        iterations=300, depth=8, learning_rate=0.05, 
        verbose=0, thread_count=-1
    )
    model_cat.fit(X_train, labels_train, sample_weight=sample_weights)
    
    # ── Evaluate on dev or generate test submission ──
    if mode == 'dev':
        print("\n=== BLOCKING (dev) ===")
        candidates_dev = run_blocking(s1_dev, s23_train, top_k=top_k)
        
        # Build pairs for dev
        all_dev_pairs = []
        for s1_idx, s23_set in candidates_dev.items():
            for s23_idx in s23_set:
                all_dev_pairs.append((s1_idx, s23_idx))
        
        if not all_dev_pairs:
            print("No dev pairs! Check blocking.")
            return
        
        print(f"\n=== FEATURES (dev) ===")
        X_dev, fn_dev = compute_features_batch(all_dev_pairs, s1_dev, s23_train)
        X_dev, fn_dev = add_competition_features(X_dev, all_dev_pairs, s1_dev, s23_train, fn_dev)
        
        probs_lgb = model_lgb.predict(X_dev)
        probs_xgb = model_xgb.predict_proba(X_dev)[:, 1]
        probs_cat = model_cat.predict_proba(X_dev)[:, 1]
        probs_dev = 0.4 * probs_lgb + 0.4 * probs_xgb + 0.2 * probs_cat
        
        # Decode
        results = decode_predictions(
            all_dev_pairs, probs_dev,
            s1_dev['entity_id'].values, s23_train['entity_id'].values,
            len(s1_dev)
        )
        
        # Convert results to sets for scoring
        pred_dict = {k: set(v) for k, v in results.items()}
        f05 = score_submission(pred_dict, truth_dev)
        print(f"\n{'='*50}")
        print(f"DEV F0.5 SCORE: {f05:.6f}")
        print(f"{'='*50}")
        
    else:  # test mode
        # Load test data
        s1_test = load_and_preprocess(os.path.join(TEST_DIR, 'test_source1.tsv'))
        s2_test = load_and_preprocess(os.path.join(TEST_DIR, 'test_source2.tsv'))
        s3_test = load_and_preprocess(os.path.join(TEST_DIR, 'test_source3.tsv'))
        s23_test = pd.concat([s2_test, s3_test], ignore_index=True)
        
        print("\n=== BLOCKING (test) ===")
        candidates_test = run_blocking(s1_test, s23_test, top_k=top_k)
        
        print(f"\n=== FEATURES & PREDICTION (test) ===")
        all_s1_indices = list(candidates_test.keys())
        chunk_size = 50000  # Process 50k S1 entities at a time
        
        all_results = {}
        for i in range(0, len(all_s1_indices), chunk_size):
            chunk_s1 = all_s1_indices[i:i+chunk_size]
            
            chunk_pairs = []
            for s1_idx in chunk_s1:
                for s23_idx in candidates_test[s1_idx]:
                    chunk_pairs.append((s1_idx, s23_idx))
                    
            if not chunk_pairs:
                continue
                
            X_chunk, fn_chunk = compute_features_batch(chunk_pairs, s1_test, s23_test)
            X_chunk, fn_chunk = add_competition_features(X_chunk, chunk_pairs, s1_test, s23_test, fn_chunk)
            
            probs_lgb = model_lgb.predict(X_chunk)
            probs_xgb = model_xgb.predict_proba(X_chunk)[:, 1]
            probs_cat = model_cat.predict_proba(X_chunk)[:, 1]
            probs_chunk = 0.4 * probs_lgb + 0.4 * probs_xgb + 0.2 * probs_cat
            
            # Decode chunk
            chunk_results = decode_predictions(
                chunk_pairs, probs_chunk,
                s1_test['entity_id'].values, s23_test['entity_id'].values,
                len(s1_test)
            )
            all_results.update(chunk_results)
            print(f"  Processed {min(i+chunk_size, len(all_s1_indices))}/{len(all_s1_indices)} S1 entities")
        
        results = all_results
        
        # Ensure all S1 entities from the test set are present in the final output
        for s1_eid in s1_test['entity_id'].values:
            if s1_eid not in results:
                results[s1_eid] = []
        
        # Write outputs
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        write_output(results, os.path.join(OUTPUT_DIR, 'matching_results.tsv'))
        write_candidates(candidates_test, s1_test['entity_id'].values,
                        s23_test['entity_id'].values,
                        os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv'))
    
    print(f"\nTotal time: {time.time()-t_start:.1f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['dev', 'test'], default='dev')
    parser.add_argument('--sample', type=int, default=None, help='Limit records per source for testing')
    parser.add_argument('--top-k', type=int, default=20, help='Top-K for TF-IDF blocking')
    args = parser.parse_args()
    
    run_pipeline(mode=args.mode, sample_limit=args.sample, top_k=args.top_k)
