"""Learn transliteration mappings from training ground truth."""
import os
import json
import time
from collections import defaultdict
import pandas as pd
import sys

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import normalize_name

BASE = os.path.dirname(os.path.dirname(__file__))
TRAIN_DIR = os.path.join(BASE, 'dataset', 'train')

def load_data():
    s1 = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source1.tsv'), sep='\t', dtype=str).fillna('')
    s2 = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source2.tsv'), sep='\t', dtype=str).fillna('')
    s3 = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source3.tsv'), sep='\t', dtype=str).fillna('')
    s23 = pd.concat([s2, s3], ignore_index=True)
    
    gt = pd.read_csv(os.path.join(TRAIN_DIR, 'train_ground_truth.tsv'), sep='\t', dtype=str).fillna('')
    return s1, s23, gt

def learn_dictionary():
    t0 = time.time()
    print("Loading data...")
    s1_df, s23_df, gt_df = load_data()
    
    # Filter India only to learn Indic transliterations
    s1_df = s1_df[s1_df['country'] == 'India']
    s23_df = s23_df[s23_df['country'] == 'India']
    
    s1_names = dict(zip(s1_df['entity_id'], s1_df['business_name']))
    s23_names = dict(zip(s23_df['entity_id'], s23_df['business_name']))
    
    gt_map = {}
    for _, row in gt_df.iterrows():
        if row['matched_entity_ids']:
            gt_map[row['source1_entity_id']] = row['matched_entity_ids'].split(',')
            
    print(f"Data loaded in {time.time()-t0:.1f}s. Learning mappings...")
    
    pair_counts = defaultdict(int)
    
    for s1_id, matched_ids in gt_map.items():
        if s1_id not in s1_names:
            continue
        
        name1 = normalize_name(s1_names[s1_id])
        if not name1:
            continue
        tokens1 = set(name1.split())
        
        for s23_id in matched_ids:
            if s23_id not in s23_names:
                continue
            
            name2 = normalize_name(s23_names[s23_id])
            if not name2:
                continue
            tokens2 = set(name2.split())
            
            # Find mismatches
            diff1 = tokens1 - tokens2
            diff2 = tokens2 - tokens1
            
            # If exactly 1 mismatch on each side, they might be transliteration equivalents
            if len(diff1) == 1 and len(diff2) == 1:
                t1 = list(diff1)[0]
                t2 = list(diff2)[0]
                
                # Exclude purely numeric mismatches or identical tokens
                if t1 != t2 and not t1.isdigit() and not t2.isdigit():
                    pair_counts[(t2, t1)] += 1 # map t2 (from S2/S3) to t1 (from S1)
    
    # Filter: keep mapping if count >= 3
    final_dict = {}
    for (t2, t1), count in pair_counts.items():
        if count >= 3:
            # Resolve conflicts: if t2 maps to multiple t1s, take the most frequent
            if t2 in final_dict:
                if count > final_dict[t2]['count']:
                    final_dict[t2] = {'target': t1, 'count': count}
            else:
                final_dict[t2] = {'target': t1, 'count': count}
                
    out_dict = {k: v['target'] for k, v in final_dict.items()}
    
    out_path = os.path.join(os.path.dirname(__file__), 'indic_dict.json')
    with open(out_path, 'w') as f:
        json.dump(out_dict, f, indent=2)
        
    print(f"Learned {len(out_dict)} transliteration mappings in {time.time()-t0:.1f}s")
    print(f"Saved to {out_path}")
    
    # Print top 10 examples
    sorted_pairs = sorted([(k, v['target'], v['count']) for k, v in final_dict.items()], key=lambda x: -x[2])
    print("\nTop 10 learned mappings:")
    for t2, t1, count in sorted_pairs[:10]:
        print(f"  {t2:15s} -> {t1:15s} (seen {count} times)")

if __name__ == '__main__':
    learn_dictionary()
