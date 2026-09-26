"""F0.5 scorer + dev split utilities."""
import pandas as pd
import numpy as np


def f05_per_entity(pred_set, true_set):
    """F0.5 for a single S1 entity."""
    if len(true_set) == 0 and len(pred_set) == 0:
        return 1.0
    if len(pred_set) == 0:
        return 0.0
    if len(true_set) == 0:
        return 0.0
    tp = len(pred_set & true_set)
    prec = tp / len(pred_set)
    rec = tp / len(true_set)
    if prec + rec == 0:
        return 0.0
    return (1.25 * prec * rec) / (0.25 * prec + rec)


def score_submission(pred_dict, truth_dict):
    """
    Macro-averaged F0.5.
    pred_dict: {s1_id: set of matched ids}
    truth_dict: {s1_id: set of matched ids}
    """
    scores = []
    for s1_id in truth_dict:
        pred = pred_dict.get(s1_id, set())
        true = truth_dict[s1_id]
        scores.append(f05_per_entity(pred, true))
    return np.mean(scores)


def load_ground_truth(path):
    """Load ground truth into {s1_id: set(matched_ids)}."""
    df = pd.read_csv(path, sep='\t', dtype=str)
    truth = {}
    for _, row in df.iterrows():
        s1_id = row['source1_entity_id']
        matched = row.get('matched_entity_ids', '')
        if pd.isna(matched) or matched.strip() == '':
            truth[s1_id] = set()
        else:
            truth[s1_id] = set(matched.split(','))
    return truth


def make_dev_split(s1_df, truth_dict, holdout_states=None, frac=0.15, seed=42):
    """
    Split S1 by holding out entire US states for realistic dev set.
    Returns (train_s1_ids, dev_s1_ids).
    """
    if holdout_states is None:
        holdout_states = {'california', 'texas', 'ohio'}

    us_mask = s1_df['country'] == 'US'
    # Check if address contains any holdout state (case-insensitive)
    def in_holdout(addr):
        if not isinstance(addr, str):
            return False
        addr_lower = addr.lower()
        for st in holdout_states:
            if st in addr_lower:
                return True
        # Also check abbreviations
        abbrev_map = {'california': 'CA', 'texas': 'TX', 'ohio': 'OH',
                      'florida': 'FL', 'illinois': 'IL', 'new york': 'NY'}
        for st in holdout_states:
            abbr = abbrev_map.get(st, '')
            if abbr and f', {abbr}' in addr:
                return True
        return False

    dev_mask = us_mask & s1_df['business_address'].apply(in_holdout)

    # Also hold out some India records randomly
    india_mask = s1_df['country'] == 'India'
    india_ids = s1_df[india_mask]['entity_id'].values
    rng = np.random.RandomState(seed)
    india_dev_ids = set(rng.choice(india_ids, size=int(len(india_ids) * frac), replace=False))

    dev_ids = set(s1_df[dev_mask]['entity_id'].values) | india_dev_ids
    train_ids = set(s1_df['entity_id'].values) - dev_ids

    return train_ids, dev_ids
