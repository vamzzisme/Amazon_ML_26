"""Smart decoding: competition decoding + F0.5 optimal subset selection."""
import numpy as np
from collections import defaultdict


def competition_decode(pairs, probs, min_gap=0.05):
    """
    Competition decoding: each S2/S3 goes to at most one S1.
    For each S2/S3, assign it to the S1 with highest probability,
    but only if the gap to 2nd best exceeds min_gap.
    
    pairs: list of (s1_idx, s23_idx)
    probs: array of match probabilities
    Returns: filtered list of (s1_idx, s23_idx, prob)
    """
    # Group by S2/S3
    s23_candidates = defaultdict(list)
    for i, (s1_idx, s23_idx) in enumerate(pairs):
        s23_candidates[s23_idx].append((s1_idx, probs[i]))
    
    # For each S2/S3, pick the best S1 if clear winner
    surviving = []
    for s23_idx, cands in s23_candidates.items():
        if len(cands) == 1:
            s1_idx, prob = cands[0]
            surviving.append((s1_idx, s23_idx, prob))
        else:
            # Sort descending
            cands.sort(key=lambda x: -x[1])
            best_s1, best_prob = cands[0]
            second_prob = cands[1][1]
            # Only assign if clear winner
            if best_prob - second_prob >= min_gap:
                surviving.append((best_s1, s23_idx, best_prob))
            # else: this S2/S3 is ambiguous, don't assign to anyone
    
    return surviving


def f05_optimal_subset(candidates_with_probs, prior_n_matches=3.46):
    """
    For a single S1, pick the subset of candidates maximizing expected F0.5.
    
    candidates_with_probs: list of (s23_idx, prob), sorted by prob descending
    prior_n_matches: expected number of true matches (used for estimating recall)
    
    Returns: list of s23_idx to include in the match set
    """
    if not candidates_with_probs:
        return []
    
    # Sort by probability descending
    sorted_cands = sorted(candidates_with_probs, key=lambda x: -x[1])
    
    best_f05 = 0.0  # F0.5 of empty set (if no true matches, empty set scores 1.0,
                      # but we don't know that, so we use 0.0 as baseline)
    best_k = 0
    
    # Try including top-1, top-2, ..., top-N candidates
    for k in range(1, len(sorted_cands) + 1):
        probs_k = [p for _, p in sorted_cands[:k]]
        
        # Expected precision = mean of probabilities of included candidates
        expected_prec = np.mean(probs_k)
        # Expected recall = sum of probabilities / estimated true matches
        expected_tp = sum(probs_k)
        expected_rec = min(expected_tp / max(prior_n_matches, 1.0), 1.0)
        
        # F0.5
        if expected_prec + expected_rec > 0:
            f05 = (1.25 * expected_prec * expected_rec) / (0.25 * expected_prec + expected_rec)
        else:
            f05 = 0.0
        
        if f05 > best_f05:
            best_f05 = f05
            best_k = k
        elif f05 < best_f05 * 0.95:
            # Early stop: F0.5 is dropping significantly
            break
    
    return [idx for idx, _ in sorted_cands[:best_k]]


def decode_predictions(pairs, probs, s1_ids, s23_ids, n_s1_total,
                       min_gap=0.05, prob_threshold=0.3):
    """
    Full decoding pipeline:
    1. Filter low-probability pairs
    2. Competition decoding
    3. F0.5 optimal subset selection per S1
    
    Returns: {s1_entity_id: [list of matched s23_entity_ids]}
    """
    # Step 1: filter very low probability pairs
    mask = probs >= prob_threshold
    filtered_pairs = [p for p, m in zip(pairs, mask) if m]
    filtered_probs = probs[mask]
    
    # Step 2: competition decoding
    surviving = competition_decode(filtered_pairs, filtered_probs, min_gap=min_gap)
    
    # Step 3: group by S1 and do F0.5 optimal subset selection
    s1_groups = defaultdict(list)
    for s1_idx, s23_idx, prob in surviving:
        s1_groups[s1_idx].append((s23_idx, prob))
    
    results = {}
    for s1_idx, cands in s1_groups.items():
        selected_s23_indices = f05_optimal_subset(cands)
        s1_eid = s1_ids[s1_idx]
        matched_eids = [s23_ids[i] for i in selected_s23_indices]
        results[s1_eid] = matched_eids
    
    # Ensure all S1 entities are present (singletons get empty list)
    # NOTE: Moved this logic to pipeline.py because doing it here 
    # inside a chunk overwrites other chunks with empty lists!
    
    return results
