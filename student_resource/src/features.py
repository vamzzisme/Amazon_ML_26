"""Pairwise feature engineering for entity resolution."""
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler
import jellyfish


def jaccard_tokens(a, b):
    """Token-level Jaccard similarity."""
    if not a or not b:
        return 0.0
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def containment(a, b):
    """What fraction of a's tokens appear in b (or vice versa, take max)."""
    if not a or not b:
        return 0.0
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return max(len(sa & sb) / len(sa), len(sa & sb) / len(sb))


def char_ngram_jaccard(a, b, n=3):
    """Character n-gram Jaccard."""
    if not a or not b or len(a) < n or len(b) < n:
        return 0.0
    sa = {a[i:i+n] for i in range(len(a)-n+1)}
    sb = {b[i:i+n] for i in range(len(b)-n+1)}
    return len(sa & sb) / len(sa | sb)


def length_ratio(a, b):
    """Ratio of shorter to longer string length."""
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    return min(la, lb) / max(la, lb)


def compute_pair_features(s1_name, s1_addr, s23_name, s23_addr,
                          s1_street_num='', s23_street_num='',
                          s1_pin='', s23_pin=''):
    """Compute all features for a single (S1, S23) pair. Returns dict."""
    feats = {}
    
    # Name features
    feats['name_jaccard'] = jaccard_tokens(s1_name, s23_name)
    feats['name_levenshtein'] = Levenshtein.normalized_similarity(s1_name or '', s23_name or '')
    feats['name_jaro_winkler'] = JaroWinkler.similarity(s1_name or '', s23_name or '')
    feats['name_fuzz_ratio'] = fuzz.ratio(s1_name or '', s23_name or '') / 100.0
    feats['name_fuzz_partial'] = fuzz.partial_ratio(s1_name or '', s23_name or '') / 100.0
    feats['name_fuzz_token_sort'] = fuzz.token_sort_ratio(s1_name or '', s23_name or '') / 100.0
    feats['name_containment'] = containment(s1_name, s23_name)
    feats['name_char3gram'] = char_ngram_jaccard(s1_name, s23_name, 3)
    feats['name_length_ratio'] = length_ratio(s1_name, s23_name)
    
    # Address features
    feats['addr_jaccard'] = jaccard_tokens(s1_addr, s23_addr)
    feats['addr_levenshtein'] = Levenshtein.normalized_similarity(s1_addr or '', s23_addr or '')
    feats['addr_containment'] = containment(s1_addr, s23_addr)
    feats['addr_char3gram'] = char_ngram_jaccard(s1_addr, s23_addr, 3)
    feats['addr_length_ratio'] = length_ratio(s1_addr, s23_addr)
    
    # Exact match features
    feats['street_num_match'] = float(s1_street_num != '' and s1_street_num == s23_street_num)
    feats['pin_match'] = float(s1_pin != '' and s1_pin == s23_pin)
    
    # Combined features
    feats['name_addr_avg'] = (feats['name_jaccard'] + feats['addr_jaccard']) / 2
    feats['addr_empty'] = float(not s1_addr or not s23_addr)
    
    # Cross-field & phonetic features
    feats['cross_name1_addr2'] = jaccard_tokens(s1_name, s23_addr)
    feats['cross_name2_addr1'] = jaccard_tokens(s23_name, s1_addr)
    feats['cross_consistency'] = 1.0 - abs(feats['name_jaccard'] - feats['addr_jaccard'])
    
    s1_first_word = s1_name.split()[0] if s1_name and s1_name.split() else ""
    s23_first_word = s23_name.split()[0] if s23_name and s23_name.split() else ""
    if s1_first_word and s23_first_word:
        feats['name_metaphone_match'] = float(jellyfish.metaphone(s1_first_word) == jellyfish.metaphone(s23_first_word))
    else:
        feats['name_metaphone_match'] = 0.0
    
    return feats


def compute_features_batch(pairs, s1_df, s23_df):
    """
    Compute features for a batch of pairs.
    pairs: list of (s1_idx, s23_idx)
    Returns: numpy array of shape (len(pairs), n_features), feature_names
    """
    s1_names = s1_df['name_norm'].values
    s1_addrs = s1_df['addr_norm'].values
    s1_snums = s1_df['street_num'].values if 'street_num' in s1_df.columns else [''] * len(s1_df)
    s1_pins = s1_df['pin_zip'].values if 'pin_zip' in s1_df.columns else [''] * len(s1_df)
    
    s23_names = s23_df['name_norm'].values
    s23_addrs = s23_df['addr_norm'].values
    s23_snums = s23_df['street_num'].values if 'street_num' in s23_df.columns else [''] * len(s23_df)
    s23_pins = s23_df['pin_zip'].values if 'pin_zip' in s23_df.columns else [''] * len(s23_df)
    
    if not pairs:
        return np.array([], dtype=np.float32), []
        
    # Compute first pair to get feature names and count
    s1_idx, s23_idx = pairs[0]
    first_feats = compute_pair_features(
        s1_names[s1_idx], s1_addrs[s1_idx],
        s23_names[s23_idx], s23_addrs[s23_idx],
        str(s1_snums[s1_idx]), str(s23_snums[s23_idx]),
        str(s1_pins[s1_idx]), str(s23_pins[s23_idx])
    )
    feature_names = list(first_feats.keys())
    num_features = len(feature_names)
    
    # Pre-allocate array
    results = np.zeros((len(pairs), num_features), dtype=np.float32)
    results[0] = [first_feats[k] for k in feature_names]
    
    for i in range(1, len(pairs)):
        s1_idx, s23_idx = pairs[i]
        feats = compute_pair_features(
            s1_names[s1_idx], s1_addrs[s1_idx],
            s23_names[s23_idx], s23_addrs[s23_idx],
            str(s1_snums[s1_idx]), str(s23_snums[s23_idx]),
            str(s1_pins[s1_idx]), str(s23_pins[s23_idx])
        )
        for j, k in enumerate(feature_names):
            results[i, j] = feats[k]
    
    return results, feature_names


def add_competition_features(features, pairs, s1_df, s23_df, feature_names):
    """
    Add competition features: for each S2/S3 record, how does this S1 rank
    among all S1s competing for it? And what's the gap to 2nd best?
    
    Modifies features in-place and returns updated feature_names.
    """
    # Build reverse index: s23_idx → [(pair_position, name_jaccard_score)]
    name_jaccard_col = feature_names.index('name_jaccard')
    s23_to_pairs = {}
    for pair_pos, (s1_idx, s23_idx) in enumerate(pairs):
        score = features[pair_pos, name_jaccard_col]
        s23_to_pairs.setdefault(s23_idx, []).append((pair_pos, score))
    
    rank_col = np.zeros(len(pairs), dtype=np.float32)
    gap_col = np.zeros(len(pairs), dtype=np.float32)
    n_competitors_col = np.zeros(len(pairs), dtype=np.float32)
    
    for s23_idx, pair_list in s23_to_pairs.items():
        if len(pair_list) <= 1:
            for pair_pos, _ in pair_list:
                rank_col[pair_pos] = 1.0
                gap_col[pair_pos] = 1.0
                n_competitors_col[pair_pos] = 1.0
            continue
        
        # Sort by score descending
        sorted_pairs = sorted(pair_list, key=lambda x: -x[1])
        best_score = sorted_pairs[0][1]
        second_score = sorted_pairs[1][1] if len(sorted_pairs) > 1 else 0.0
        
        for rank, (pair_pos, score) in enumerate(sorted_pairs):
            rank_col[pair_pos] = 1.0 / (rank + 1)  # 1.0 for best, 0.5 for 2nd, etc.
            gap_col[pair_pos] = score - second_score if rank == 0 else score - best_score
            n_competitors_col[pair_pos] = len(pair_list)
    
    # Append to features
    new_cols = np.column_stack([rank_col, gap_col, n_competitors_col])
    features = np.hstack([features, new_cols])
    feature_names = feature_names + ['comp_rank', 'comp_gap', 'comp_n_competitors']
    
    return features, feature_names
