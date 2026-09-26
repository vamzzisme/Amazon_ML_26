"""Blocking: fast candidate generation using inverted indices."""
import numpy as np
import pandas as pd
from collections import defaultdict
import time
import jellyfish


def _build_inverted_index(texts, min_token_len=2, max_posting=8000):
    """Build inverted index: token → list of doc indices. Cap posting lists."""
    inv = defaultdict(list)
    for idx, text in enumerate(texts):
        if not text:
            continue
        for t in set(text.split()):
            if len(t) >= min_token_len:
                inv[t].append(idx)
    # Prune: remove tokens with too many or too few postings
    inv = {t: idxs for t, idxs in inv.items() if 2 <= len(idxs) <= max_posting}
    return inv


def token_blocking(s1_names, s23_names, top_k=20):
    """Word-token inverted index blocking with capped posting lists."""
    t0 = time.time()
    
    # Cap at 5000 postings per token — tokens with more are not discriminative
    inv = _build_inverted_index(s23_names, min_token_len=2, max_posting=8000)
    print(f"  Token inv index: {len(inv)} tokens, built in {time.time()-t0:.1f}s")
    
    candidates = {}
    for s1_idx, name in enumerate(s1_names):
        if not name:
            candidates[s1_idx] = []
            continue
        
        tokens = set(name.split())
        # Collect all candidate s23 indices from posting lists
        all_hits = []
        for t in tokens:
            if len(t) >= 2 and t in inv:
                all_hits.extend(inv[t])
        
        if not all_hits:
            candidates[s1_idx] = []
            continue
        
        # Count occurrences using numpy (much faster than defaultdict)
        hit_arr = np.array(all_hits, dtype=np.int32)
        unique_ids, counts = np.unique(hit_arr, return_counts=True)
        
        # Score = shared_tokens / max(s1_tokens, s23_tokens)
        n_s1 = len(tokens)
        # Get top-K by count first (fast), then refine scores
        if len(counts) > top_k:
            top_pos = np.argpartition(counts, -top_k)[-top_k:]
        else:
            top_pos = np.arange(len(counts))
        
        candidates[s1_idx] = list(zip(unique_ids[top_pos].tolist(), counts[top_pos].tolist()))
        
        if s1_idx % 50000 == 0 and s1_idx > 0:
            print(f"    {s1_idx}/{len(s1_names)}, elapsed {time.time()-t0:.1f}s")
    
    print(f"  Token blocking done in {time.time()-t0:.1f}s")
    return candidates


def combined_name_addr_blocking(s1_names, s1_addrs, s23_names, s23_addrs, top_k=20):
    """
    Combined blocking: tokens from BOTH name and address go into a single 
    inverted index. This catches cases where name is different but address matches.
    """
    t0 = time.time()
    
    # Build combined index
    inv = defaultdict(list)
    for idx in range(len(s23_names)):
        combined = (s23_names[idx] or '') + ' ' + (s23_addrs[idx] or '')
        for t in set(combined.split()):
            if len(t) >= 2:
                inv[t].append(idx)
    
    inv = {t: idxs for t, idxs in inv.items() if 2 <= len(idxs) <= 5000}
    print(f"  Combined inv index: {len(inv)} tokens, built in {time.time()-t0:.1f}s")
    
    candidates = {}
    for s1_idx in range(len(s1_names)):
        combined = (s1_names[s1_idx] or '') + ' ' + (s1_addrs[s1_idx] or '')
        tokens = set(combined.split())
        
        all_hits = []
        for t in tokens:
            if len(t) >= 2 and t in inv:
                all_hits.extend(inv[t])
        
        if not all_hits:
            candidates[s1_idx] = []
            continue
        
        hit_arr = np.array(all_hits, dtype=np.int32)
        unique_ids, counts = np.unique(hit_arr, return_counts=True)
        
        if len(counts) > top_k:
            top_pos = np.argpartition(counts, -top_k)[-top_k:]
        else:
            top_pos = np.arange(len(counts))
        
        candidates[s1_idx] = list(zip(unique_ids[top_pos].tolist(), counts[top_pos].tolist()))
        
        if s1_idx % 50000 == 0 and s1_idx > 0:
            print(f"    {s1_idx}/{len(s1_names)}, elapsed {time.time()-t0:.1f}s")
    
    print(f"  Combined blocking done in {time.time()-t0:.1f}s")
    return candidates


def address_key_block(s1_addrs, s23_addrs):
    """Address-key blocking: street_number + first significant word."""
    import re
    t0 = time.time()
    
    def extract_key(addr):
        if not addr:
            return None
        num_m = re.search(r'\b(\d+)\b', addr)
        num = num_m.group(1) if num_m else ''
        for w in addr.split():
            if not w.isdigit() and len(w) > 2:
                return f"{num}_{w}" if num else None
        return None
    
    s23_index = defaultdict(list)
    for idx, addr in enumerate(s23_addrs):
        key = extract_key(addr)
        if key:
            s23_index[key].append(idx)
    
    # Cap posting lists
    s23_index = {k: v for k, v in s23_index.items() if len(v) <= 500}
    
    candidates = {}
    for idx, addr in enumerate(s1_addrs):
        key = extract_key(addr)
        if key and key in s23_index:
            candidates[idx] = [(s23_idx, 1) for s23_idx in s23_index[key]]
    
    print(f"  Address key blocking done in {time.time()-t0:.1f}s, keys: {len(s23_index)}")
    return candidates


def prefix_blocking(s1_names, s23_names, prefix_len=4, top_k=20):
    """Block by first N chars of the first word — catches transliteration survivors."""
    t0 = time.time()
    inv = defaultdict(list)
    for idx, name in enumerate(s23_names):
        if not name:
            continue
        first_word = name.split()[0] if name.split() else ''
        if len(first_word) >= prefix_len:
            inv[first_word[:prefix_len]].append(idx)
    
    inv = {k: v for k, v in inv.items() if 2 <= len(v) <= 3000}
    
    candidates = {}
    for s1_idx, name in enumerate(s1_names):
        if not name:
            continue
        first_word = name.split()[0] if name.split() else ''
        if len(first_word) >= prefix_len:
            key = first_word[:prefix_len]
            if key in inv:
                candidates[s1_idx] = [(idx, 1) for idx in inv[key][:top_k]]
    
    print(f"  Prefix blocking done in {time.time()-t0:.1f}s, keys: {len(inv)}")
    return candidates


def phonetic_blocking(s1_names, s23_names, top_k=20):
    """Block by Metaphone code of the first word (catches severe typos and sounds-like names)."""
    t0 = time.time()
    inv = defaultdict(list)
    for idx, name in enumerate(s23_names):
        if not name:
            continue
        first_word = name.split()[0] if name.split() else ''
        if len(first_word) >= 3:
            code = jellyfish.metaphone(first_word)
            if code:
                inv[code].append(idx)
    
    inv = {k: v for k, v in inv.items() if 2 <= len(v) <= 3000}
    
    candidates = {}
    for s1_idx, name in enumerate(s1_names):
        if not name:
            continue
        first_word = name.split()[0] if name.split() else ''
        if len(first_word) >= 3:
            code = jellyfish.metaphone(first_word)
            if code and code in inv:
                candidates[s1_idx] = [(idx, 1) for idx in inv[code][:top_k]]
    
    print(f"  Phonetic blocking done in {time.time()-t0:.1f}s, keys: {len(inv)}")
    return candidates


def merge_candidates(*candidate_dicts):
    """Union of multiple blocking strategies."""
    merged = {}
    for cands in candidate_dicts:
        for s1_idx, s23_list in cands.items():
            if s1_idx not in merged:
                merged[s1_idx] = set()
            for item in s23_list:
                if isinstance(item, tuple):
                    merged[s1_idx].add(item[0])
                else:
                    merged[s1_idx].add(item)
    return merged


def run_blocking(s1_df, s23_df, top_k=20):
    """Full blocking: country partition → multi-pass → union."""
    all_candidates = {}
    
    countries = s1_df['country'].unique()
    for country in countries:
        print(f"\nBlocking for country: {country}")
        s1_mask = s1_df['country'].values == country
        s23_mask = s23_df['country'].values == country
        
        s1_local_to_global = np.where(s1_mask)[0]
        s23_local_to_global = np.where(s23_mask)[0]
        
        s1_names = s1_df.loc[s1_mask, 'name_norm'].values
        s23_names = s23_df.loc[s23_mask, 'name_norm'].values
        s1_addrs = s1_df.loc[s1_mask, 'addr_norm'].values
        s23_addrs = s23_df.loc[s23_mask, 'addr_norm'].values
        
        print(f"  {len(s1_names)} S1 x {len(s23_names)} S23")
        
        # Pass A: name token blocking
        cands_name = token_blocking(s1_names, s23_names, top_k=top_k)
        
        # Pass B: combined name+address blocking  
        cands_combined = combined_name_addr_blocking(s1_names, s1_addrs, s23_names, s23_addrs, top_k=top_k)
        
        # Pass C: address key blocking
        cands_addr = address_key_block(s1_addrs, s23_addrs)
        
        # Pass D: prefix blocking
        cands_prefix = prefix_blocking(s1_names, s23_names, prefix_len=4, top_k=top_k)
        
        # Pass E: phonetic blocking
        cands_phonetic = phonetic_blocking(s1_names, s23_names, top_k=top_k)
        
        # Merge and map to global indices
        local_merged = merge_candidates(cands_name, cands_combined, cands_addr, cands_prefix, cands_phonetic)
        
        for local_s1, local_s23_set in local_merged.items():
            global_s1 = s1_local_to_global[local_s1]
            global_s23_set = {s23_local_to_global[i] for i in local_s23_set}
            all_candidates[global_s1] = all_candidates.get(global_s1, set()) | global_s23_set
    
    print(f"\nTotal S1 with candidates: {len(all_candidates)}/{len(s1_df)}")
    total_pairs = sum(len(v) for v in all_candidates.values())
    print(f"Total candidate pairs: {total_pairs:,}")
    if all_candidates:
        print(f"Avg candidates per S1: {total_pairs/len(all_candidates):.1f}")
    
    return all_candidates
