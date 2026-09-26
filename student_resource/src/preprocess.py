"""Text preprocessing for entity resolution."""
import re
import os
import json
import unicodedata
from unidecode import unidecode

_INDIC_DICT = None
def load_indic_dict():
    global _INDIC_DICT
    if _INDIC_DICT is None:
        dict_path = os.path.join(os.path.dirname(__file__), 'indic_dict.json')
        if os.path.exists(dict_path):
            with open(dict_path, 'r') as f:
                _INDIC_DICT = json.load(f)
        else:
            _INDIC_DICT = {}

# Legal suffix normalization (order matters: longest first)
LEGAL_SUFFIXES = {
    'pvt': 'private', 'ltd': 'limited', 'llp': 'llp',
    'corp': 'corporation', 'inc': 'incorporated',
    'llc': 'llc', 'co': 'company', 'assn': 'association',
    'intl': 'international', 'natl': 'national',
    'grp': 'group', 'svcs': 'services', 'svc': 'service',
    'mfg': 'manufacturing', 'engr': 'engineering',
    'mgmt': 'management', 'dept': 'department',
    # French
    'sarl': 'sarl', 'sas': 'sas', 'sci': 'sci', 'sa': 'sa',
    'ste': 'societe', 'ets': 'etablissements',
}

ADDRESS_ABBREVS = {
    'st': 'street', 'rd': 'road', 'ave': 'avenue', 'blvd': 'boulevard',
    'dr': 'drive', 'ln': 'lane', 'ct': 'court', 'pl': 'place',
    'cir': 'circle', 'pkwy': 'parkway', 'hwy': 'highway',
    'apt': 'apartment', 'ste': 'suite', 'fl': 'floor',
    # French address
    'r': 'rue', 'bd': 'boulevard', 'av': 'avenue', 'imp': 'impasse',
}

# US state abbreviations
US_STATES = {
    'al': 'alabama', 'ak': 'alaska', 'az': 'arizona', 'ar': 'arkansas',
    'ca': 'california', 'co': 'colorado', 'ct': 'connecticut', 'de': 'delaware',
    'fl': 'florida', 'ga': 'georgia', 'hi': 'hawaii', 'id': 'idaho',
    'il': 'illinois', 'in': 'indiana', 'ia': 'iowa', 'ks': 'kansas',
    'ky': 'kentucky', 'la': 'louisiana', 'me': 'maine', 'md': 'maryland',
    'ma': 'massachusetts', 'mi': 'michigan', 'mn': 'minnesota', 'ms': 'mississippi',
    'mo': 'missouri', 'mt': 'montana', 'ne': 'nebraska', 'nv': 'nevada',
    'nh': 'new hampshire', 'nj': 'new jersey', 'nm': 'new mexico', 'ny': 'new york',
    'nc': 'north carolina', 'nd': 'north dakota', 'oh': 'ohio', 'ok': 'oklahoma',
    'or': 'oregon', 'pa': 'pennsylvania', 'ri': 'rhode island', 'sc': 'south carolina',
    'sd': 'south dakota', 'tn': 'tennessee', 'tx': 'texas', 'ut': 'utah',
    'vt': 'vermont', 'va': 'virginia', 'wa': 'washington', 'wv': 'west virginia',
    'wi': 'wisconsin', 'wy': 'wyoming', 'dc': 'district of columbia',
}

_JUNK_PREFIX = re.compile(r'^[\-\*<>\#\!\@\~\+]+\s*')
_DOMAIN_RE = re.compile(r'\.(com|net|org|co\.in|in|io|biz|us|fr)$', re.I)
_PUNCT = re.compile(r'[^\w\s]', re.UNICODE)
_MULTI_SPACE = re.compile(r'\s+')


def normalize_text(text):
    """Core normalization: transliterate → lowercase → strip junk → clean."""
    if not text or text == '<NULL>':
        return ''
    # Transliterate non-Latin scripts to ASCII
    text = unidecode(text)
    # Lowercase
    text = text.lower().strip()
    # Strip junk prefixes
    text = _JUNK_PREFIX.sub('', text)
    # Replace & with 'and'
    text = text.replace('&', ' and ')
    # Remove punctuation (keep alphanumeric + spaces)
    text = _PUNCT.sub(' ', text)
    # Collapse whitespace
    text = _MULTI_SPACE.sub(' ', text).strip()
    return text


def normalize_name(name):
    """Normalize a business name."""
    text = normalize_text(name)
    if not text:
        return ''
    # Handle domain-style names: "colonialfoods.com" → "colonialfoods"
    text = re.sub(r'\.(com|net|org|co in|in|io|biz|us|fr)\b', '', text)
    # Split concatenated words (simple: insert space before capitals won't work after lowering)
    # Expand legal suffixes
    tokens = text.split()
    out = []
    
    # Load dict if not loaded
    if _INDIC_DICT is None:
        load_indic_dict()
        
    for t in tokens:
        # Apply indic transliteration replacement
        if _INDIC_DICT and t in _INDIC_DICT:
            t = _INDIC_DICT[t]
            
        if t in LEGAL_SUFFIXES:
            out.append(LEGAL_SUFFIXES[t])
        elif t in ('aka', 'dba', 'ta', 'fka'):
            continue  # drop trade-name markers
        else:
            out.append(t)
    return ' '.join(out)


def normalize_address(addr):
    """Normalize a business address."""
    text = normalize_text(addr)
    if not text:
        return ''
    tokens = text.split()
    out = []
    for t in tokens:
        if t in ADDRESS_ABBREVS:
            out.append(ADDRESS_ABBREVS[t])
        elif t in US_STATES:
            out.append(US_STATES[t])
        else:
            out.append(t)
    return ' '.join(out)


def extract_street_number(addr_normalized):
    """Extract the first number from an address."""
    if not addr_normalized:
        return ''
    m = re.search(r'\b(\d+)\b', addr_normalized)
    return m.group(1) if m else ''


def extract_pin_zip(addr_normalized):
    """Extract PIN/ZIP code from address."""
    if not addr_normalized:
        return ''
    # US ZIP: 5 digits
    m = re.search(r'\b(\d{5})\b', addr_normalized)
    if m:
        return m.group(1)
    # India PIN: 6 digits
    m = re.search(r'\b(\d{6})\b', addr_normalized)
    if m:
        return m.group(1)
    return ''
