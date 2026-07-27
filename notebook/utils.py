"""
Shared preprocessing utilities for the CV Screening Engine.

Both 01_data_preprocessing.ipynb and 03_prediction.ipynb import this file
directly, instead of each having their own copy of clean_text() and the
other helper functions.

This is the actual fix for training-serving skew: skew happens when the
notebook that TRAINS the model and the notebook that SERVES predictions
end up with two slightly different copies of the same cleaning function
(one gets edited later and the other doesn't). Importing one shared file
makes that impossible -- there is only one copy of the logic to edit.
"""

import re
import ast
from datetime import datetime

import numpy as np
import spacy

# ── spaCy model, loaded once and cached ─────────────────────────────────────
_NLP = None


HR_FLUFF = {
    'team', 'player', 'highly', 'motivated', 'results', 'driven', 'synergy',
    'experienced', 'proficient', 'strong', 'excellent', 'dynamic', 'innovative',
    'passionate', 'dedicated', 'self', 'starter', 'goal', 'oriented',
    'communication', 'ability', 'responsible', 'proactive', 'collaborative',
    'hardworking', 'detail', 'fast', 'learner',
    'year', 'years', 'month', 'months', 'present', 'current', 'till', 'date',
    'company', 'ltd', 'pvt', 'inc', 'llc', 'corp', 'group', 'role', 'position',
    'responsibilities', 'responsibility', 'duties',
    'university', 'college', 'institute', 'school', 'campus',
    'city', 'state', 'country', 'address', 'location',

    'robust', 'handling', 'fresher', 'looking', 'join',
    'meaningful', 'reputed', 'relevant', 'discipline', 'activities',

    'seeking', 'career', 'objective', 'summary', 'proven', 'track', 'record',
    'successful', 'successfully', 'demonstrated', 'skills', 'knowledge', 'environment',
    'organization', 'opportunity', 'growth', 'fast-paced', 'hands-on',
    'working', 'understanding', 'expertise', 'candidate', 'required',

    'enthusiastic', 'committed', 'exceptional', 'outstanding', 'professional',
    'various', 'multiple', 'numerous', 'diverse', 'wide', 'range',
    'flexible', 'adaptable', 'versatile', 'efficient', 'effective', 'effectively',
    'interpersonal', 'possess', 'possessing', 'demonstrate', 'demonstrating',
    'ensure', 'ensuring', 'maintain', 'provide', 'providing', 'deliver', 'delivering',
    'job', 'related', 'appropriate', 'suitable', 'level', 'levels', 'basis', 'daily',
}

# clean_text() was stripping digits/symbols before the length/lemma filter,
# which destroyed short technical terms ("AI", "ML", "C++", "AWS" -> "aw"
# after lemmatization). These two maps protect them.
TECH_TERM_MAP = {
    r'c\+\+': 'cplusplus', r'c#': 'csharp', r'\.net\b': 'dotnet',
    r'\bnode\.js\b': 'nodejs', r'\bci\s*/\s*cd\b': 'cicd', r'\bk8s\b': 'kubernetes',
    r'\bux\s*/\s*ui\b': 'ux ui', r'\bui\s*/\s*ux\b': 'ui ux',
}
PROTECTED_SHORT_TERMS = {
    'ai', 'ml', 'ux', 'ui', 'qa', 'bi', 'js', 'hr', 'ar', 'vr', 'os', 'db',
    'ci', 'cd', 'aws', 'gcp', 'sql', 'api', 'css', 'sdk', 'kubernetes', 'devops',
}


def get_nlp():
    """Load spaCy once and cache it. Every caller gets the exact same
    pipeline with the exact same stop-word list, in both notebooks."""
    global _NLP
    if _NLP is None:
        _NLP = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
        for w in HR_FLUFF:
            _NLP.vocab[w].is_stop = True
    return _NLP


def clean_text(text):
    """Lowercase, strip URLs/emails/punctuation, lemmatise, remove stop words.
    This exact function is what TF-IDF and SBERT-facing text is built from
    during training, and it is what score_cv() must call at prediction time
    too -- otherwise the model sees a different kind of text than it learned on."""
    if not isinstance(text, str) or not text.strip():
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    for pattern, replacement in TECH_TERM_MAP.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    doc = get_nlp()(text)
    tokens = []
    for t in doc:
        if t.is_stop:
            continue
        if t.text in PROTECTED_SHORT_TERMS:
            tokens.append(t.text)
        elif len(t.text) > 2:
            tokens.append(t.lemma_)
    return ' '.join(tokens)


# ── address normalisation ────────────────────────────────────────────────
def extract_city_state(raw):
    """Turn a messy free-text address into a clean 'City, State' string."""
    if not raw or str(raw).strip() in ('', 'nan', 'None'):
        return 'Unknown'
    addr = str(raw).strip()
    addr = re.sub(r'(?i)\b(canada|usa|united\s+states|uk|country|zip\s+code)\b', '', addr).strip().rstrip(',')
    addr = re.sub(r'\b\d{5}(?:-\d{4})?\b', '', addr)
    addr = re.sub(r'\b[A-Za-z]\d[A-Za-z]\s?\d[A-Za-z]\d\b', '', addr)
    addr = re.sub(r'\b[A-Z]{1,2}\d[A-Z\d]?\s\d[A-Z]{2}\b', '', addr)
    addr = addr.strip().rstrip(',')
    if ',' in addr:
        parts = [p.strip() for p in addr.split(',')]
        state, city_raw = parts[-1], parts[-2]
        city_raw = re.sub(r'(?i)\b(apartment|apt|rm\.?|room|suite|ste)\s*[\w.]+\b', '', city_raw).strip()
        return f"{city_raw}, {state}"
    return addr


# ── list-string parsing ──────────────────────────────────────────────────
def safe_extract_list(val):
    """The dataset stores multi-value fields (companies, dates, passing
    years) as text that looks like a Python list, e.g. "['2018', '2020']".
    This turns that string back into a real list. A single plain value
    that was never wrapped in brackets is returned as a one-item list."""
    if not isinstance(val, str) or val.strip() in ('', 'nan', 'None'):
        return []
    if val.strip().startswith('['):
        try:
            return ast.literal_eval(val)
        except Exception:
            return []
    return [val]


# ── work-experience calculation ──────────────────────────────────────────
def get_role_experience_years(start_str, end_str):
    """Decimal years between a job's start and end date. 'Present' / 'Current'
    are treated as today, since the candidate is still working there."""
    if not start_str or str(start_str).lower() in ('none', 'nan', ''):
        return 0.0
    try:
        from dateutil import parser as dateparser
        start_dt = dateparser.parse(str(start_str), fuzzy=True)
    except Exception:
        return 0.0
    ongoing = {'present', 'current', 'till date', 'ongoing', 'none', '', 'nan', 'n/a'}
    if not end_str or str(end_str).lower().strip() in ongoing:
        end_dt = datetime.now()
    else:
        try:
            from dateutil import parser as dateparser
            end_dt = dateparser.parse(str(end_str), fuzzy=True)
        except Exception:
            end_dt = datetime.now()
    days = (end_dt - start_dt).days
    return round(max(0, days) / 365.25, 1)


# ── graduation-year based experience & age proxy ─────────────────────────
TYPICAL_GRADUATION_AGE = 22  # approximation: most candidates finish their first
                             # degree around this age. Used only to derive a rough
                             # implied age when the dataset has no birth date field.


def extract_graduation_year(passing_years_raw):
    """Pull the most recent 4-digit year out of a passing_years field that
    may contain one degree or a list of several degrees."""
    years = safe_extract_list(passing_years_raw)
    parsed = []
    for y in years:
        match = re.search(r'(19|20)\d{2}', str(y))
        if match:
            parsed.append(int(match.group()))
    return max(parsed) if parsed else None


def compute_years_since_graduation(passing_years_raw, current_year=None):
    current_year = current_year or datetime.now().year
    grad_year = extract_graduation_year(passing_years_raw)
    if grad_year is None:
        return None
    return max(0.0, float(current_year - grad_year))


def compute_implied_age(passing_years_raw, current_year=None):
    """Rough age estimate = years since graduation + typical graduation age.
    This is an approximation, not a real age -- see the caveat in the
    Stage 5 markdown of 01_data_preprocessing.ipynb before relying on it."""
    yrs = compute_years_since_graduation(passing_years_raw, current_year)
    return None if yrs is None else TYPICAL_GRADUATION_AGE + yrs


def parse_age_requirement(age_str):
    """Extract the maximum allowed age from JD text like 'below 30',
    '25-35 years', or 'max age 40'. Returns the largest number found,
    which is the upper bound in a range or the single stated cap."""
    if not age_str or str(age_str).strip() in ('', 'nan', 'None'):
        return None
    nums = [int(n) for n in re.findall(r'\d+', str(age_str))]
    return max(nums) if nums else None


# ── experience-requirement parsing (dual-number: min + preferred) ───────
def parse_experience_range(exp_str):
    """Extract BOTH the minimum required years and the preferred/ideal years
    from a JD's experience requirement text. '3-5 years' becomes (3, 5) not
    just 3 -- so a hard gate can use the minimum while a scoring feature can
    reward candidates who reach the preferred level too."""
    if not exp_str or str(exp_str).strip() == '':
        return 0.0, 0.0
    s = str(exp_str).lower()
    if any(w in s for w in ['fresher', 'intern', 'entry', 'no experience']):
        return 0.0, 0.0
    nums = [float(n) for n in re.findall(r'\d+(?:\.\d+)?', s)]
    if not nums:
        return 0.0, 0.0
    if len(nums) == 1:
        return nums[0], nums[0]
    return min(nums[:2]), max(nums[:2])


def exp_tier(years):
    """Human-readable experience bracket, used for EDA and reporting."""
    if years is None or years == 0:
        return 'Fresher'
    if years <= 2:
        return 'Junior (0-2y)'
    if years <= 5:
        return 'Mid (2-5y)'
    if years <= 10:
        return 'Senior (5-10y)'
    return 'Lead (10y+)'
