# [paste the entire data_processing.py content here]
# src/data_processing.py
"""
Module: data_processing.py
Purpose: Load, filter, clean, validate, and export arXiv dataset.

All public functions are pure or side-effect-isolated so they are
fully unit-testable without a filesystem.
"""

from __future__ import annotations

import html
import json
import logging
import re
from typing import Iterator, List, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

TARGET_CATEGORIES = [
    'cs.AI', 'cs.LG', 'cs.CL', 'cs.CV',
    'stat.ML', 'cs.IR', 'cs.NE', 'cs.RO',
]

CATEGORY_DISTRIBUTION = {
    'cs.AI': 0.20, 'cs.LG': 0.20, 'cs.CL': 0.15,
    'cs.CV': 0.15, 'stat.ML': 0.10, 'cs.IR': 0.10,
    'cs.NE': 0.05, 'cs.RO': 0.05,
}


def load_arxiv_jsonl(path: str, chunksize: int = 10_000) -> Iterator[pd.DataFrame]:
    chunk: List[dict] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                chunk.append(json.loads(line))
            except Exception:
                continue
            if len(chunk) >= chunksize:
                yield pd.DataFrame(chunk)
                chunk = []
    if chunk:
        yield pd.DataFrame(chunk)


def filter_categories(df: pd.DataFrame, categories: List[str]) -> pd.DataFrame:
    if 'categories' not in df.columns:
        return df.iloc[0:0]
    mask = df['categories'].apply(
        lambda cats: any(cat in str(cats).split() for cat in categories)
    )
    return df[mask].copy()


def filter_date_range(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    date_col = None
    for col in ['update_date', 'versions', 'created']:
        if col in df.columns:
            date_col = col
            break
    if date_col is None:
        return df
    df = df.copy()
    df['_parsed_date'] = pd.to_datetime(df[date_col], errors='coerce', utc=True)
    start_dt = pd.Timestamp(start, tz='UTC')
    end_dt   = pd.Timestamp(end,   tz='UTC')
    mask = (df['_parsed_date'] >= start_dt) & (df['_parsed_date'] <= end_dt)
    return df[mask].drop(columns=['_parsed_date']).copy()


def clean_abstract(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = re.sub(r'\$[^$]+\$', ' ', text)
    text = re.sub(r'\\begin\{.*?\}', ' ', text, flags=re.DOTALL)
    text = re.sub(r'\\end\{.*?\}',   ' ', text, flags=re.DOTALL)
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)
    text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def parse_authors(authors_raw: str) -> Dict[str, object]:
    """
    Extract first author last name, author count, and citation string.

    Handles three arXiv author string formats:
      Format A:  'Vaswani, A., Shazeer, N., Parmar, N.'
                 Multiple authors as Last, Initial pairs.
                 Token count is even; odd-indexed tokens are short initials.
      Format B:  'Smith, John'
                 Single author as Last, FirstName.
                 Exactly 2 tokens; second token is NOT a short initial.
      Format C:  'John Smith, Jane Jones'
                 Full names separated by commas.
    """
    if not isinstance(authors_raw, str) or not authors_raw.strip():
        return {'first_author_last': 'Unknown', 'author_count': 0, 'citation_str': 'Unknown'}

    tokens = [t.strip() for t in authors_raw.split(',') if t.strip()]

    def is_initial(s: str) -> bool:
        # An initial is 1-3 chars (possibly followed by a period)
        # e.g. 'A', 'A.', 'AB', 'J'
        cleaned = s.strip().rstrip('.')
        return len(cleaned) <= 3 and cleaned.replace(' ', '') != ''

    # ── Format B: exactly 2 tokens, second is NOT a short initial ──
    # 'Smith, John' -> ['Smith', 'John'] -> single author
    # Format B: exactly 2 tokens, second is NOT a short initial,
    # AND both tokens are single words (= 'LastName, FirstName' for one person).
    # If either token has multiple words it's two separate author names.
    # Examples:
    #   'Smith, John'         -> 2 tokens, both 1 word  -> 1 author
    #   'John Smith, Jane Jones' -> 2 tokens, both 2 words -> 2 authors
    def is_single_word(s: str) -> bool:
        return len(s.strip().split()) == 1

    if len(tokens) == 2 and not is_initial(tokens[1]) and all(is_single_word(t) for t in tokens):
        lastname = tokens[0]
        return {
            'first_author_last': lastname,
            'author_count': 1,
            'citation_str': lastname,
        }

    # ── Format A: even token count, all odd-indexed tokens are initials ──
    # 'Smith, J., Jones, A.' -> ['Smith', 'J.', 'Jones', 'A.']
    format_a = (
        len(tokens) >= 2
        and len(tokens) % 2 == 0
        and all(is_initial(tokens[i]) for i in range(1, len(tokens), 2))
    )

    if format_a:
        lastnames = [tokens[i] for i in range(0, len(tokens), 2)]
    else:
        # Format C: each comma-token is a full author name
        lastnames = [t.split()[-1] for t in tokens if t]

    count = len(lastnames)
    first = lastnames[0] if lastnames else 'Unknown'

    if count == 1:
        citation = first
    elif count == 2:
        citation = f'{first} and {lastnames[1]}'
    else:
        citation = f'{first} et al.'

    return {'first_author_last': first, 'author_count': count, 'citation_str': citation}


def validate_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply data quality rules. Returns only valid records.

    Rules (applied in order, each narrows the set):
      1. Abstract word count: 50 <= words <= 1000
      2. Non-null, non-empty title
      3. Non-null, non-empty abstract
      4. Deduplicate by id
    """
    df = df.copy()
    initial = len(df)

    # Rule 1: abstract word count
    if 'abstract' in df.columns:
        wc = df['abstract'].apply(
            lambda x: len(str(x).split()) if pd.notna(x) and str(x).strip() else 0
        )
        df = df[(wc >= 50) & (wc <= 1000)].copy()

    # Rule 2: non-empty title
    if 'title' in df.columns:
        df = df[df['title'].notna() & (df['title'].astype(str).str.strip() != '')].copy()

    # Rule 3: non-empty abstract
    if 'abstract' in df.columns:
        df = df[df['abstract'].notna() & (df['abstract'].astype(str).str.strip() != '')].copy()

    # Rule 4: deduplicate by id
    if 'id' in df.columns:
        df = df.drop_duplicates(subset=['id']).copy()

    logger.info(f'validate_records: {initial} -> {len(df)} records')
    return df.reset_index(drop=True)


def deduplicate(df: pd.DataFrame, key: str = 'id') -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=[key]).reset_index(drop=True)
    logger.info(f'deduplicate: removed {before - len(df):,} duplicates')
    return df


def stratified_sample(
    df: pd.DataFrame,
    n: int,
    category_weights: Dict[str, float],
    category_col: str = 'categories',
    random_state: int = 42,
) -> pd.DataFrame:
    sampled: List[pd.DataFrame] = []
    for cat, weight in category_weights.items():
        target_n = int(n * weight)
        subset = df[df[category_col].str.contains(cat, na=False)]
        if len(subset) == 0:
            continue
        take = min(target_n, len(subset))
        sampled.append(subset.sample(n=take, random_state=random_state))
    result = pd.concat(sampled, ignore_index=True) if sampled else pd.DataFrame()
    id_col = 'id' if 'id' in result.columns else None
    if id_col:
        result = result.drop_duplicates(subset=[id_col])
    return result.reset_index(drop=True)


def export_parquet(df: pd.DataFrame, path: str) -> None:
    from pathlib import Path as _Path
    _Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression='snappy', index=False)
    logger.info(f'Exported {len(df):,} records to {path}')


def build_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'authors' in df.columns:
        parsed = df['authors'].apply(parse_authors)
        df['first_author_last'] = parsed.apply(lambda x: x['first_author_last'])
        df['author_count']      = parsed.apply(lambda x: x['author_count'])
        df['citation_str']      = parsed.apply(lambda x: x['citation_str'])
    if 'update_date' in df.columns:
        df['year'] = pd.to_datetime(
            df['update_date'], errors='coerce'
        ).dt.year.fillna(2023).astype(int)
    if 'abstract' in df.columns:
        df['abstract'] = df['abstract'].apply(clean_abstract)
    if 'categories' in df.columns:
        df['primary_category'] = df['categories'].apply(
            lambda x: str(x).split()[0] if isinstance(x, str) else 'cs.AI'
        )
    return df
