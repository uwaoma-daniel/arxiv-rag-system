# paste full test file content here
# tests/test_data_processing.py
"""
Unit tests for src/data_processing.py

Run with:
    pytest tests/test_data_processing.py -v --tb=short
"""

import sys
import json
import tempfile
from pathlib import Path
sys.path.insert(0, '/kaggle/working')

import pandas as pd
import pytest
from src.data_processing import (
    clean_abstract,
    parse_authors,
    filter_categories,
    filter_date_range,
    validate_records,
    deduplicate,
    stratified_sample,
    export_parquet,
    build_metadata_columns,
)


# ── helpers ──────────────────────────────────────────────────────────

LONG_ABSTRACT = (
    'This paper presents a novel approach to machine learning using attention. '
    'We propose a new transformer model that achieves state-of-the-art results. '
    'Our method is evaluated on standard NLP benchmarks and shows improvements. '
    'The model captures long-range dependencies via multi-head self-attention. '
    'We demonstrate significant gains over existing strong baselines on all tasks. '
    'Ablation studies confirm the contribution of each architectural component. '
)


def make_df(n=5):
    cats = ['cs.LG cs.AI', 'cs.CL', 'cs.CV', 'stat.ML', 'cs.IR']
    dates = ['2022-01-15', '2021-06-20', '2023-03-10', '2020-11-05', '2024-07-01']
    return pd.DataFrame({
        'id':          [f'2301.{i:05d}' for i in range(n)],
        'title':       [f'Paper Title {i}' for i in range(n)],
        'abstract':    [LONG_ABSTRACT for _ in range(n)],
        'authors':     [f'Smith{i}, J., Jones{i}, A.' for i in range(n)],
        'categories':  cats[:n],
        'update_date': dates[:n],
    })


# ── clean_abstract ────────────────────────────────────────────────────

class TestCleanAbstract:

    def test_removes_latex_math(self):
        result = clean_abstract('We use $\\alpha$ and $\\beta$ in our model.')
        assert '$' not in result

    def test_removes_latex_commands(self):
        result = clean_abstract('We use \\textbf{bold} and \\emph{italic} text.')
        assert '\\' not in result

    def test_normalizes_whitespace(self):
        result = clean_abstract('Word1   Word2\t\tWord3\n\nWord4')
        assert '  ' not in result
        assert '\t' not in result
        assert '\n' not in result

    def test_handles_non_string(self):
        assert clean_abstract(None) == ''
        assert clean_abstract(123) == ''

    def test_returns_string(self):
        assert isinstance(clean_abstract('Hello world.'), str)


# ── parse_authors ─────────────────────────────────────────────────────

class TestParseAuthors:

    def test_single_author(self):
        result = parse_authors('Smith, John')
        assert result['author_count'] == 1
        assert 'et al' not in result['citation_str']
        assert 'and' not in result['citation_str']

    def test_two_authors(self):
        result = parse_authors('Smith, J., Jones, A.')
        assert result['author_count'] == 2
        assert 'and' in result['citation_str']

    def test_three_or_more_authors(self):
        result = parse_authors('Smith, J., Jones, A., Brown, C.')
        assert result['author_count'] == 3
        assert 'et al' in result['citation_str']

    def test_empty_string(self):
        result = parse_authors('')
        assert result['first_author_last'] == 'Unknown'

    def test_returns_dict_with_required_keys(self):
        result = parse_authors('Vaswani, A.')
        assert 'first_author_last' in result
        assert 'author_count' in result
        assert 'citation_str' in result


# ── filter_categories ────────────────────────────────────────────────

class TestFilterCategories:

    def test_keeps_matching_rows(self):
        df = make_df(5)
        result = filter_categories(df, ['cs.LG'])
        assert len(result) >= 1
        for _, row in result.iterrows():
            assert 'cs.LG' in str(row['categories'])

    def test_removes_non_matching_rows(self):
        df = make_df(5)
        result = filter_categories(df, ['cs.RO'])
        assert len(result) == 0

    def test_handles_missing_column(self):
        df = pd.DataFrame({'id': [1, 2]})
        result = filter_categories(df, ['cs.LG'])
        assert len(result) == 0


# ── filter_date_range ────────────────────────────────────────────────

class TestFilterDateRange:

    def test_keeps_rows_in_range(self):
        df = make_df(5)
        result = filter_date_range(df, '2020-01-01', '2024-12-31')
        assert len(result) >= 1

    def test_removes_rows_outside_range(self):
        df = make_df(3)
        result = filter_date_range(df, '2025-01-01', '2025-12-31')
        assert len(result) == 0

    def test_returns_dataframe(self):
        df = make_df(3)
        result = filter_date_range(df, '2018-01-01', '2024-12-31')
        assert isinstance(result, pd.DataFrame)


# ── validate_records ─────────────────────────────────────────────────

class TestValidateRecords:

    def test_valid_records_pass(self):
        df = make_df(5)
        result = validate_records(df)
        assert len(result) == 5, (
            f'Expected 5 valid records, got {len(result)}. '
            f'Abstract word count: {len(LONG_ABSTRACT.split())}'
        )

    def test_removes_empty_title(self):
        df = make_df(3)
        df.loc[0, 'title'] = ''
        result = validate_records(df)
        assert len(result) == 2, f'Expected 2, got {len(result)}'

    def test_removes_short_abstract(self):
        df = make_df(3)
        df.loc[0, 'abstract'] = 'Too short.'
        result = validate_records(df)
        assert len(result) == 2, f'Expected 2, got {len(result)}'

    def test_removes_duplicate_ids(self):
        df = make_df(3)
        df.loc[1, 'id'] = df.loc[0, 'id']
        result = validate_records(df)
        assert len(result) == 2, f'Expected 2, got {len(result)}'

    def test_returns_dataframe(self):
        result = validate_records(make_df(3))
        assert isinstance(result, pd.DataFrame)


# ── deduplicate ──────────────────────────────────────────────────────

class TestDeduplicate:

    def test_removes_exact_duplicates(self):
        df = make_df(3)
        df.loc[2, 'id'] = df.loc[0, 'id']
        result = deduplicate(df, key='id')
        assert len(result) == 2

    def test_no_duplicates_unchanged(self):
        df = make_df(3)
        result = deduplicate(df, key='id')
        assert len(result) == 3


# ── stratified_sample ────────────────────────────────────────────────

class TestStratifiedSample:

    def test_returns_correct_approximate_total(self):
        df = pd.DataFrame({
            'id': [f'p{i}' for i in range(100)],
            'categories': ['cs.LG'] * 50 + ['cs.AI'] * 50,
        })
        result = stratified_sample(
            df, n=20,
            category_weights={'cs.LG': 0.5, 'cs.AI': 0.5},
        )
        assert 15 <= len(result) <= 25

    def test_reproducible_with_same_seed(self):
        df = pd.DataFrame({
            'id': [f'p{i}' for i in range(50)],
            'categories': ['cs.LG'] * 50,
        })
        r1 = stratified_sample(df, n=10, category_weights={'cs.LG': 1.0}, random_state=42)
        r2 = stratified_sample(df, n=10, category_weights={'cs.LG': 1.0}, random_state=42)
        assert list(r1['id']) == list(r2['id'])


# ── export_parquet ───────────────────────────────────────────────────

class TestExportParquet:

    def test_creates_parquet_file(self):
        df = make_df(3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f'{tmpdir}/test.parquet'
            export_parquet(df, path)
            assert Path(path).exists()
            assert Path(path).stat().st_size > 0

    def test_parquet_reloads_correctly(self):
        df = make_df(3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f'{tmpdir}/test.parquet'
            export_parquet(df, path)
            reloaded = pd.read_parquet(path)
            assert len(reloaded) == len(df)
            assert list(reloaded.columns) == list(df.columns)


# ── build_metadata_columns ───────────────────────────────────────────

class TestBuildMetadataColumns:

    def test_adds_required_columns(self):
        df = make_df(3)
        result = build_metadata_columns(df)
        for col in ['first_author_last', 'author_count', 'citation_str', 'year']:
            assert col in result.columns, f'Missing column: {col}'

    def test_year_is_integer(self):
        df = make_df(3)
        result = build_metadata_columns(df)
        assert result['year'].dtype in ['int32', 'int64']

    def test_abstract_cleaned(self):
        df = make_df(3)
        df.loc[0, 'abstract'] = 'Text with $\\alpha$ math and more words here.'
        result = build_metadata_columns(df)
        assert '$' not in result.loc[0, 'abstract']
