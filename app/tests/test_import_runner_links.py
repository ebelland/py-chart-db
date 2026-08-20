"""Tests for import links: refresh always replaces, ignored columns stay out."""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from app.data.sqlite_repo import SqliteRepo
from app.utils.import_runner import refresh_link


def _write_csv(path: Path, rows: list[tuple[object, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['a', 'b', 'c'])
        w.writerows(rows)


def test_refresh_link_forces_replace_even_if_append(tmp_db_path: Path, test_results_dir: Path) -> None:
    repo = SqliteRepo(db_path=tmp_db_path)

    csv_path = test_results_dir / 'link_source.csv'
    _write_csv(csv_path, [(1, 10, 100), (2, 20, 200), (3, 30, 300)])

    settings = {
        'source': {'kind': 'file', 'path': str(csv_path), 'sheet': None},
        'read': {'skiprows': 0, 'skip_last': 0, 'header': True, 'delimiter': ',', 'encoding': 'utf-8'},
        'destination': {'table': 't_link', 'if_exists': 'append', 'normalize_columns': True},
        'columns': {'types': {'a': 'Auto', 'b': 'Auto', 'c': 'Auto'}},
    }

    link_id = repo.upsert_link(table_name='t_link', source_path=str(csv_path), settings=settings)

    repo.query_df('CREATE TABLE IF NOT EXISTS t_link (a INTEGER, b INTEGER, c INTEGER)')
    repo.query_df('INSERT INTO t_link (a,b,c) VALUES (999,999,999)')

    refresh_link(repo, link_id=link_id)

    db_path = repo.ensure_dhub_extension(tmp_db_path)
    with sqlite3.connect(str(db_path)) as con:
        n = con.execute('SELECT COUNT(*) FROM t_link').fetchone()[0]
    assert int(n) == 3


def test_ignore_column_not_imported(tmp_db_path: Path, test_results_dir: Path) -> None:
    repo = SqliteRepo(db_path=tmp_db_path)

    csv_path = test_results_dir / 'ignore_source.csv'
    _write_csv(csv_path, [(1, 10, 100), (2, 20, 200)])

    settings = {
        'source': {'kind': 'file', 'path': str(csv_path), 'sheet': None},
        'read': {'skiprows': 0, 'skip_last': 0, 'header': True, 'delimiter': ',', 'encoding': 'utf-8'},
        'destination': {'table': 't_ignore', 'if_exists': 'replace', 'normalize_columns': True},
        'columns': {'types': {'a': 'Auto', 'b': 'Ignore', 'c': 'Auto'}},
    }

    link_id = repo.upsert_link(table_name='t_ignore', source_path=str(csv_path), settings=settings)
    refresh_link(repo, link_id=link_id)

    db_path = repo.ensure_dhub_extension(tmp_db_path)
    with sqlite3.connect(str(db_path)) as con:
        cols = [r[1] for r in con.execute('PRAGMA table_info(t_ignore)').fetchall()]
    assert cols == ['a', 'c']
