"""Update Link, and the password prompt a server-database link needs.

A file or web link's settings carry everything a refresh needs. A PostgreSQL
or MySQL link never carries its password - see
app.utils.data_sources.DatabaseConnection.to_link_settings - so refreshing
one has to ask for it fresh, every time, rather than silently failing deep
inside execute_import with no chance to supply it.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.widgets import table_list as table_list_module
from app.widgets.table_list import TableListPanel


@pytest.fixture
def repo(tmp_db_path: Path) -> SqliteRepo:
    for path in (
        tmp_db_path,
        tmp_db_path.with_suffix(".dhub-wal"),
        tmp_db_path.with_suffix(".dhub-shm"),
    ):
        path.unlink(missing_ok=True)

    built = SqliteRepo(db_path=tmp_db_path)
    built.import_dataframe(
        pd.DataFrame({"a": [1, 2]}), table_name="local_only", normalize_columns=False
    )
    yield built
    built.close()


@pytest.fixture
def panel(qapp, repo: SqliteRepo):
    built = TableListPanel(repo=repo, parent=None)
    built.reload()
    return built


def _select(panel: TableListPanel, name: str) -> None:
    for row in range(panel._model.rowCount()):
        item = panel._model.item(row, panel.COL_TABLE)
        if item is not None and str(item.data(panel.ROLE_TABLE_NAME)) == name:
            panel._view.selectionModel().setCurrentIndex(
                panel._model.index(row, 0),
                panel._view.selectionModel().SelectionFlag.ClearAndSelect
                | panel._view.selectionModel().SelectionFlag.Rows,
            )
            return
    raise AssertionError(f"{name} is not in the list")


def _make_link(repo: SqliteRepo, table: str, source: dict) -> None:
    repo.import_dataframe(pd.DataFrame({"x": [1]}), table_name=table, normalize_columns=False)
    repo.upsert_link(
        table_name=table,
        source_path=source.get("path") or source.get("url") or source.get("host", ""),
        settings={
            "source": source,
            "read": {"skiprows": 0, "skip_last": 0, "header": True},
            "destination": {"table": table, "normalize_columns": True},
            "columns": {"types": {"x": "Auto"}},
        },
    )


def test_refreshing_a_file_link_asks_for_no_password(
    panel: TableListPanel, repo: SqliteRepo, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    csv_path = tmp_path / "source.csv"
    csv_path.write_text("x\n9\n", encoding="utf-8")
    _make_link(repo, "file_linked", {"kind": "file", "path": str(csv_path), "sheet": None})
    panel.reload()
    _select(panel, "file_linked")

    from PySide6.QtWidgets import QInputDialog

    asked: list[object] = []
    monkeypatch.setattr(
        QInputDialog,
        "getText",
        staticmethod(lambda *a, **k: asked.append(1) or ("", False)),
    )

    panel._refresh_link_for_table()

    assert asked == []
    assert repo.query_df("SELECT x FROM file_linked")["x"].tolist() == [9]


def test_refreshing_a_postgres_link_asks_for_a_password(
    panel: TableListPanel, repo: SqliteRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_link(
        repo,
        "pg_linked",
        {
            "kind": "postgres", "host": "db.example.com", "port": 5432,
            "database": "analytics", "username": "reader", "table": "orders",
        },
    )
    panel.reload()
    _select(panel, "pg_linked")

    from PySide6.QtWidgets import QInputDialog

    prompts: list[str] = []
    monkeypatch.setattr(
        QInputDialog,
        "getText",
        staticmethod(lambda _p, _title, label, *_a, **_k: (prompts.append(label) or ("hunter2", True))),
    )

    calls: dict[str, object] = {}

    def fake_refresh_link(_repo, *, link_id, password=None):
        calls["link_id"] = link_id
        calls["password"] = password
        from app.utils.import_runner import LinkRefreshResult
        return LinkRefreshResult(link_id=link_id, table_name="pg_linked", rows=0, cols=0)

    monkeypatch.setattr(table_list_module, "refresh_link", fake_refresh_link)

    panel._refresh_link_for_table()

    assert len(prompts) == 1
    assert "reader" in prompts[0] and "db.example.com" in prompts[0]
    assert calls["password"] == "hunter2"


def test_declining_the_password_prompt_does_not_refresh(
    panel: TableListPanel, repo: SqliteRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_link(
        repo,
        "mysql_linked",
        {
            "kind": "mysql", "host": "db.example.com", "port": 3306,
            "database": "shop", "username": "reader", "table": "orders",
        },
    )
    panel.reload()
    _select(panel, "mysql_linked")

    from PySide6.QtWidgets import QInputDialog

    monkeypatch.setattr(
        QInputDialog, "getText", staticmethod(lambda *_a, **_k: ("", False))
    )

    called = []
    monkeypatch.setattr(
        table_list_module, "refresh_link", lambda *_a, **_k: called.append(1)
    )

    panel._refresh_link_for_table()

    assert called == []
