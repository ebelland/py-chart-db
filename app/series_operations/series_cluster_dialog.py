"""Chart-series clustering dialog.

Drop-in companion for the shared ``SeriesOperationDialogBase`` workflow.

Features
--------
- Select one chart axis and one or more source series.
- Cluster selected source data with ``scipy.cluster``:
  - K-means / vector quantization via ``scipy.cluster.vq``
  - Hierarchical / agglomerative clustering via ``scipy.cluster.hierarchy``
- Add a ``ClusterId`` column to the generated dataset.
- Render either:
  - one series colored by ``ClusterId`` through the Color role; or
  - separate chart series per cluster using ``WHERE ClusterId = x`` queries.
- Fill the results pane with a plain-text preview summary.

Notes
-----
This module reuses the shared SeriesOperationDialogBase preview/apply flow.
Clustering overrides only the axis-application step because it updates source
table ClusterId values and either recolors the selected series or creates split
series per cluster.
"""

from __future__ import annotations

import html
import re
import webbrowser
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Any, Final, cast

import numpy as np
import pandas as pd
from matplotlib import rcParams
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from scipy.cluster import hierarchy
from scipy.cluster import vq

from app.data.data_source import parse_roles
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.utils.messages import show_message
from app.series_operations.series_operation_dialog_base import (
    ResultSeriesSpec,
    SeriesOperationDialogBase,
)
from app.styles.style import (
    create_card_widget,
    stdSizeAndlayout,
)
from app.widgets.axis_series_selector_widget import AxisSeriesSelector
from app.utils.i18n import _


CLUSTER_KMEANS: Final[str] = "K-means / vector quantization"
CLUSTER_HIERARCHICAL: Final[str] = "Hierarchical / agglomerative"
CLUSTER_SKLEARN: Final[str] = "scikit-learn"

FEATURE_XY: Final[str] = "X and Y"
FEATURE_Y_ONLY: Final[str] = "Y only"
FEATURE_XYZ: Final[str] = "X, Y and Z"
FEATURE_ALL_NUMERIC: Final[str] = "All numeric columns"

# Clustering writes into the user's own table, so a preview has to be able to
# put the previous state back byte for byte.  The backup column is renamed
# aside rather than copied: O(1) metadata, and it restores the original values
# rather than a re-computed approximation.
CLUSTER_COLUMN: Final[str] = "ClusterId"
CLUSTER_BACKUP_COLUMN: Final[str] = "__ClusterId_preview_backup__"

RENDER_COLORED_SERIES: Final[str] = "Single series colored by ClusterId"
RENDER_SEPARATE_SERIES: Final[str] = "Separate series per cluster"

TOOL_WHITEN: Final[str] = "vq.whiten + kmeans2"
TOOL_VQ: Final[str] = "vq.vq"
TOOL_KMEANS: Final[str] = "vq.kmeans"
TOOL_KMEANS2: Final[str] = "vq.kmeans2"
TOOL_FCLUSTER: Final[str] = "hierarchy.fcluster"
TOOL_FCLUSTERDATA: Final[str] = "hierarchy.fclusterdata"
TOOL_LEADERS: Final[str] = "hierarchy.leaders"

KMEANS_TOOLS: Final[tuple[str, ...]] = (
    TOOL_WHITEN,
    TOOL_VQ,
    TOOL_KMEANS,
    TOOL_KMEANS2,
)
HIERARCHY_TOOLS: Final[tuple[str, ...]] = (
    TOOL_FCLUSTER,
    TOOL_FCLUSTERDATA,
    TOOL_LEADERS,
)

SKLEARN_KMEANS: Final[str] = "sklearn.KMeans"
SKLEARN_MINIBATCH_KMEANS: Final[str] = "sklearn.MiniBatchKMeans"
SKLEARN_BISECTING_KMEANS: Final[str] = "sklearn.BisectingKMeans"
SKLEARN_AGGLOMERATIVE: Final[str] = "sklearn.AgglomerativeClustering"
SKLEARN_DBSCAN: Final[str] = "sklearn.DBSCAN"
SKLEARN_OPTICS: Final[str] = "sklearn.OPTICS"
SKLEARN_BIRCH: Final[str] = "sklearn.Birch"
SKLEARN_MEANSHIFT: Final[str] = "sklearn.MeanShift"
SKLEARN_SPECTRAL: Final[str] = "sklearn.SpectralClustering"
SKLEARN_GAUSSIAN_MIXTURE: Final[str] = "sklearn.GaussianMixture"

SKLEARN_TOOLS: Final[tuple[str, ...]] = (
    SKLEARN_KMEANS,
    SKLEARN_MINIBATCH_KMEANS,
    SKLEARN_BISECTING_KMEANS,
    SKLEARN_AGGLOMERATIVE,
    SKLEARN_DBSCAN,
    SKLEARN_OPTICS,
    SKLEARN_BIRCH,
    SKLEARN_MEANSHIFT,
    SKLEARN_SPECTRAL,
    SKLEARN_GAUSSIAN_MIXTURE,
)

HIERARCHY_METHODS: Final[tuple[str, ...]] = (
    "single",
    "complete",
    "average",
    "weighted",
    "centroid",
    "median",
    "ward",
)

HIERARCHY_METRICS: Final[tuple[str, ...]] = (
    "euclidean",
    "cityblock",
    "cosine",
    "correlation",
    "chebyshev",
    "minkowski",
)

HIERARCHY_CRITERIA: Final[tuple[str, ...]] = (
    "maxclust",
    "distance",
    "inconsistent",
)

CLUSTER_DOCS: Final[dict[str, tuple[str, str]]] = {
    CLUSTER_KMEANS: (
        "SciPy k-means / vector quantization",
        "https://docs.scipy.org/doc/scipy/reference/cluster.vq.html",
    ),
    CLUSTER_HIERARCHICAL: (
        "SciPy hierarchical clustering",
        "https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html",
    ),
    CLUSTER_SKLEARN: (
        "scikit-learn clustering",
        "https://scikit-learn.org/stable/modules/clustering.html",
    ),
}



@dataclass(slots=True)
class ClusterSeriesChoice:
    """Selectable chart series descriptor materialized from a source query."""

    name: str
    frame: pd.DataFrame
    x_col: str
    y_col: str
    z_col: str | None
    roles: dict[str, Any]
    source_table: str
    source_x_column: str
    source_sql_query: str
    source: Any | None = None


@dataclass(slots=True)
class ClusterResult:
    """Clustering output for one source series or one generated cluster series."""

    source_name: str
    result_name: str
    method: str
    frame: pd.DataFrame
    x_col: str
    y_col: str
    z_col: str | None
    feature_columns: list[str]
    metadata: dict[str, Any]

def _source_table_from_sql(sql_query: str) -> str:
    match = re.search(
        r'\bFROM\s+(?:"([^"]+)"|\[([^\]]+)\]|`([^`]+)`|([A-Za-z_][A-Za-z0-9_]*))',
        str(sql_query),
        flags=re.IGNORECASE,
    )
    if match is None:
        applogger.error(
            "Selected series SQL must contain the source table in a FROM clause.",
            show_dialog=True,
            raise_error=True,
        )
        return ""
    for group in match.groups():
        if group:
            return str(group)
    applogger.error(
        "Selected series source table could not be resolved.",
        show_dialog=True,
        raise_error=True,
    )
    return ""


def _source_column_for_alias(sql_query: str, alias: str) -> str:
    quoted_alias = re.escape(str(alias))
    alias_pattern = (
        r'"' + quoted_alias + r'"'
        r'|\[' + quoted_alias + r'\]'
        r'|`' + quoted_alias + r'`'
        r'|' + quoted_alias + r'(?=\s|,|$)'
    )
    pattern = (
        r'(?:"([^"]+)"|\[([^\]]+)\]|`([^`]+)`|([A-Za-z_][A-Za-z0-9_]*))'
        r'\s+AS\s+(?:' + alias_pattern + r')'
    )
    match = re.search(pattern, str(sql_query), flags=re.IGNORECASE)
    if match is None:
        applogger.error(
            f'Selected series SQL must project a source field as "{alias}".',
            show_dialog=True,
            raise_error=True,
        )
        return ""
    for group in match.groups():
        if group:
            return str(group)
    applogger.error(
        f'Source field for alias "{alias}" could not be resolved.',
        show_dialog=True,
        raise_error=True,
    )
    return ""


def _sql_insert_select_expression(sql_query: str, expression: str) -> str:
    """Insert or replace a SELECT expression by alias.

    Used by clustering to make the selected series expose:

        "ClusterId" AS "color"

    The function is intentionally idempotent. If the SQL already projects a
    column/expression as color, replace that projection instead of appending a
    second color column.

    Supported color aliases:
        AS color
        AS "color"
        AS [color]
        AS `color`

    This avoids duplicate DataFrame columns where df["color"] returns a
    DataFrame instead of a Series.
    """
    sql = str(sql_query).strip().rstrip(";")
    match = re.search(r"\bFROM\b", sql, flags=re.IGNORECASE)
    if match is None:
        applogger.error(
            "Selected series SQL must contain a FROM clause.",
            show_dialog=True,
            raise_error=True,
        )
        return sql

    select_part = sql[: match.start()].rstrip()
    from_part = sql[match.start():].lstrip()

    alias = _select_alias_from_expression(expression)
    if not alias:
        return f"{select_part}, {expression} {from_part}"

    rewritten_select = _replace_select_projection_by_alias(
        select_part,
        alias,
        expression,
    )
    if rewritten_select is not None:
        return f"{rewritten_select} {from_part}"

    return f"{select_part}, {expression} {from_part}"


def _sql_with_clusterid_color(sql_query: str) -> str:
    """Return SQL that exposes ClusterId through the scatter color role."""
    return _sql_insert_select_expression(
        str(sql_query),
        '"ClusterId" AS "color"',
    )


def _select_alias_from_expression(expression: str) -> str:
    """Extract the alias name from an expression ending with AS alias."""
    match = re.search(
        r'\bAS\s+(?:"([^"]+)"|\[([^\]]+)\]|`([^`]+)`|([A-Za-z_][A-Za-z0-9_]*))\s*$',
        str(expression).strip(),
        flags=re.IGNORECASE,
    )
    if match is None:
        return ""
    for group in match.groups():
        if group:
            return str(group)
    return ""


def _replace_select_projection_by_alias(
    select_part: str,
    alias: str,
    replacement_expression: str,
) -> str | None:
    """Replace the first top-level SELECT projection whose alias matches.

    This avoids regex-only parsing of comma-separated SELECT lists, so common
    expressions containing commas, functions, quoted names, or CAST(...) are
    handled without producing duplicate aliases.
    """
    text = str(select_part).rstrip()
    match = re.match(r"(?is)^\s*SELECT\s+", text)
    if match is None:
        return None

    prefix = text[: match.end()]
    body = text[match.end():]
    projections = _split_top_level_select_items(body)
    changed = False

    for index, projection in enumerate(projections):
        projection_alias = _projection_alias(projection)
        if projection_alias.lower() == str(alias).lower():
            projections[index] = replacement_expression
            changed = True
            break

    if not changed:
        return None

    return prefix + ", ".join(projections)


def _split_top_level_select_items(select_body: str) -> list[str]:
    """Split SELECT body on top-level commas only."""
    items: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    bracket_quote = False
    text = str(select_body)
    index = 0

    while index < len(text):
        char = text[index]

        if quote is not None:
            if bracket_quote:
                if char == "]":
                    quote = None
                    bracket_quote = False
            elif char == quote:
                if index + 1 < len(text) and text[index + 1] == quote:
                    index += 1
                else:
                    quote = None
            index += 1
            continue

        if char in {'"', "'", "`"}:
            quote = char
            bracket_quote = False
            index += 1
            continue

        if char == "[":
            quote = "]"
            bracket_quote = True
            index += 1
            continue

        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
        elif char == "," and depth == 0:
            items.append(text[start:index].strip())
            start = index + 1

        index += 1

    tail = text[start:].strip()
    if tail:
        items.append(tail)
    return items


def _projection_alias(projection: str) -> str:
    """Return the explicit AS alias for one SELECT projection, if present."""
    match = re.search(
        r'\bAS\s+(?:"([^"]+)"|\[([^\]]+)\]|`([^`]+)`|([A-Za-z_][A-Za-z0-9_]*))\s*$',
        str(projection).strip(),
        flags=re.IGNORECASE,
    )
    if match is None:
        return ""
    for group in match.groups():
        if group:
            return str(group)
    return ""


def _sql_with_cluster_filter(sql_query: str, cluster_id: int) -> str:
    """Return source SQL filtered to one ClusterId.

    Keeps ORDER BY/GROUP BY/HAVING/LIMIT/OFFSET clauses at the end by
    inserting the ClusterId predicate into the WHERE section.
    """
    sql = str(sql_query).strip().rstrip(";")
    cluster_clause = f'"ClusterId"={int(cluster_id)}'

    boundary = re.search(
        r"\b(GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT|OFFSET)\b",
        sql,
        flags=re.IGNORECASE,
    )

    if boundary is None:
        head = sql
        tail = ""
    else:
        head = sql[: boundary.start()].rstrip()
        tail = " " + sql[boundary.start():].lstrip()

    if re.search(r"\bWHERE\b", head, flags=re.IGNORECASE):
        return f"{head} AND {cluster_clause}{tail}"

    return f"{head} WHERE {cluster_clause}{tail}"


def _rc_cycle_color(cluster_id: int) -> str:
    colors = rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])
    return str(colors[(int(cluster_id) - 1) % len(colors)])


def _safe_numeric_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return finite feature matrix and retained-row mask."""
    if frame.empty:
        applogger.error("Selected series query returned no rows.", show_dialog=True, raise_error=True)

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        applogger.error(f"Missing selected feature columns: {', '.join(missing)}", show_dialog=True, raise_error=True)

    numeric = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    matrix = numeric.to_numpy(dtype=float)
    mask = np.asarray(np.isfinite(matrix).all(axis=1), dtype=bool).reshape(-1)

    if int(mask.sum()) < 2:
        applogger.error("At least two finite rows are required for clustering.", show_dialog=True, raise_error=True)

    return matrix[mask], mask


def _stable_cluster_ids(labels: np.ndarray) -> np.ndarray:
    """Convert arbitrary clustering labels to stable 1-based cluster ids."""
    raw = np.asarray(labels, dtype=int).reshape(-1)
    unique = sorted(int(value) for value in np.unique(raw))
    mapping = {value: index + 1 for index, value in enumerate(unique)}
    return np.asarray([mapping[int(value)] for value in raw], dtype=int)



def _run_with_timeout(
    timeout_seconds: float,
    func: Callable[..., tuple[np.ndarray, dict[str, Any]]],
    *args: Any,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run a clustering callable with an optional wall-time timeout.

    A timeout stops waiting for the preview/apply operation and returns control
    to the dialog. Python cannot safely kill a running SciPy call inside a
    thread, so the abandoned worker is detached with ``shutdown(wait=False)``.
    """
    timeout = float(timeout_seconds)
    if timeout <= 0.0:
        return func(*args, **kwargs)

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="datahub-cluster")
    future = executor.submit(func, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except FuturesTimeoutError as exc:
        future.cancel()
        message = (
            f"Clustering exceeded the configured timeout of {timeout:.1f} seconds. "
            "The preview/apply operation was stopped."
        )
        applogger.error(message, show_dialog=True, raise_error=True)
        raise TimeoutError(message) from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _initial_code_book(obs: np.ndarray, clusters: int) -> np.ndarray:
    """Create a deterministic code book for the raw vq operation."""
    matrix = np.asarray(obs, dtype=float)
    n_rows = matrix.shape[0]
    k = int(np.clip(int(clusters), 1, n_rows))
    indices = np.linspace(0, n_rows - 1, num=k, dtype=int)
    return np.asarray(matrix[indices, :], dtype=float)


def cluster_kmeans(
    features: np.ndarray,
    *,
    scipy_tool: str,
    clusters: int,
    use_whiten: bool,
    iterations: int,
    threshold: float,
    timeout_seconds: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Cluster observations with the selected ``scipy.cluster.vq`` function."""

    def _compute() -> tuple[np.ndarray, dict[str, Any]]:
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2:
            applogger.error("K-means/vector quantization expects a 2D observation matrix.", show_dialog=True, raise_error=True)

        n_rows = matrix.shape[0]
        if n_rows < 2:
            applogger.error("K-means/vector quantization requires at least two observations.", show_dialog=True, raise_error=True)

        k = int(np.clip(int(clusters), 1, n_rows))
        iterations_safe = max(1, int(iterations))
        threshold_safe = max(float(threshold), 0.0)
        tool = str(scipy_tool or TOOL_KMEANS2)
        whiten_applied = bool(use_whiten or tool == TOOL_WHITEN)
        obs = cast(np.ndarray, vq.whiten(matrix)) if whiten_applied else matrix

        centroids: np.ndarray
        distances: np.ndarray
        distortion: float | None = None

        if tool == TOOL_VQ:
            centroids = _initial_code_book(obs, k)
            codes, distances = vq.vq(obs, centroids, check_finite=True)
        elif tool == TOOL_KMEANS:
            raw_centroids, raw_distortion = vq.kmeans(
                obs,
                k,
                iter=iterations_safe,
                thresh=threshold_safe,
                check_finite=True,
            )
            centroids = np.asarray(raw_centroids, dtype=float)
            distortion_values = np.asarray(raw_distortion, dtype=float).reshape(-1)
            distortion = float(distortion_values[0]) if distortion_values.size else 0.0
            codes, distances = vq.vq(obs, centroids, check_finite=True)
        else:
            # TOOL_KMEANS2 and TOOL_WHITEN both classify with kmeans2. TOOL_WHITEN
            # explicitly makes whitening part of the selected SciPy operation.
            centroids, labels = vq.kmeans2(
                obs,
                k,
                iter=iterations_safe,
                thresh=threshold_safe,
                minit="++",
                missing="warn",
                check_finite=True,
            )
            codes = np.asarray(labels, dtype=int)
            _codes_for_distance, distances = vq.vq(obs, centroids, check_finite=True)

        cluster_ids = _stable_cluster_ids(np.asarray(codes, dtype=int))
        metadata: dict[str, Any] = {
            "scipy_tool": tool,
            "centroids": np.asarray(centroids, dtype=float).tolist(),
            "distortion": None if distortion is None else float(distortion),
            "mean_distance": float(np.mean(distances)) if distances.size else 0.0,
            "max_distance": float(np.max(distances)) if distances.size else 0.0,
            "clusters_requested": int(clusters),
            "clusters_found": int(np.unique(cluster_ids).size),
            "whiten": whiten_applied,
            "iterations": iterations_safe,
            "threshold": threshold_safe,
        }
        return cluster_ids, metadata

    return _run_with_timeout(timeout_seconds, _compute)


def cluster_hierarchical(
    features: np.ndarray,
    *,
    scipy_tool: str,
    clusters: int,
    linkage_method: str,
    metric: str,
    criterion: str,
    distance_threshold: float,
    timeout_seconds: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Cluster observations with the selected ``scipy.cluster.hierarchy`` function."""

    def _compute() -> tuple[np.ndarray, dict[str, Any]]:
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2:
            applogger.error("Hierarchical clustering expects a 2D observation matrix.", show_dialog=True, raise_error=True)

        n_rows = matrix.shape[0]
        if n_rows < 2:
            applogger.error("Hierarchical clustering requires at least two observations.", show_dialog=True, raise_error=True)

        tool = str(scipy_tool or TOOL_FCLUSTER)
        method = str(linkage_method or "ward")
        metric_name = str(metric or "euclidean")
        criterion_name = str(criterion or "maxclust")

        if method == "ward":
            metric_name = "euclidean"

        if criterion_name == "distance":
            threshold = float(distance_threshold)
            if threshold <= 0.0:
                threshold = 1.0
            t_value: float | int = threshold
        elif criterion_name == "inconsistent":
            t_value = max(float(distance_threshold), 1.0)
        else:
            t_value = int(np.clip(int(clusters), 1, n_rows))

        linkage_matrix: np.ndarray | None = None
        leader_nodes: list[int] = []
        leader_cluster_ids: list[int] = []

        if tool == TOOL_FCLUSTERDATA:
            labels = hierarchy.fclusterdata(
                matrix,
                t=t_value,
                criterion=criterion_name,
                metric=metric_name,
                depth=2,
                method=method,
            )
        else:
            linkage = np.asarray(
                hierarchy.linkage(
                    matrix,
                    method=method,
                    metric=metric_name,
                    optimal_ordering=True,
                ),
                dtype=float,
            )
            linkage_matrix = linkage
            if criterion_name == "distance" and float(distance_threshold) <= 0.0:
                t_value = float(np.median(linkage[:, 2])) if linkage.size else 1.0
            labels = hierarchy.fcluster(linkage, t=t_value, criterion=criterion_name)
            if tool == TOOL_LEADERS:
                leaders, leader_ids = hierarchy.leaders(linkage, labels)
                leader_nodes = [int(value) for value in np.asarray(leaders).reshape(-1)]
                leader_cluster_ids = [int(value) for value in np.asarray(leader_ids).reshape(-1)]

        cluster_ids = _stable_cluster_ids(np.asarray(labels, dtype=int))
        metadata: dict[str, Any] = {
            "scipy_tool": tool,
            "clusters_requested": int(clusters),
            "clusters_found": int(np.unique(cluster_ids).size),
            "linkage_method": method,
            "metric": metric_name,
            "criterion": criterion_name,
            "distance_threshold": float(distance_threshold),
            "cut_value": float(t_value),
            "linkage_rows": 0 if linkage_matrix is None else int(linkage_matrix.shape[0]),
            "leader_nodes": leader_nodes,
            "leader_cluster_ids": leader_cluster_ids,
        }
        return cluster_ids, metadata

    return _run_with_timeout(timeout_seconds, _compute)


def cluster_sklearn(
    features: np.ndarray,
    *,
    sklearn_tool: str,
    clusters: int,
    linkage_method: str,
    metric: str,
    eps: float,
    min_samples: int,
    timeout_seconds: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Cluster observations with scikit-learn algorithms when available."""

    def _compute() -> tuple[np.ndarray, dict[str, Any]]:
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2:
            applogger.error("scikit-learn clustering expects a 2D observation matrix.", show_dialog=True, raise_error=True)
        n_rows = matrix.shape[0]
        if n_rows < 2:
            applogger.error("scikit-learn clustering requires at least two observations.", show_dialog=True, raise_error=True)

        try:
            from sklearn.cluster import (
                AgglomerativeClustering,
                Birch,
                BisectingKMeans,
                DBSCAN,
                KMeans,
                MeanShift,
                MiniBatchKMeans,
                OPTICS,
                SpectralClustering,
            )
            from sklearn.mixture import GaussianMixture
        except Exception as exc:
            applogger.error(
                "scikit-learn clustering requires the optional dependency 'scikit-learn'. "
                "Install it in the current Python environment.",
                show_dialog=True,
                raise_error=True,
            )
            raise exc

        tool = str(sklearn_tool or SKLEARN_KMEANS)
        k = int(np.clip(int(clusters), 1, n_rows))
        metric_name = str(metric or "euclidean")
        linkage = str(linkage_method or "ward")
        eps_value = float(eps) if float(eps) > 0.0 else 0.5
        min_samples_value = max(1, int(min_samples))
        random_state = 0

        if tool == SKLEARN_KMEANS:
            model = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
            labels = model.fit_predict(matrix)
            extra = {"inertia": float(model.inertia_)}
        elif tool == SKLEARN_MINIBATCH_KMEANS:
            model = MiniBatchKMeans(n_clusters=k, n_init="auto", random_state=random_state)
            labels = model.fit_predict(matrix)
            extra = {"inertia": float(model.inertia_)}
        elif tool == SKLEARN_BISECTING_KMEANS:
            model = BisectingKMeans(n_clusters=k, random_state=random_state)
            labels = model.fit_predict(matrix)
            extra = {"inertia": float(model.inertia_)}
        elif tool == SKLEARN_AGGLOMERATIVE:
            if linkage == "ward":
                metric_name = "euclidean"
            model = AgglomerativeClustering(n_clusters=k, linkage=cast(Any, linkage), metric=metric_name)
            labels = model.fit_predict(matrix)
            extra = {}
        elif tool == SKLEARN_DBSCAN:
            model = DBSCAN(eps=eps_value, min_samples=min_samples_value, metric=metric_name)
            labels = model.fit_predict(matrix)
            extra = {"eps": eps_value, "min_samples": min_samples_value}
        elif tool == SKLEARN_OPTICS:
            model = OPTICS(min_samples=min_samples_value, metric=metric_name)
            labels = model.fit_predict(matrix)
            extra = {"min_samples": min_samples_value}
        elif tool == SKLEARN_BIRCH:
            model = Birch(n_clusters=k)
            labels = model.fit_predict(matrix)
            extra = {}
        elif tool == SKLEARN_MEANSHIFT:
            model = MeanShift()
            labels = model.fit_predict(matrix)
            extra = {}
        elif tool == SKLEARN_SPECTRAL:
            model = SpectralClustering(n_clusters=k, assign_labels="kmeans", random_state=random_state)
            labels = model.fit_predict(matrix)
            extra = {}
        elif tool == SKLEARN_GAUSSIAN_MIXTURE:
            model = GaussianMixture(n_components=k, random_state=random_state)
            labels = model.fit_predict(matrix)
            extra = {"converged": bool(model.converged_), "lower_bound": float(model.lower_bound_)}
        else:
            applogger.error(f"Unsupported scikit-learn clustering algorithm: {tool}", show_dialog=True, raise_error=True)

        raw = np.asarray(labels, dtype=int)
        # DBSCAN/OPTICS noise is -1. Keep it as a stable positive cluster id
        # rather than NULL so renderers can color it consistently.
        cluster_ids = _stable_cluster_ids(raw)
        metadata: dict[str, Any] = {
            "scipy_tool": tool,
            "sklearn_tool": tool,
            "clusters_requested": int(clusters),
            "clusters_found": int(np.unique(cluster_ids).size),
            "metric": metric_name,
            "linkage_method": linkage,
            **extra,
        }
        return cluster_ids, metadata

    return _run_with_timeout(timeout_seconds, _compute)


class SeriesClusterDialog(SeriesOperationDialogBase):
    """Dialog that clusters selected chart-series data and colors/splits clusters."""
    Name: str  = "Clustering"
    Description = "Group similar data"

    # Clustering reads an observation matrix, not a function of x, so order and
    # repeated x carry no meaning here and must not be rejected: two samples
    # with the same x are an ordinary pair of observations. Only the universal
    # empty/length/non-finite checks apply.
    INPUT_MINIMUM_POINTS = 2

    Icon = """
    <circle cx="7" cy="8" r="2"/>
    <circle cx="16.5" cy="7" r="2"/>
    <circle cx="10" cy="16" r="2"/>
    <circle cx="18" cy="15.5" r="2"/>
    <path d="M8.8 9.5l5.9 4.8"/>
    <path d="M15 8.3l-3.6 6.1"/>
    <path d="M12 16h4"/>
    """
    def __init__(
        self,
        *,
        repo: SqliteRepo,
        figure_id: int,
        applied_callback: Callable[[], None] | None = None,
        table: str | None = None,
        parent: QWidget | None = None,
    ) -> None:

        if repo is None:
            applogger.error("SeriesClusterDialog requires a repository instance.", show_dialog=True, raise_error=True)

        self._repo: Any = repo
        self._figure_id = int(figure_id)
        self._applied_callback = applied_callback
        self._initial_table = table
        self._last_results: list[ClusterResult] = []
        self._field_rows: dict[str, tuple[QWidget, QWidget]] = {}
        self._last_report_html = ""

        # source table -> did a ClusterId column exist before the preview.
        self._cluster_snapshots: dict[str, bool] = {}
        # series id -> its SQL before the preview rewrote it.
        self._series_sql_snapshots: dict[int, str] = {}

        super().__init__(
            repo=repo,
            figure_id=figure_id,
            title="Series Clustering",
            parent=parent,
            width=760,
            height=680,
        )

        self.model_combo.setVisible(False)
        self._refresh_visibility()
        self.refresh_results()

    def create_axis_series_selector(self) -> AxisSeriesSelector:
        return AxisSeriesSelector(self._repo, self._figure_id, self)

    def init_operation_widgets(self) -> None:
        self._create_controls()

    def build_model_selector(self) -> QWidget:
        panel = create_card_widget(self, "clusterModelCard")
        layout = QVBoxLayout(panel)
        stdSizeAndlayout(layout)

        self.method_combo = QComboBox(self)
        self.method_combo.addItems([CLUSTER_KMEANS, CLUSTER_HIERARCHICAL, CLUSTER_SKLEARN])
        self.method_combo.setToolTip(_("Choose the clustering algorithm family."))

        self.scipy_tool_combo = QComboBox(self)
        self.scipy_tool_combo.setToolTip(_("Choose the exact scipy.cluster function/workflow to use."))

        self.feature_combo = QComboBox(self)
        self.feature_combo.addItems([FEATURE_XY, FEATURE_Y_ONLY, FEATURE_XYZ, FEATURE_ALL_NUMERIC])
        self.feature_combo.setToolTip(_("Choose which numeric columns are used as clustering features."))

        self.render_mode_combo = QComboBox(self)
        self.render_mode_combo.addItems([RENDER_COLORED_SERIES, RENDER_SEPARATE_SERIES])
        self.render_mode_combo.setToolTip(
            _("Choose whether clusters are rendered as one point-colored series or as separate series.")
        )

        self._doc_link = QLabel(self)
        self._doc_link.setOpenExternalLinks(True)
        self._doc_link.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)

        form_widget = QWidget(panel)
        form = QFormLayout(form_widget)
        stdSizeAndlayout(form)
        form.addRow(_("Family:"), self.method_combo)
        form.addRow(_("Algorithm:"), self.scipy_tool_combo)
        form.addRow(_("Features:"), self.feature_combo)
        form.addRow(_("Render as:"), self.render_mode_combo)
        form.addRow(_("Docs:"), self._doc_link)

        layout.addWidget(form_widget)
        return panel

    def build_parameter_selector(self) -> QWidget:
        settings_widget = create_card_widget(self, "clusterParamsCard")
        self.form = QFormLayout(settings_widget)
        stdSizeAndlayout(self.form)
        self._add_parameter_rows()

        scroll = QScrollArea(self)
        stdSizeAndlayout(scroll)
        scroll.setWidgetResizable(True)
        scroll.setWidget(settings_widget)
        return scroll

    def connect_operation_signals(self) -> None:
        self.series_selector.selection_changed.connect(lambda *_args: self.refresh_results())
        self.series_selector.axis_changed.connect(lambda *_args: self.refresh_results())

        self.method_combo.currentIndexChanged.connect(self._refresh_tools)
        self.method_combo.currentIndexChanged.connect(self._refresh_visibility)
        self.method_combo.currentIndexChanged.connect(self.refresh_results)
        self.scipy_tool_combo.currentIndexChanged.connect(self._refresh_visibility)
        self.scipy_tool_combo.currentIndexChanged.connect(self.refresh_results)
        self.feature_combo.currentIndexChanged.connect(self.refresh_results)
        self.render_mode_combo.currentIndexChanged.connect(self.refresh_results)

        self.cluster_count_spin.valueChanged.connect(self.refresh_results)
        self.kmeans_iter_spin.valueChanged.connect(self.refresh_results)
        self.kmeans_thresh_spin.valueChanged.connect(self.refresh_results)
        self.whiten_check.stateChanged.connect(self.refresh_results)
        self.max_runtime_spin.valueChanged.connect(self.refresh_results)
        self.linkage_method_combo.currentIndexChanged.connect(self.refresh_results)
        self.metric_combo.currentIndexChanged.connect(self.refresh_results)
        self.hierarchy_criterion_combo.currentIndexChanged.connect(self.refresh_results)
        self.distance_threshold_spin.valueChanged.connect(self.refresh_results)
        self.sklearn_eps_spin.valueChanged.connect(self.refresh_results)
        self.sklearn_min_samples_spin.valueChanged.connect(self.refresh_results)

        self._doc_link.linkActivated.connect(self._open_description)

    @staticmethod
    def _spin(minimum: int, maximum: int, value: int) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        return widget

    @staticmethod
    def _double_spin(minimum: float, maximum: float, value: float, decimals: int) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        widget.setDecimals(decimals)
        widget.setSingleStep(10 ** -min(decimals, 3))
        return widget

    def _create_controls(self) -> None:
        self.cluster_count_spin = self._spin(1, 10_000, 4)
        self.cluster_count_spin.setToolTip(_("Number of clusters to request."))

        self.whiten_check = QCheckBox()
        self.whiten_check.setChecked(True)
        self.whiten_check.setToolTip(_("Normalize features by standard deviation before k-means."))

        self.kmeans_iter_spin = self._spin(1, 10_000, 50)
        self.kmeans_iter_spin.setToolTip(_("Maximum k-means iterations."))

        self.kmeans_thresh_spin = self._double_spin(0.0, 1.0, 1e-5, 8)
        self.kmeans_thresh_spin.setToolTip(_("K-means convergence threshold."))

        self.max_runtime_spin = self._double_spin(0.0, 3600.0, 15.0, 1)
        self.max_runtime_spin.setSpecialValueText(_("no timeout"))
        self.max_runtime_spin.setToolTip(
            _("Maximum seconds to wait for clustering. Use 0 for no timeout.")
        )


        self.linkage_method_combo = QComboBox()
        self.linkage_method_combo.addItems(list(HIERARCHY_METHODS))
        self.linkage_method_combo.setCurrentText("ward")

        self.metric_combo = QComboBox()
        self.metric_combo.addItems(list(HIERARCHY_METRICS))
        self.metric_combo.setCurrentText("euclidean")

        self.hierarchy_criterion_combo = QComboBox()
        self.hierarchy_criterion_combo.addItems(list(HIERARCHY_CRITERIA))
        self.hierarchy_criterion_combo.setCurrentText("maxclust")

        self.distance_threshold_spin = self._double_spin(0.0, 1e18, 0.0, 6)
        self.distance_threshold_spin.setSpecialValueText(_("auto"))
        self.distance_threshold_spin.setToolTip(
            _("Cut threshold for distance/inconsistent criteria. Leave at auto for median linkage distance.")
        )

        self.sklearn_eps_spin = self._double_spin(0.0, 1e18, 0.5, 6)
        self.sklearn_eps_spin.setSpecialValueText(_("auto"))
        self.sklearn_eps_spin.setToolTip(_("Neighborhood radius for DBSCAN. 0 uses the default 0.5."))

        self.sklearn_min_samples_spin = self._spin(1, 1_000_000, 5)
        self.sklearn_min_samples_spin.setToolTip(_("Minimum samples for DBSCAN/OPTICS core points."))


    def _add_parameter_rows(self) -> None:
        rows: tuple[tuple[str, str, QWidget], ...] = (
            ("clusters", "Clusters:", self.cluster_count_spin),
            ("whiten", "Whiten features:", self.whiten_check),
            ("kmeans_iter", "K-means iterations:", self.kmeans_iter_spin),
            ("kmeans_thresh", "K-means threshold:", self.kmeans_thresh_spin),
            ("max_runtime", "Max runtime:", self.max_runtime_spin),
            ("linkage_method", "Linkage method:", self.linkage_method_combo),
            ("metric", "Distance metric:", self.metric_combo),
            ("criterion", "Cut criterion:", self.hierarchy_criterion_combo),
            ("distance_threshold", "Cut threshold:", self.distance_threshold_spin),
            ("sklearn_eps", "Neighborhood radius:", self.sklearn_eps_spin),
            ("sklearn_min_samples", "Min samples:", self.sklearn_min_samples_spin),
        )
        for key, label, widget in rows:
            self._add_row(key, label, widget)

    def _add_row(self, key: str, label: str, widget: QWidget) -> None:
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        stdSizeAndlayout(row_layout)
        row_layout.addWidget(widget)
        row_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        label_widget = QLabel(label)
        self.form.addRow(label_widget, row_widget)
        self._field_rows[key] = (label_widget, row_widget)

    def _refresh_tools(self) -> None:
        """Refresh the SciPy function list for the selected clustering family."""
        current = self.scipy_tool_combo.currentText()
        method = self.method_combo.currentText()
        if method == CLUSTER_KMEANS:
            tools = KMEANS_TOOLS
        elif method == CLUSTER_HIERARCHICAL:
            tools = HIERARCHY_TOOLS
        else:
            tools = SKLEARN_TOOLS
        self.scipy_tool_combo.blockSignals(True)
        self.scipy_tool_combo.clear()
        self.scipy_tool_combo.addItems(list(tools))
        old_index = self.scipy_tool_combo.findText(current)
        self.scipy_tool_combo.setCurrentIndex(old_index if old_index >= 0 else 0)
        self.scipy_tool_combo.blockSignals(False)
        self._update_description_link()

    def _refresh_visibility(self) -> None:
        if self.scipy_tool_combo.count() == 0:
            self._refresh_tools()

        method = self.method_combo.currentText()
        tool = self.scipy_tool_combo.currentText()
        visible = {"clusters", "max_runtime"}

        if method == CLUSTER_KMEANS:
            visible.update({"kmeans_iter", "kmeans_thresh"})
            if tool in {TOOL_WHITEN, TOOL_KMEANS, TOOL_KMEANS2}:
                visible.add("whiten")
        elif method == CLUSTER_HIERARCHICAL:
            visible.update({"linkage_method", "metric", "criterion", "distance_threshold"})
        else:
            if tool in {SKLEARN_AGGLOMERATIVE}:
                visible.update({"linkage_method", "metric"})
            if tool in {SKLEARN_DBSCAN}:
                visible.update({"metric", "sklearn_eps", "sklearn_min_samples"})
            if tool in {SKLEARN_OPTICS}:
                visible.update({"metric", "sklearn_min_samples"})

        for key, widgets in self._field_rows.items():
            is_visible = key in visible
            widgets[0].setVisible(is_visible)
            widgets[1].setVisible(is_visible)

        self._update_description_link()

    def _update_description_link(self) -> None:
        tool = self.scipy_tool_combo.currentText() if hasattr(self, "scipy_tool_combo") else ""
        url_by_tool = {
            TOOL_WHITEN: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.whiten.html",
            TOOL_VQ: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.vq.html",
            TOOL_KMEANS: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans.html",
            TOOL_KMEANS2: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans2.html",
            TOOL_FCLUSTER: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html",
            TOOL_FCLUSTERDATA: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fclusterdata.html",
            TOOL_LEADERS: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.leaders.html",
            SKLEARN_KMEANS: "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html",
            SKLEARN_MINIBATCH_KMEANS: "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html",
            SKLEARN_BISECTING_KMEANS: "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.BisectingKMeans.html",
            SKLEARN_AGGLOMERATIVE: "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html",
            SKLEARN_DBSCAN: "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html",
            SKLEARN_OPTICS: "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html",
            SKLEARN_BIRCH: "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html",
            SKLEARN_MEANSHIFT: "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html",
            SKLEARN_SPECTRAL: "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html",
            SKLEARN_GAUSSIAN_MIXTURE: "https://scikit-learn.org/stable/modules/mixture.html",
        }
        title = tool or "SciPy clustering"
        url = url_by_tool.get(tool, "https://scikit-learn.org/stable/modules/clustering.html" if str(tool).startswith("sklearn.") else "https://docs.scipy.org/doc/scipy/reference/cluster.html")
        self.set_doc_link(title, url)

    def _open_description(self, _link: str = "") -> None:
        tool = self.scipy_tool_combo.currentText()
        url_by_tool = {
            TOOL_WHITEN: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.whiten.html",
            TOOL_VQ: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.vq.html",
            TOOL_KMEANS: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans.html",
            TOOL_KMEANS2: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans2.html",
            TOOL_FCLUSTER: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html",
            TOOL_FCLUSTERDATA: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fclusterdata.html",
            TOOL_LEADERS: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.leaders.html",
            SKLEARN_KMEANS: "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html",
            SKLEARN_MINIBATCH_KMEANS: "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html",
            SKLEARN_BISECTING_KMEANS: "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.BisectingKMeans.html",
            SKLEARN_AGGLOMERATIVE: "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html",
            SKLEARN_DBSCAN: "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html",
            SKLEARN_OPTICS: "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html",
            SKLEARN_BIRCH: "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html",
            SKLEARN_MEANSHIFT: "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html",
            SKLEARN_SPECTRAL: "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html",
            SKLEARN_GAUSSIAN_MIXTURE: "https://scikit-learn.org/stable/modules/mixture.html",
        }
        title = tool or "SciPy clustering"
        url = url_by_tool.get(tool, "https://scikit-learn.org/stable/modules/clustering.html" if str(tool).startswith("sklearn.") else "https://docs.scipy.org/doc/scipy/reference/cluster.html")
        try:
            webbrowser.open(url)
        except Exception:
            show_message(self, "series.open_docs_failed", title=title, url=url)

    def _params(self) -> dict[str, Any]:
        return {
            "method": self.method_combo.currentText(),
            "scipy_tool": self.scipy_tool_combo.currentText(),
            "feature_mode": self.feature_combo.currentText(),
            "render_mode": self.render_mode_combo.currentText(),
            "clusters": self.cluster_count_spin.value(),
            "whiten": self.whiten_check.isChecked(),
            "kmeans_iter": self.kmeans_iter_spin.value(),
            "kmeans_thresh": self.kmeans_thresh_spin.value(),
            "linkage_method": self.linkage_method_combo.currentText(),
            "metric": self.metric_combo.currentText(),
            "criterion": self.hierarchy_criterion_combo.currentText(),
            "distance_threshold": self.distance_threshold_spin.value(),
            "max_runtime_seconds": self.max_runtime_spin.value(),
            "sklearn_eps": self.sklearn_eps_spin.value(),
            "sklearn_min_samples": self.sklearn_min_samples_spin.value(),
        }

    def _current_axis_name(self) -> str:
        return self.series_selector.selected_axis_name()

    def _series_display_name(self, row: Any) -> str:
        return str(row["name"])

    def _series_choice_from_row(self, row: Any) -> ClusterSeriesChoice:
        name = str(row["name"])
        sql_query = str(row["sql_query"])
        if not sql_query:
            applogger.error("Selected series has no SQL query.", show_dialog=True, raise_error=True)

        source_table = _source_table_from_sql(sql_query)
        source_x_column = _source_column_for_alias(sql_query, "x")
        frame = self._repo.query_df(sql_query)
        if frame.empty:
            applogger.error("Selected series query returned no rows.", show_dialog=True, raise_error=True)

        roles = parse_roles(row["roles"])
        columns = [str(column) for column in frame.columns]
        numeric = [str(column) for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]

        x_col = str(roles.get("x") or "")
        y_col = str(roles.get("y") or "")
        z_col = str(roles.get("z") or "")

        if x_col not in columns:
            x_col = numeric[0] if numeric else columns[0]
        if y_col not in columns:
            y_col = numeric[1] if len(numeric) > 1 else x_col
        if z_col and z_col not in columns:
            z_col = ""

        return ClusterSeriesChoice(
            name=name,
            frame=frame.copy(),
            x_col=x_col,
            y_col=y_col,
            z_col=z_col or None,
            roles=roles,
            source_table=source_table,
            source_x_column=source_x_column,
            source_sql_query=sql_query,
            source=row,
        )

    def _feature_columns(self, series: ClusterSeriesChoice) -> list[str]:
        mode = self.feature_combo.currentText()
        frame = series.frame

        if mode == FEATURE_Y_ONLY:
            return [series.y_col]

        if mode == FEATURE_XYZ:
            z_col = series.z_col or ""
            if not z_col:
                applogger.error(
                    "Selected series has no Z role/column for X, Y and Z clustering.",
                    show_dialog=True,
                    raise_error=True,
                )
            return [series.x_col, series.y_col, z_col]

        if mode == FEATURE_ALL_NUMERIC:
            excluded = {"ClusterId"}
            numeric = [
                str(column)
                for column in frame.columns
                if pd.api.types.is_numeric_dtype(frame[column]) and str(column) not in excluded
            ]
            if not numeric:
                applogger.error("No numeric columns are available for clustering.", show_dialog=True, raise_error=True)
            return numeric

        return [series.x_col, series.y_col]

    def compute_results(self) -> list[ClusterResult]:
        axis_name = self._current_axis_name()
        params = self._params()
        selected_rows = self.selected_series()
        if not selected_rows:
            return []

        results: list[ClusterResult] = []
        errors: list[str] = []

        for row in selected_rows:
            try:
                full_result = self._cluster_one_row(row, axis_name, params)
                if params["render_mode"] == RENDER_SEPARATE_SERIES:
                    results.extend(self._split_result_by_cluster(full_result))
                else:
                    results.append(full_result)
            except Exception as exc:
                errors.append(f"{self._series_display_name(row)}: {exc}")

        if errors and not results:
            applogger.error("\n".join(errors), show_dialog=True, raise_error=True)

        if errors:
            show_message(
                self,
                "series.some_failed",
                title=self.operation_label,
                errors="\n".join(errors),
            )

        return results

    def _cluster_one_row(self, row: Any, axis_name: str, params: Mapping[str, Any]) -> ClusterResult:
        series = self._series_choice_from_row(row)
        feature_columns = self._feature_columns(series)
        features, finite_mask = _safe_numeric_matrix(series.frame, feature_columns)

        method = str(params["method"])
        if method == CLUSTER_KMEANS:
            cluster_ids, metadata = cluster_kmeans(
                features,
                scipy_tool=str(params["scipy_tool"]),
                clusters=int(params["clusters"]),
                use_whiten=bool(params["whiten"]),
                iterations=int(params["kmeans_iter"]),
                threshold=float(params["kmeans_thresh"]),
                timeout_seconds=float(params.get("max_runtime_seconds", 0.0)),
            )
        elif method == CLUSTER_HIERARCHICAL:
            cluster_ids, metadata = cluster_hierarchical(
                features,
                scipy_tool=str(params["scipy_tool"]),
                clusters=int(params["clusters"]),
                linkage_method=str(params["linkage_method"]),
                metric=str(params["metric"]),
                criterion=str(params["criterion"]),
                distance_threshold=float(params["distance_threshold"]),
                timeout_seconds=float(params.get("max_runtime_seconds", 0.0)),
            )
        else:
            cluster_ids, metadata = cluster_sklearn(
                features,
                sklearn_tool=str(params["scipy_tool"]),
                clusters=int(params["clusters"]),
                linkage_method=str(params["linkage_method"]),
                metric=str(params["metric"]),
                eps=float(params.get("sklearn_eps", 0.5)),
                min_samples=int(params.get("sklearn_min_samples", 5)),
                timeout_seconds=float(params.get("max_runtime_seconds", 0.0)),
            )

        output = series.frame.copy()
        output["ClusterId"] = pd.Series([pd.NA] * len(output), dtype="Int64")
        output.loc[finite_mask, "ClusterId"] = cluster_ids

        metadata.update(
            {
                **dict(params),
                "figure_id": self._figure_id,
                "axis_name": axis_name,
                "source_series_id": self._source_series_id(series),
                "source_series_name": series.name,
                "source_table": series.source_table,
                "source_x_column": series.source_x_column,
                "source_sql_query": series.source_sql_query,
                "feature_columns": feature_columns,
                "finite_rows": int(finite_mask.sum()),
                "total_rows": int(len(output)),
                "roles": series.roles,
                "base_result_name": f"{series.name} - clusters",
                "is_split_series": False,
            }
        )

        return ClusterResult(
            source_name=series.name,
            result_name=f"{series.name} - clusters",
            method=method,
            frame=output,
            x_col=series.x_col,
            y_col=series.y_col,
            z_col=series.z_col,
            feature_columns=feature_columns,
            metadata=metadata,
        )

    def _split_result_by_cluster(self, result: ClusterResult) -> list[ClusterResult]:
        ids = pd.to_numeric(result.frame["ClusterId"], errors="coerce")
        cluster_ids = sorted(int(value) for value in ids.dropna().unique())
        split_results: list[ClusterResult] = []

        for cluster_id in cluster_ids:
            metadata = dict(result.metadata)
            metadata.update(
                {
                    "cluster_id": int(cluster_id),
                    "is_split_series": True,
                    "render_mode": RENDER_SEPARATE_SERIES,
                }
            )
            split_results.append(
                ClusterResult(
                    source_name=result.source_name,
                    result_name=f"{result.source_name} - Cluster {cluster_id}",
                    method=result.method,
                    frame=result.frame,
                    x_col=result.x_col,
                    y_col=result.y_col,
                    z_col=result.z_col,
                    feature_columns=list(result.feature_columns),
                    metadata=metadata,
                )
            )

        return split_results

    def _snapshot_cluster_state(self, source_table: str) -> None:
        """Move the existing ClusterId aside before a preview overwrites it.

        Why an explicit column snapshot rather than relying on the preview
        SAVEPOINT: clustering does not create removable preview artifacts, it
        overwrites a column in the user's own table.  A savepoint should cover
        that too, but any repository helper that commits - directly or through
        pandas ``to_sql`` - silently ends the transaction and takes the
        savepoint with it, and the user then finds their data changed after
        pressing Close.  Renaming the column is O(1) metadata, cannot fail on
        size, and restores the original bytes rather than a re-computed
        approximation.
        """
        if source_table in self._cluster_snapshots:
            return

        try:
            had_column = self._repo.snapshot_column(
                source_table, CLUSTER_COLUMN, CLUSTER_BACKUP_COLUMN
            )
        except Exception:
            applogger.exception(
                "Failed to snapshot %s.%s before preview", source_table, CLUSTER_COLUMN
            )
            return

        self._cluster_snapshots[source_table] = had_column

    def _snapshot_series_sql(self, series_id: int) -> None:
        """Remember a series' SQL before the preview rewrites it."""
        if series_id in self._series_sql_snapshots:
            return
        try:
            self._series_sql_snapshots[series_id] = str(
                self._repo.get_series_sql_query(series_id) or ""
            )
        except Exception:
            applogger.exception("Failed to snapshot SQL for series id=%s", series_id)

    def restore_operation_snapshots(self) -> bool:
        """Undo everything a cluster preview wrote.  Returns True if it did."""
        restored = False

        for source_table, had_column in list(self._cluster_snapshots.items()):
            try:
                if had_column:
                    self._repo.restore_column_snapshot(
                        source_table, CLUSTER_COLUMN, CLUSTER_BACKUP_COLUMN
                    )
                elif self._repo.has_column(source_table, CLUSTER_COLUMN):
                    # There was no ClusterId before the preview, so removing the
                    # one the preview added is the correct restore.
                    self._repo.delete_table_column(source_table, CLUSTER_COLUMN)
                restored = True
            except Exception:
                applogger.exception(
                    "Failed to restore %s.%s after preview", source_table, CLUSTER_COLUMN
                )
        self._cluster_snapshots.clear()

        for series_id, sql_query in list(self._series_sql_snapshots.items()):
            try:
                self._repo.update_series_sql_query(int(series_id), sql_query)
                restored = True
            except Exception:
                applogger.exception("Failed to restore SQL for series id=%s", series_id)
        self._series_sql_snapshots.clear()

        return restored

    def discard_operation_snapshots(self) -> None:
        """Drop the snapshots after Apply has made the changes permanent."""
        for source_table in list(self._cluster_snapshots):
            try:
                self._repo.discard_column_snapshot(source_table, CLUSTER_BACKUP_COLUMN)
            except Exception:
                applogger.exception(
                    "Failed to discard the %s snapshot on %s",
                    CLUSTER_BACKUP_COLUMN,
                    source_table,
                )
        self._cluster_snapshots.clear()
        self._series_sql_snapshots.clear()

    def _write_cluster_ids_to_source_table(self, result: ClusterResult) -> None:
        source_table = str(result.metadata["source_table"])
        source_x_column = str(result.metadata["source_x_column"])
        self._snapshot_cluster_state(source_table)
        self._repo.ensure_cluster_column(source_table)

        cluster_values = pd.to_numeric(result.frame["ClusterId"], errors="coerce")
        x_values = result.frame[result.x_col]
        self._repo.clear_cluster_column(source_table)
        self._repo.set_ClusterId(source_table,source_x_column,x_values,cluster_values)


    def _colored_series_sql_query(self, table_name: str, result: ClusterResult) -> str:
        del table_name
        return _sql_with_clusterid_color(str(result.metadata["source_sql_query"]))

    def result_series_spec(self, axis_id: int, table_name: str, result: ClusterResult) -> ResultSeriesSpec:
        del axis_id

        roles = self._base_roles(result)
        roles.update({"ClusterId": "ClusterId", "cluster": "ClusterId"})

        cluster_id = result.metadata.get("cluster_id")
        is_split = bool(result.metadata.get("is_split_series", False))

        if is_split and cluster_id is not None:
            sql_query = _sql_with_cluster_filter(
                str(result.metadata["source_sql_query"]),
                int(cluster_id),
            )
            color = _rc_cycle_color(int(cluster_id))
            style = {
                "generated_clustering": True,
                "clustering_dialog": "series_clustering",
                "source_name": result.source_name,
                "source_series_id": result.metadata.get("source_series_id"),
                "method": result.method,
                "features": result.feature_columns,
                "cluster_id": int(cluster_id),
                "color": color,
                "marker": "o",
                "linestyle": "",
                "use_point_colors": False,
                "render_mode": RENDER_SEPARATE_SERIES,
            }
            return ResultSeriesSpec(
                name=result.result_name,
                sql_query=sql_query,
                roles=roles,
                style=style,
            )

        sql_query = self._colored_series_sql_query(table_name, result)
        roles.update({"color": "color"})
        return ResultSeriesSpec(
            name=result.result_name,
            sql_query=sql_query,
            roles=roles,
            style={
                "generated_clustering": True,
                "clustering_dialog": "series_clustering",
                "source_name": result.source_name,
                "source_series_id": result.metadata.get("source_series_id"),
                "method": result.method,
                "features": result.feature_columns,
                "marker": "o",
                "linestyle": "",
                "use_point_colors": True,
                "color_role": "color",
                "render_mode": RENDER_COLORED_SERIES,
            },
        )


    def _order_columns(self, result: ClusterResult) -> list[str]:
        order_cols = [result.x_col]
        if result.y_col and result.y_col != result.x_col:
            order_cols.append(result.y_col)
        if result.z_col and result.z_col not in order_cols:
            order_cols.append(result.z_col)
        return order_cols

    def _base_roles(self, result: ClusterResult) -> dict[str, Any]:
        roles: dict[str, Any] = dict(result.metadata.get("roles", {}))
        roles.update({"x": result.x_col, "y": result.y_col})
        if result.z_col:
            roles["z"] = result.z_col
        return roles

    @property
    def generated_style_filter(self) -> Mapping[str, Any]:
        return {"generated_clustering": True, "clustering_dialog": "series_clustering"}

    def format_results(self, results: Sequence[ClusterResult]) -> str:
        """Return a plain-text preview summary only.

        The preview deliberately avoids HTML because some result widgets display
        markup as literal text depending on the Qt backend in use.
        """
        if not results:
            return ""

        unique_results = self._unique_report_results(results)
        lines: list[str] = ["Clustering preview"]
        for result in unique_results:
            ids = pd.to_numeric(result.frame["ClusterId"], errors="coerce")
            valid_ids = ids.dropna().astype(int)
            clusters_found = int(valid_ids.nunique()) if not valid_ids.empty else 0
            total_rows = int(len(result.frame))
            finite_rows = int(valid_ids.size)
            feature_text = ", ".join(result.feature_columns)
            lines.append(
                f"- {result.source_name}: {clusters_found} clusters, "
                f"{finite_rows}/{total_rows} clustered rows, "
                f"tool={result.metadata.get('scipy_tool', result.method)}, "
                f"features={feature_text}"
            )
        return "\n".join(lines)

    @staticmethod
    def _unique_report_results(results: Sequence[ClusterResult]) -> list[ClusterResult]:
        unique: dict[tuple[str, str], ClusterResult] = {}
        for result in results:
            key = (
                str(result.metadata.get("source_series_id") or result.source_name),
                str(result.metadata.get("base_result_name") or result.result_name),
            )
            unique.setdefault(key, result)
        return list(unique.values())

    @staticmethod
    def _html_escape(value: Any) -> str:
        return html.escape(str(value), quote=True)

    def refresh_results(self) -> None:
        try:
            results = self.compute_results()
        except Exception as exc:
            self._last_results = []
            self._last_report_html = ""
            self.set_results_text(f"Error:\n{exc}")
            return

        self._last_results = results
        if not results:
            self._last_report_html = ""
            self.set_results_text("Select one or more source series.")
            return

        preview_text = self.format_results(results)
        self._last_report_html = ""
        self.set_results_text(preview_text)

    def apply_results_to_axis(self, axis_id: int, results: Sequence[ClusterResult]) -> None:
        if self.render_mode_combo.currentText() == RENDER_SEPARATE_SERIES:
            self.remove_previous_generated_series(axis_id)

        updated_tables: set[str] = set()
        for result in results:
            source_table = str(result.metadata["source_table"])
            if source_table not in updated_tables:
                self._write_cluster_ids_to_source_table(result)
                updated_tables.add(source_table)
                applogger.info(f"Updated ClusterId in selected table: {source_table}")

            if self.render_mode_combo.currentText() == RENDER_SEPARATE_SERIES:
                self.create_result_series(axis_id, source_table, result)
                continue

            sql_query = self._colored_series_sql_query(source_table, result)
            source_series_id = result.metadata.get("source_series_id")
            if source_series_id is None:
                applogger.error(
                    "Selected series id is missing; cannot update source series SQL.",
                    show_dialog=True,
                    raise_error=True,
                )
                continue
            source_series_id_int = int(source_series_id)
            self._snapshot_series_sql(source_series_id_int)
            self._repo.update_series_sql_query(source_series_id_int, sql_query)
            applogger.info(f"Updated selected series SQL for ClusterId color: {result.source_name}")

        self.applied.emit()

    def preview_results_to_axis(self, axis_id: int, results: Sequence[ClusterResult]) -> None:
        """Preview clustering by applying the cluster result to the active chart.

        Clustering is different from generated-table operations: the chart only
        changes after ClusterId is written to the source table and the selected
        source series SQL is updated or generated split series are created.
        Therefore preview must use the same chart-update path as apply.
        """
        self.apply_results_to_axis(int(axis_id), results)

    def preview(self) -> bool:
        """Preview clustering through the shared pipeline.

        The override exists only to notify the owner window afterwards.  It no
        longer reimplements the pipeline: doing so bypassed
        ``cancel_operation_changes`` and the preview SAVEPOINT, which is why a
        cluster preview used to survive Close - it writes ClusterId into the
        user's own source table and rewrites the selected series' SQL, and
        neither of those is a removable "preview artifact".
        """
        success = super().preview()
        if success and self._applied_callback is not None:
            self._applied_callback()
        return success

    def apply(self) -> bool:
        success = super().apply()
        if success:
            # Committed: the snapshots are no longer a safety net, just clutter.
            self.discard_operation_snapshots()
            if self._applied_callback is not None:
                self._applied_callback()
        return success

    def cancel_operation_changes(self, *, refresh: bool = True) -> None:
        """Undo a cluster preview.

        Overridden because clustering's preview is not made of removable
        artifacts: it overwrites ClusterId in the source table and rewrites the
        selected series' SQL.  Both are restored from the snapshots taken when
        the preview ran.
        """
        restored = self.restore_operation_snapshots()
        super().cancel_operation_changes(refresh=refresh and not restored)

        if restored and refresh:
            self._refresh_after_preview_state_change()

    @staticmethod
    def _source_series_id(series: ClusterSeriesChoice) -> int | None:
        source = series.source
        if source is None:
            return None
        try:
            value = source["id"]
        except (KeyError, TypeError, IndexError):
            return None
        return int(value) if value is not None else None
