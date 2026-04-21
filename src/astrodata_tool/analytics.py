from __future__ import annotations

from dataclasses import dataclass
from html import escape
from io import BytesIO
from math import cos, pi, sin
import os
from pathlib import Path
import tempfile

MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "astrodata_tool_matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib
import numpy as np
import pandas as pd
from pandas import CategoricalDtype

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter


SUPPORTED_AGGREGATIONS = {
    "sum": "sum",
    "mean": "mean",
    "median": "median",
    "count": "count",
    "min": "min",
    "max": "max",
}

SUPPORTED_STATISTICS = [
    "dtype",
    "count",
    "non_null",
    "nulls",
    "unique",
    "mean",
    "median",
    "std",
    "variance",
    "min",
    "max",
    "range",
    "sum",
    "q1",
    "q3",
    "iqr",
    "mode",
]

SVG_PAGE_BACKGROUND = "#ffffff"
SVG_CARD_BACKGROUND = "#ffffff"
SVG_CARD_BORDER = "#d7e1ee"
SVG_PLOT_BACKGROUND = "#f7faff"
SVG_PLOT_BORDER = "#d7e1ee"
SVG_TITLE_COLOR = "#172033"
SVG_SUBTITLE_COLOR = "#53627f"
SVG_ACCENT_COLOR = "#2f6bff"
SVG_GRID_COLOR = "#dbe5f2"
SVG_AXIS_COLOR = "#8a99b2"
SVG_TEXT_ON_COLOR = "#ffffff"


@dataclass(slots=True)
class ChartSpec:
    chart_type: str
    x_column: str
    y_column: str | None = None
    aggregation: str = "sum"
    top_n: int = 12


@dataclass(slots=True)
class ChartArtifact:
    dataframe: pd.DataFrame
    svg_bytes: bytes
    png_bytes: bytes
    title: str
    x_label: str
    y_label: str


def build_chart_artifact(dataframe: pd.DataFrame, spec: ChartSpec, *, dataset_name: str) -> ChartArtifact:
    chart_type = spec.chart_type.lower()
    if spec.aggregation not in SUPPORTED_AGGREGATIONS:
        raise ValueError(f"Unsupported aggregation: {spec.aggregation}")
    if chart_type == "scatter":
        chart_frame = _prepare_scatter_frame(dataframe, spec)
        svg = _build_scatter_svg(chart_frame, dataset_name=dataset_name, spec=spec)
        png = _build_chart_png(chart_frame, dataset_name=dataset_name, spec=spec)
        return ChartArtifact(chart_frame, svg.encode("utf-8"), png, f"{dataset_name}: Scatter Plot", spec.x_column, spec.y_column or "")
    if chart_type == "line":
        chart_frame = _prepare_series_frame(dataframe, spec)
        svg = _build_line_svg(chart_frame, dataset_name=dataset_name, spec=spec)
        png = _build_chart_png(chart_frame, dataset_name=dataset_name, spec=spec)
        return ChartArtifact(chart_frame, svg.encode("utf-8"), png, f"{dataset_name}: Line Chart", spec.x_column, spec.y_column or "Count")
    if chart_type == "bar":
        chart_frame = _prepare_series_frame(dataframe, spec, sort_desc=True, limit=spec.top_n)
        svg = _build_bar_svg(chart_frame, dataset_name=dataset_name, spec=spec)
        png = _build_chart_png(chart_frame, dataset_name=dataset_name, spec=spec)
        return ChartArtifact(chart_frame, svg.encode("utf-8"), png, f"{dataset_name}: Bar Chart", spec.x_column, spec.y_column or "Count")
    if chart_type == "pie":
        chart_frame = _prepare_pie_frame(dataframe, spec)
        svg = _build_pie_svg(chart_frame, dataset_name=dataset_name, spec=spec)
        y_label = spec.y_column or "count"
        png = _build_chart_png(chart_frame, dataset_name=dataset_name, spec=spec)
        return ChartArtifact(chart_frame, svg.encode("utf-8"), png, f"{dataset_name}: Pie Chart", spec.x_column, y_label)
    if chart_type == "heatmap":
        chart_frame = _prepare_heatmap_frame(dataframe, spec)
        y_label = spec.y_column or "count"
        svg = _build_heatmap_svg(chart_frame, dataset_name=dataset_name, spec=spec)
        png = _build_chart_png(chart_frame, dataset_name=dataset_name, spec=spec)
        return ChartArtifact(chart_frame, svg.encode("utf-8"), png, f"{dataset_name}: Heatmap", spec.x_column, y_label)
    raise ValueError(f"Unsupported chart type: {spec.chart_type}")


def _build_chart_png(dataframe: pd.DataFrame, *, dataset_name: str, spec: ChartSpec) -> bytes:
    figure, axis = plt.subplots(figsize=(16, 9), dpi=220, facecolor=SVG_PAGE_BACKGROUND)
    axis.set_facecolor(SVG_PAGE_BACKGROUND)

    try:
        chart_type = spec.chart_type.lower()
        if chart_type == "line":
            _plot_line_png(axis, dataframe, spec=spec)
        elif chart_type == "scatter":
            _plot_scatter_png(axis, dataframe, spec=spec)
        elif chart_type == "bar":
            _plot_bar_png(axis, dataframe, spec=spec)
        elif chart_type == "pie":
            _plot_pie_png(axis, dataframe, spec=spec)
        elif chart_type == "heatmap":
            _plot_heatmap_png(figure, axis, dataframe, spec=spec)
        else:
            raise ValueError(f"Unsupported chart type: {spec.chart_type}")

        png_buffer = BytesIO()
        figure.savefig(
            png_buffer,
            format="png",
            dpi=220,
            facecolor=SVG_PAGE_BACKGROUND,
            bbox_inches="tight",
            pad_inches=0.28,
        )
        png_buffer.seek(0)
        return png_buffer.read()
    finally:
        plt.close(figure)


def _plot_line_png(axis, dataframe: pd.DataFrame, *, spec: ChartSpec) -> None:
    positions = np.arange(len(dataframe), dtype=float)
    values = pd.to_numeric(dataframe["y"], errors="coerce").astype(float).to_numpy()
    labels = [str(value) for value in dataframe["x"].tolist()]
    axis.plot(positions, values, color=SVG_ACCENT_COLOR, linewidth=3.4, marker="o", markersize=6.5)
    axis.fill_between(positions, values, color=SVG_ACCENT_COLOR, alpha=0.10)
    _style_xy_png_axis(axis, positions, labels, x_label=spec.x_column, y_label=spec.y_column or "Count")


def _plot_bar_png(axis, dataframe: pd.DataFrame, *, spec: ChartSpec) -> None:
    positions = np.arange(len(dataframe), dtype=float)
    values = pd.to_numeric(dataframe["y"], errors="coerce").astype(float).to_numpy()
    labels = [str(value) for value in dataframe["x"].tolist()]
    colors = plt.cm.Blues(np.linspace(0.55, 0.85, len(values))) if len(values) else SVG_ACCENT_COLOR
    axis.bar(positions, values, width=0.72, color=colors, edgecolor="#ffffff", linewidth=1.0)
    _style_xy_png_axis(axis, positions, labels, x_label=spec.x_column, y_label=spec.y_column or "Count")


def _plot_scatter_png(axis, dataframe: pd.DataFrame, *, spec: ChartSpec) -> None:
    x_series = dataframe["x"]
    y_values = pd.to_numeric(dataframe["y"], errors="coerce").astype(float).to_numpy()
    if pd.api.types.is_numeric_dtype(x_series) or pd.api.types.is_datetime64_any_dtype(x_series):
        x_values = pd.to_datetime(x_series) if pd.api.types.is_datetime64_any_dtype(x_series) else pd.to_numeric(x_series, errors="coerce")
        axis.scatter(x_values, y_values, s=84, color="#4ba3ff", edgecolors="#ffffff", linewidths=1.1, alpha=0.95)
        _style_numeric_axis(axis, x_label=spec.x_column, y_label=spec.y_column or "")
        return

    positions = np.arange(len(dataframe), dtype=float)
    labels = [str(value) for value in x_series.tolist()]
    axis.scatter(positions, y_values, s=84, color="#4ba3ff", edgecolors="#ffffff", linewidths=1.1, alpha=0.95)
    _style_xy_png_axis(axis, positions, labels, x_label=spec.x_column, y_label=spec.y_column or "")


def _plot_pie_png(axis, dataframe: pd.DataFrame, *, spec: ChartSpec) -> None:
    values = pd.to_numeric(dataframe["y"], errors="coerce").astype(float).to_numpy()
    labels = [_truncate_label(str(value), 18) for value in dataframe["x"].tolist()]
    colors = ["#2f6bff", "#4ba3ff", "#5ef0b7", "#ffd166", "#ff7c6b", "#b28dff", "#3ddc97", "#ff9f43", "#97c1ff", "#ff6384"]
    wedges, texts, autotexts = axis.pie(
        values,
        labels=labels,
        colors=colors[: len(values)],
        startangle=90,
        counterclock=False,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else "",
        pctdistance=0.72,
        labeldistance=1.06,
        wedgeprops={"linewidth": 1.4, "edgecolor": "#ffffff"},
        textprops={"color": SVG_TITLE_COLOR, "fontsize": 11},
    )
    for auto_text in autotexts:
        auto_text.set_color(SVG_TEXT_ON_COLOR)
        auto_text.set_fontsize(10)
        auto_text.set_weight("bold")
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _plot_heatmap_png(figure, axis, dataframe: pd.DataFrame, *, spec: ChartSpec) -> None:
    heatmap_frame = dataframe.pivot(index="y", columns="x", values="value").fillna(0)
    image = axis.imshow(heatmap_frame.to_numpy(), cmap="Blues", aspect="auto")
    axis.set_xticks(np.arange(len(heatmap_frame.columns)))
    axis.set_xticklabels([_truncate_label(str(value), 12) for value in heatmap_frame.columns], rotation=0, fontsize=10, color=SVG_SUBTITLE_COLOR)
    axis.set_yticks(np.arange(len(heatmap_frame.index)))
    axis.set_yticklabels([_truncate_label(str(value), 16) for value in heatmap_frame.index], fontsize=10, color=SVG_SUBTITLE_COLOR)
    axis.set_xlabel(spec.x_column, fontsize=12, color=SVG_TITLE_COLOR, labelpad=12)
    axis.set_ylabel(spec.y_column or "", fontsize=12, color=SVG_TITLE_COLOR, labelpad=12)
    for row_index, row_label in enumerate(heatmap_frame.index):
        for column_index, column_label in enumerate(heatmap_frame.columns):
            value = heatmap_frame.loc[row_label, column_label]
            axis.text(column_index, row_index, _format_number(float(value)), ha="center", va="center", color=SVG_TITLE_COLOR, fontsize=9)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
    colorbar.outline.set_edgecolor(SVG_CARD_BORDER)
    colorbar.ax.tick_params(colors=SVG_SUBTITLE_COLOR, labelsize=9)
    _style_numeric_axis(axis, x_label=spec.x_column, y_label=spec.y_column or "")
    axis.grid(False)


def _style_xy_png_axis(axis, positions, labels: list[str], *, x_label: str, y_label: str) -> None:
    axis.set_xlim(-0.5, max(len(positions) - 0.5, 0.5))
    axis.set_xticks(positions)
    axis.set_xticklabels([_truncate_label(label, 14) for label in labels], fontsize=10, color=SVG_SUBTITLE_COLOR)
    _style_numeric_axis(axis, x_label=x_label, y_label=y_label)


def _style_numeric_axis(axis, *, x_label: str, y_label: str) -> None:
    axis.set_xlabel(x_label, fontsize=12, color=SVG_TITLE_COLOR, labelpad=12)
    axis.set_ylabel(y_label, fontsize=12, color=SVG_TITLE_COLOR, labelpad=12)
    axis.tick_params(axis="y", colors=SVG_SUBTITLE_COLOR, labelsize=10)
    axis.tick_params(axis="x", colors=SVG_SUBTITLE_COLOR)
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _format_number(float(value))))
    axis.grid(axis="y", color=SVG_GRID_COLOR, linewidth=0.8)
    axis.grid(axis="x", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(SVG_CARD_BORDER)
    axis.spines["bottom"].set_color(SVG_CARD_BORDER)
    axis.margins(x=0.02, y=0.08)


def recommend_chart_specs(
    dataframe: pd.DataFrame,
    *,
    chart_type: str,
    preferred_x: str | None = None,
    preferred_y: str | None = None,
    aggregation: str = "sum",
    top_n: int = 12,
) -> list[ChartSpec]:
    all_columns = list(dataframe.columns)
    if not all_columns:
        return []

    numeric_columns = [column for column in all_columns if pd.api.types.is_numeric_dtype(dataframe[column])]
    dimension_columns = [
        column
        for column in all_columns
        if pd.api.types.is_datetime64_any_dtype(dataframe[column])
        or pd.api.types.is_object_dtype(dataframe[column])
        or pd.api.types.is_string_dtype(dataframe[column])
        or isinstance(dataframe[column].dtype, CategoricalDtype)
        or pd.api.types.is_bool_dtype(dataframe[column])
    ]

    candidate_specs: list[ChartSpec] = []
    chart_type = chart_type.lower()

    if chart_type in {"line", "bar"}:
        x_candidates = _ordered_unique([preferred_x, *dimension_columns, *all_columns])
        y_candidates = _ordered_unique([preferred_y, *numeric_columns])
        if None not in y_candidates:
            y_candidates.append(None)
        candidate_specs.extend(
            ChartSpec(chart_type=chart_type, x_column=x_column, y_column=y_column, aggregation=aggregation, top_n=top_n)
            for x_column in x_candidates
            for y_column in y_candidates
        )
    elif chart_type == "scatter":
        xy_candidates = _ordered_unique([preferred_x, preferred_y, *numeric_columns])
        candidate_specs.extend(
            ChartSpec(chart_type="scatter", x_column=x_column, y_column=y_column, aggregation=aggregation, top_n=top_n)
            for x_column in xy_candidates
            for y_column in _ordered_unique([candidate for candidate in xy_candidates if candidate != x_column] or [x_column])
        )
    elif chart_type == "pie":
        x_candidates = _ordered_unique([preferred_x, *dimension_columns, *all_columns])
        y_candidates = _ordered_unique([preferred_y, *numeric_columns])
        candidate_specs.extend(
            ChartSpec(chart_type="pie", x_column=x_column, y_column=y_column, aggregation=aggregation, top_n=top_n)
            for x_column in x_candidates
            for y_column in y_candidates
        )
        candidate_specs.extend(
            ChartSpec(chart_type="pie", x_column=x_column, y_column=None, aggregation=aggregation, top_n=top_n)
            for x_column in x_candidates
        )
    elif chart_type == "heatmap":
        x_candidates = _ordered_unique([preferred_x, *dimension_columns, *all_columns])
        y_candidates = _ordered_unique([preferred_y, *dimension_columns, *all_columns])
        candidate_specs.extend(
            ChartSpec(chart_type="heatmap", x_column=x_column, y_column=y_column, aggregation=aggregation, top_n=top_n)
            for x_column in x_candidates
            for y_column in y_candidates
            if y_column != x_column
        )

    return _deduplicate_specs(candidate_specs)


def build_statistics_table(dataframe: pd.DataFrame, *, columns: list[str], statistics: list[str]) -> pd.DataFrame:
    if not columns:
        raise ValueError("Select at least one column.")
    if not statistics:
        raise ValueError("Select at least one statistic.")

    invalid_columns = sorted(set(columns) - set(dataframe.columns))
    if invalid_columns:
        raise ValueError(f"Unknown columns: {', '.join(invalid_columns)}")

    invalid_statistics = sorted(set(statistics) - set(SUPPORTED_STATISTICS))
    if invalid_statistics:
        raise ValueError(f"Unsupported statistics: {', '.join(invalid_statistics)}")

    rows: list[dict[str, object]] = []
    for column in columns:
        series = dataframe[column]
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_available = numeric.notna().any()
        mode_values = series.mode(dropna=True)
        q1 = numeric.quantile(0.25) if numeric_available else np.nan
        q3 = numeric.quantile(0.75) if numeric_available else np.nan
        min_value = numeric.min() if numeric_available else np.nan
        max_value = numeric.max() if numeric_available else np.nan
        row = {
            "column": column,
            "dtype": str(series.dtype),
            "count": int(series.count()),
            "non_null": int(series.notna().sum()),
            "nulls": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
            "mean": numeric.mean() if numeric_available else np.nan,
            "median": numeric.median() if numeric_available else np.nan,
            "std": numeric.std() if numeric_available else np.nan,
            "variance": numeric.var() if numeric_available else np.nan,
            "min": min_value if numeric_available else _safe_ordered_value(series, reducer="min"),
            "max": max_value if numeric_available else _safe_ordered_value(series, reducer="max"),
            "range": (max_value - min_value) if numeric_available else np.nan,
            "sum": numeric.sum() if numeric_available else np.nan,
            "q1": q1,
            "q3": q3,
            "iqr": (q3 - q1) if numeric_available else np.nan,
            "mode": _safe_scalar(mode_values.iloc[0]) if not mode_values.empty else "",
        }
        filtered_row = {"column": column}
        for statistic in statistics:
            filtered_row[statistic] = row[statistic]
        rows.append(filtered_row)

    return pd.DataFrame(rows)


def statistics_table_to_html(dataframe: pd.DataFrame, *, title: str) -> str:
    header = "".join(f"<th>{escape(str(column))}</th>" for column in dataframe.columns)
    body_rows: list[str] = []
    for _, row in dataframe.iterrows():
        cells = "".join(f"<td>{escape(_format_table_value(row[column]))}</td>" for column in dataframe.columns)
        body_rows.append(f"<tr>{cells}</tr>")
    body = "".join(body_rows)
    safe_title = escape(title)
    return f"""
    <div class="word-table-wrapper">
        <h3>{safe_title}</h3>
        <table class="word-table">
            <thead><tr>{header}</tr></thead>
            <tbody>{body}</tbody>
        </table>
    </div>
    """


def build_word_copy_component(html_table: str, *, component_id: str) -> str:
    safe_html = html_table.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    return f"""
    <html>
    <head>
        <style>
            body {{
                margin: 0;
                background: #0b1324;
                color: #e8eefc;
                font-family: "Segoe UI", sans-serif;
            }}
            .toolbar {{
                display: flex;
                gap: 12px;
                align-items: center;
                margin-bottom: 14px;
            }}
            button {{
                border: none;
                border-radius: 999px;
                padding: 10px 16px;
                background: linear-gradient(135deg, #53b7ff, #5ef0b7);
                color: #07111d;
                font-weight: 700;
                cursor: pointer;
            }}
            .note {{
                color: #99a9c9;
                font-size: 13px;
            }}
            .word-table-wrapper {{
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 16px;
                padding: 18px;
                background: linear-gradient(180deg, rgba(13,21,41,0.96), rgba(10,18,34,0.98));
            }}
            .word-table-wrapper h3 {{
                margin: 0 0 14px 0;
                font-size: 18px;
            }}
            .word-table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 13px;
            }}
            .word-table th, .word-table td {{
                border: 1px solid rgba(140,160,196,0.35);
                padding: 8px 10px;
                text-align: left;
            }}
            .word-table th {{
                background: rgba(83, 183, 255, 0.12);
                color: #eff6ff;
            }}
            .word-table tr:nth-child(even) td {{
                background: rgba(255,255,255,0.03);
            }}
            #status-{component_id} {{
                color: #7fe2b1;
                min-height: 18px;
            }}
        </style>
    </head>
    <body>
        <div class="toolbar">
            <button id="copy-{component_id}">Copy Table for Word</button>
            <div class="note">Paste directly into Microsoft Word to keep the table structure.</div>
        </div>
        <div id="status-{component_id}"></div>
        <div id="table-{component_id}">{html_table}</div>
        <script>
            const copyButton = document.getElementById("copy-{component_id}");
            const statusNode = document.getElementById("status-{component_id}");
            const tableNode = document.getElementById("table-{component_id}");
            const htmlValue = `{safe_html}`;
            const textValue = tableNode.innerText;

            copyButton.addEventListener("click", async () => {{
                try {{
                    if (window.ClipboardItem && navigator.clipboard && navigator.clipboard.write) {{
                        await navigator.clipboard.write([
                            new ClipboardItem({{
                                "text/html": new Blob([htmlValue], {{ type: "text/html" }}),
                                "text/plain": new Blob([textValue], {{ type: "text/plain" }}),
                            }})
                        ]);
                    }} else if (navigator.clipboard && navigator.clipboard.writeText) {{
                        await navigator.clipboard.writeText(textValue);
                    }} else {{
                        throw new Error("Clipboard API unavailable");
                    }}
                    statusNode.textContent = "Table copied. Paste into Word.";
                }} catch (error) {{
                    statusNode.textContent = "Clipboard access is blocked in this browser. Use the HTML download instead.";
                }}
            }});
        </script>
    </body>
    </html>
    """


def _prepare_scatter_frame(dataframe: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    if spec.y_column is None:
        raise ValueError("Scatter charts require a numeric Y-axis column.")
    x_series = _get_chart_series(dataframe, spec.x_column)
    y_series = pd.to_numeric(_get_chart_series(dataframe, spec.y_column), errors="coerce")
    working = pd.DataFrame({"x": x_series, "y": y_series}).dropna(subset=["x", "y"]).head(1500)
    if working.empty:
        raise ValueError("The selected columns do not contain enough numeric data to plot.")
    return working


def _prepare_series_frame(
    dataframe: pd.DataFrame,
    spec: ChartSpec,
    *,
    sort_desc: bool = False,
    limit: int | None = None,
) -> pd.DataFrame:
    x_series = _get_chart_series(dataframe, spec.x_column)
    if spec.y_column is None:
        grouped = (
            pd.DataFrame({"x": x_series})
            .dropna(subset=["x"])
            .groupby("x", dropna=False, sort=False, observed=False)
            .size()
            .reset_index(name="y")
        )
    else:
        y_series = pd.to_numeric(_get_chart_series(dataframe, spec.y_column), errors="coerce")
        working = pd.DataFrame({"x": x_series, "y": y_series}).dropna(subset=["x", "y"])
        if working.empty:
            raise ValueError("The selected columns do not contain enough numeric data to plot.")

        aggregation = SUPPORTED_AGGREGATIONS[spec.aggregation]
        grouped = (
            working.groupby("x", dropna=False, sort=False, observed=False)["y"]
            .agg(aggregation)
            .reset_index()
        )
    if _is_sortable_series(grouped["x"]):
        grouped = grouped.sort_values("x")
    if sort_desc:
        grouped = grouped.sort_values("y", ascending=False)
    if limit is not None:
        grouped = grouped.head(limit)
    if grouped.empty:
        raise ValueError("No data points were generated for the selected chart.")
    return grouped


def _prepare_pie_frame(dataframe: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    if spec.y_column:
        return _prepare_series_frame(dataframe, spec, sort_desc=True, limit=spec.top_n)

    grouped = (
        pd.DataFrame({"x": _get_chart_series(dataframe, spec.x_column)})
        .dropna(subset=["x"])
        .groupby("x", dropna=False, observed=False)
        .size()
        .reset_index(name="y")
        .sort_values("y", ascending=False)
        .head(spec.top_n)
    )
    if grouped.empty:
        raise ValueError("No values are available to build the pie chart.")
    return grouped


def _prepare_heatmap_frame(dataframe: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    if spec.y_column is None:
        raise ValueError("Heatmaps require a second column.")

    working = pd.DataFrame(
        {
            "x": _get_chart_series(dataframe, spec.x_column).astype(str),
            "y": _get_chart_series(dataframe, spec.y_column).astype(str),
        }
    ).dropna(subset=["x", "y"])
    if working.empty:
        raise ValueError("The selected columns do not contain enough values to build a heatmap.")

    top_x = working["x"].value_counts().head(spec.top_n).index.tolist()
    top_y = working["y"].value_counts().head(spec.top_n).index.tolist()
    filtered = working[working["x"].isin(top_x) & working["y"].isin(top_y)].copy(deep=True)
    grouped = (
        filtered.groupby(["x", "y"], observed=False)
        .size()
        .reset_index(name="value")
    )
    if grouped.empty:
        raise ValueError("No values are available to build the heatmap.")
    return grouped


def _get_chart_series(dataframe: pd.DataFrame, column_name: str) -> pd.Series:
    series_or_frame = dataframe.loc[:, column_name]
    if isinstance(series_or_frame, pd.DataFrame):
        raise ValueError(
            f"Charting requires unique column names. Rename duplicate column '{column_name}' before plotting."
        )
    return series_or_frame.copy(deep=True)


def _build_line_svg(dataframe: pd.DataFrame, *, dataset_name: str, spec: ChartSpec) -> str:
    return _build_xy_chart_svg(
        dataframe=dataframe,
        dataset_name=dataset_name,
        title=f"Line Chart | {dataset_name}",
        subtitle=f"{spec.aggregation.title()} of {spec.y_column} by {spec.x_column}",
        chart_style="line",
        x_label=spec.x_column,
        y_label=spec.y_column or "",
    )


def _build_scatter_svg(dataframe: pd.DataFrame, *, dataset_name: str, spec: ChartSpec) -> str:
    return _build_xy_chart_svg(
        dataframe=dataframe,
        dataset_name=dataset_name,
        title=f"Scatter Plot | {dataset_name}",
        subtitle=f"{spec.y_column} versus {spec.x_column}",
        chart_style="scatter",
        x_label=spec.x_column,
        y_label=spec.y_column or "",
    )


def _build_bar_svg(dataframe: pd.DataFrame, *, dataset_name: str, spec: ChartSpec) -> str:
    return _build_xy_chart_svg(
        dataframe=dataframe,
        dataset_name=dataset_name,
        title=f"Bar Chart | {dataset_name}",
        subtitle=f"Top {len(dataframe)} {spec.x_column} values by {spec.aggregation}",
        chart_style="bar",
        x_label=spec.x_column,
        y_label=spec.y_column or "",
    )


def _build_xy_chart_svg(
    *,
    dataframe: pd.DataFrame,
    dataset_name: str,
    title: str,
    subtitle: str,
    chart_style: str,
    x_label: str,
    y_label: str,
) -> str:
    width = 1280
    height = 760
    left = 110
    right = 1180
    top = 150
    bottom = 620
    plot_width = right - left
    plot_height = bottom - top

    x_values = dataframe["x"].tolist()
    y_values = pd.to_numeric(dataframe["y"], errors="coerce").astype(float).tolist()
    y_max = max(y_values) if y_values else 1.0
    y_max = y_max if y_max > 0 else 1.0

    if len(x_values) == 1:
        x_positions = [left + (plot_width / 2)]
    else:
        x_positions = [left + (index / (len(x_values) - 1)) * plot_width for index in range(len(x_values))]

    def y_to_pixel(value: float) -> float:
        return bottom - ((value / y_max) * plot_height)

    grid_lines = []
    y_ticks = []
    for idx in range(6):
        value = y_max * idx / 5
        y_pixel = y_to_pixel(value)
        grid_lines.append(f'<line x1="{left}" y1="{y_pixel:.2f}" x2="{right}" y2="{y_pixel:.2f}" stroke="{SVG_GRID_COLOR}" stroke-width="1" />')
        y_ticks.append(
            f'<text x="{left - 18}" y="{y_pixel + 5:.2f}" text-anchor="end" fill="{SVG_AXIS_COLOR}" font-size="13">{escape(_format_number(value))}</text>'
        )

    x_ticks = []
    for x_position, label in zip(x_positions, x_values, strict=False):
        truncated = _truncate_label(label, 16)
        x_ticks.append(
            f'<text x="{x_position:.2f}" y="{bottom + 28}" text-anchor="middle" fill="{SVG_AXIS_COLOR}" font-size="12">{escape(truncated)}</text>'
        )

    points = [f"{x:.2f},{y_to_pixel(y):.2f}" for x, y in zip(x_positions, y_values, strict=False)]
    path_d = ""
    if points:
        first, *rest = points
        path_d = "M " + first + "".join(f" L {point}" for point in rest)

    marks: list[str] = []
    if chart_style == "line":
        marks.append(f'<path d="{path_d}" fill="none" stroke="{SVG_ACCENT_COLOR}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />')
        area_points = " ".join(points + [f"{right:.2f},{bottom:.2f}", f"{left:.2f},{bottom:.2f}"])
        marks.append(f'<polygon points="{area_points}" fill="url(#lineArea)" opacity="0.25" />')
    elif chart_style == "bar":
        bar_width = max(18.0, min(72.0, plot_width / max(len(x_positions) * 1.8, 1)))
        for x_position, value in zip(x_positions, y_values, strict=False):
            top_y = y_to_pixel(value)
            marks.append(
                f'<rect x="{x_position - (bar_width / 2):.2f}" y="{top_y:.2f}" width="{bar_width:.2f}" height="{bottom - top_y:.2f}" rx="10" fill="url(#barFill)" />'
            )
            marks.append(
                f'<text x="{x_position:.2f}" y="{top_y - 10:.2f}" text-anchor="middle" fill="{SVG_TITLE_COLOR}" font-size="12">{escape(_format_number(value))}</text>'
            )
    else:
        for x_position, value in zip(x_positions, y_values, strict=False):
            y_pixel = y_to_pixel(value)
            marks.append(f'<circle cx="{x_position:.2f}" cy="{y_pixel:.2f}" r="7.5" fill="#59d6d6" stroke="{SVG_TEXT_ON_COLOR}" stroke-width="2" />')
            marks.append(f'<circle cx="{x_position:.2f}" cy="{y_pixel:.2f}" r="18" fill="#59d6d6" opacity="0.10" />')

    if chart_style == "line":
        for x_position, value in zip(x_positions, y_values, strict=False):
            y_pixel = y_to_pixel(value)
            marks.append(f'<circle cx="{x_position:.2f}" cy="{y_pixel:.2f}" r="6" fill="{SVG_PAGE_BACKGROUND}" stroke="#4ba3ff" stroke-width="3" />')

    return f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
        <defs>
            <linearGradient id="cardFill" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="#ffffff" />
                <stop offset="100%" stop-color="#f7faff" />
            </linearGradient>
            <linearGradient id="lineArea" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#2f6bff" stop-opacity="0.24" />
                <stop offset="100%" stop-color="#2f6bff" stop-opacity="0" />
            </linearGradient>
            <linearGradient id="barFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#4ba3ff" />
                <stop offset="100%" stop-color="#2f6bff" />
            </linearGradient>
        </defs>
        <rect width="{width}" height="{height}" fill="{SVG_PAGE_BACKGROUND}" />
        <rect x="34" y="28" width="{width - 68}" height="{height - 56}" rx="26" fill="url(#cardFill)" stroke="{SVG_CARD_BORDER}" />
        <text x="76" y="90" fill="{SVG_TITLE_COLOR}" font-size="32" font-weight="700">{escape(title)}</text>
        <text x="76" y="124" fill="{SVG_SUBTITLE_COLOR}" font-size="16">{escape(subtitle)}</text>
        <text x="{right}" y="90" fill="{SVG_ACCENT_COLOR}" font-size="16" text-anchor="end">{escape(dataset_name)}</text>
        <rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" rx="20" fill="{SVG_PLOT_BACKGROUND}" stroke="{SVG_PLOT_BORDER}" />
        {''.join(grid_lines)}
        <line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="{SVG_CARD_BORDER}" stroke-width="2" />
        <line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="{SVG_CARD_BORDER}" stroke-width="2" />
        {''.join(y_ticks)}
        {''.join(x_ticks)}
        {''.join(marks)}
        <text x="{(left + right) / 2:.2f}" y="700" fill="{SVG_SUBTITLE_COLOR}" font-size="15" text-anchor="middle">{escape(x_label)}</text>
        <text x="36" y="{(top + bottom) / 2:.2f}" fill="{SVG_SUBTITLE_COLOR}" font-size="15" transform="rotate(-90 36 {(top + bottom) / 2:.2f})" text-anchor="middle">{escape(y_label)}</text>
    </svg>
    """


def _build_pie_svg(dataframe: pd.DataFrame, *, dataset_name: str, spec: ChartSpec) -> str:
    width = 1280
    height = 760
    center_x = 410
    center_y = 390
    radius = 180
    values = pd.to_numeric(dataframe["y"], errors="coerce").astype(float).tolist()
    total = sum(values) or 1.0
    colors = ["#59d6d6", "#4ba3ff", "#5ef0b7", "#ffd166", "#ff7c6b", "#b28dff", "#3ddc97", "#ff9f43", "#97c1ff", "#ff6384"]

    slices: list[str] = []
    legend: list[str] = []
    start_angle = -pi / 2
    for index, (_, row) in enumerate(dataframe.iterrows()):
        value = float(row["y"])
        label = str(row["x"])
        angle = (value / total) * 2 * pi
        end_angle = start_angle + angle
        large_arc = 1 if angle > pi else 0
        x1 = center_x + radius * cos(start_angle)
        y1 = center_y + radius * sin(start_angle)
        x2 = center_x + radius * cos(end_angle)
        y2 = center_y + radius * sin(end_angle)
        color = colors[index % len(colors)]
        slices.append(
            f'<path d="M {center_x} {center_y} L {x1:.2f} {y1:.2f} A {radius} {radius} 0 {large_arc} 1 {x2:.2f} {y2:.2f} Z" fill="{color}" stroke="{SVG_CARD_BACKGROUND}" stroke-width="3" />'
        )
        legend_y = 195 + index * 42
        percentage = (value / total) * 100
        legend.append(f'<rect x="740" y="{legend_y - 16}" width="18" height="18" rx="4" fill="{color}" />')
        legend.append(f'<text x="770" y="{legend_y}" fill="{SVG_TITLE_COLOR}" font-size="15">{escape(_truncate_label(label, 28))}</text>')
        legend.append(f'<text x="1120" y="{legend_y}" fill="{SVG_SUBTITLE_COLOR}" font-size="14" text-anchor="end">{escape(_format_number(value))} | {percentage:.1f}%</text>')
        start_angle = end_angle

    subtitle = f"{(spec.y_column or 'count')} distribution by {spec.x_column}"
    return f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
        <rect width="{width}" height="{height}" fill="{SVG_PAGE_BACKGROUND}" />
        <rect x="34" y="28" width="{width - 68}" height="{height - 56}" rx="26" fill="{SVG_CARD_BACKGROUND}" stroke="{SVG_CARD_BORDER}" />
        <text x="76" y="90" fill="{SVG_TITLE_COLOR}" font-size="32" font-weight="700">Pie Chart | {escape(dataset_name)}</text>
        <text x="76" y="124" fill="{SVG_SUBTITLE_COLOR}" font-size="16">{escape(subtitle)}</text>
        <circle cx="{center_x}" cy="{center_y}" r="{radius + 30}" fill="rgba(89,214,214,0.05)" />
        {''.join(slices)}
        <circle cx="{center_x}" cy="{center_y}" r="86" fill="{SVG_CARD_BACKGROUND}" />
        <text x="{center_x}" y="{center_y - 6}" fill="{SVG_TITLE_COLOR}" font-size="18" text-anchor="middle" font-weight="700">{escape(_format_number(total))}</text>
        <text x="{center_x}" y="{center_y + 24}" fill="{SVG_SUBTITLE_COLOR}" font-size="14" text-anchor="middle">Total</text>
        <text x="740" y="150" fill="{SVG_TITLE_COLOR}" font-size="22" font-weight="700">Breakdown</text>
        {''.join(legend)}
    </svg>
    """


def _build_heatmap_svg(dataframe: pd.DataFrame, *, dataset_name: str, spec: ChartSpec) -> str:
    width = 1280
    height = 760
    left = 170
    top = 170
    cell_width = 72
    cell_height = 52
    x_values = list(dict.fromkeys(dataframe["x"].tolist()))
    y_values = list(dict.fromkeys(dataframe["y"].tolist()))
    max_value = float(pd.to_numeric(dataframe["value"], errors="coerce").max())
    max_value = max_value if max_value > 0 else 1.0

    cells: list[str] = []
    for _, row in dataframe.iterrows():
        x_index = x_values.index(row["x"])
        y_index = y_values.index(row["y"])
        value = float(row["value"])
        intensity = value / max_value
        fill = _blend_color("#0f1b33", "#58d0ff", intensity)
        x_pixel = left + x_index * cell_width
        y_pixel = top + y_index * cell_height
        cells.append(f'<rect x="{x_pixel}" y="{y_pixel}" width="{cell_width - 6}" height="{cell_height - 6}" rx="12" fill="{fill}" />')
        cells.append(f'<text x="{x_pixel + (cell_width / 2) - 3}" y="{y_pixel + 31}" fill="{SVG_TITLE_COLOR}" font-size="13" text-anchor="middle">{escape(_format_number(value))}</text>')

    x_labels = [
        f'<text x="{left + index * cell_width + 30}" y="{top - 18}" fill="{SVG_SUBTITLE_COLOR}" font-size="12" text-anchor="middle">{escape(_truncate_label(label, 12))}</text>'
        for index, label in enumerate(x_values)
    ]
    y_labels = [
        f'<text x="{left - 14}" y="{top + index * cell_height + 30}" fill="{SVG_SUBTITLE_COLOR}" font-size="12" text-anchor="end">{escape(_truncate_label(label, 18))}</text>'
        for index, label in enumerate(y_values)
    ]

    subtitle = f"Frequency heatmap for {spec.x_column} and {spec.y_column}"
    return f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
        <rect width="{width}" height="{height}" fill="{SVG_PAGE_BACKGROUND}" />
        <rect x="34" y="28" width="{width - 68}" height="{height - 56}" rx="26" fill="{SVG_CARD_BACKGROUND}" stroke="{SVG_CARD_BORDER}" />
        <text x="76" y="90" fill="{SVG_TITLE_COLOR}" font-size="32" font-weight="700">Heatmap | {escape(dataset_name)}</text>
        <text x="76" y="124" fill="{SVG_SUBTITLE_COLOR}" font-size="16">{escape(subtitle)}</text>
        <rect x="{left - 26}" y="{top - 42}" width="{len(x_values) * cell_width + 42}" height="{len(y_values) * cell_height + 42}" rx="24" fill="{SVG_PLOT_BACKGROUND}" stroke="{SVG_PLOT_BORDER}" />
        {''.join(x_labels)}
        {''.join(y_labels)}
        {''.join(cells)}
    </svg>
    """


def _format_number(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:,.1f}"
    return f"{value:,.2f}"


def _format_table_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if pd.isna(value):
            return ""
        return f"{value:,.4f}"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if pd.isna(value):
        return ""
    return str(value)


def _is_sortable_series(series: pd.Series) -> bool:
    try:
        series.sort_values()
        return True
    except TypeError:
        return False


def _safe_scalar(value: object) -> object:
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return value


def _safe_ordered_value(series: pd.Series, *, reducer: str) -> object:
    non_null = series.dropna()
    if non_null.empty:
        return ""
    try:
        return _safe_scalar(getattr(non_null, reducer)())
    except TypeError:
        ordered_values = sorted(str(value) for value in non_null.tolist())
        return ordered_values[0] if reducer == "min" else ordered_values[-1]


def _truncate_label(value: object, limit: int) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _blend_color(start_hex: str, end_hex: str, factor: float) -> str:
    factor = max(0.0, min(1.0, factor))
    start = tuple(int(start_hex[index:index + 2], 16) for index in (1, 3, 5))
    end = tuple(int(end_hex[index:index + 2], 16) for index in (1, 3, 5))
    blended = tuple(round(start[idx] + ((end[idx] - start[idx]) * factor)) for idx in range(3))
    return "#" + "".join(f"{channel:02x}" for channel in blended)


def _ordered_unique(values: list[str | None]) -> list[str]:
    ordered: list[str] = []
    for value in values:
        if value is None or value in ordered:
            continue
        ordered.append(value)
    return ordered


def _deduplicate_specs(specs: list[ChartSpec]) -> list[ChartSpec]:
    seen: set[tuple[str, str, str | None, str, int]] = set()
    unique_specs: list[ChartSpec] = []
    for spec in specs:
        key = (spec.chart_type, spec.x_column, spec.y_column, spec.aggregation, spec.top_n)
        if key in seen:
            continue
        seen.add(key)
        unique_specs.append(spec)
    return unique_specs
