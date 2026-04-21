from __future__ import annotations

import pandas as pd

from astrodata_tool.analytics import ChartSpec, build_chart_artifact, build_statistics_table, recommend_chart_specs, statistics_table_to_html


def test_build_statistics_table_includes_requested_numeric_metrics(sample_dataframe: pd.DataFrame) -> None:
    stats = build_statistics_table(
        sample_dataframe,
        columns=["value", "category"],
        statistics=["dtype", "count", "nulls", "mean", "median", "std", "mode"],
    )

    value_row = stats.loc[stats["column"] == "value"].iloc[0]
    category_row = stats.loc[stats["column"] == "category"].iloc[0]

    assert value_row["count"] == 2
    assert value_row["nulls"] == 2
    assert value_row["mean"] == 205.0
    assert value_row["median"] == 205.0
    assert category_row["mode"] == "b"


def test_build_chart_artifact_returns_svg_for_bar_chart(sample_dataframe: pd.DataFrame) -> None:
    artifact = build_chart_artifact(
        sample_dataframe,
        ChartSpec(chart_type="bar", x_column="category", y_column="value", aggregation="sum", top_n=5),
        dataset_name="sample",
    )

    assert artifact.dataframe.columns.tolist() == ["x", "y"]
    assert b"<svg" in artifact.svg_bytes
    assert artifact.png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert "Bar Chart" in artifact.title


def test_build_chart_artifact_allows_same_column_for_x_and_y(sample_dataframe: pd.DataFrame) -> None:
    artifact = build_chart_artifact(
        sample_dataframe,
        ChartSpec(chart_type="line", x_column="id", y_column="id", aggregation="sum", top_n=5),
        dataset_name="sample",
    )

    assert artifact.dataframe.columns.tolist() == ["x", "y"]
    assert not artifact.dataframe.empty
    assert b"<svg" in artifact.svg_bytes


def test_build_chart_artifact_supports_count_based_bar_chart(sample_dataframe: pd.DataFrame) -> None:
    artifact = build_chart_artifact(
        sample_dataframe,
        ChartSpec(chart_type="bar", x_column="category", y_column=None, aggregation="count", top_n=5),
        dataset_name="sample",
    )

    assert artifact.dataframe.columns.tolist() == ["x", "y"]
    assert artifact.y_label == "Count"
    assert artifact.dataframe["y"].sum() == 4


def test_recommend_chart_specs_prefers_dimension_x_and_numeric_y(sample_dataframe: pd.DataFrame) -> None:
    specs = recommend_chart_specs(sample_dataframe, chart_type="bar")

    assert specs
    assert specs[0].x_column == "category"
    assert specs[0].y_column == "id"


def test_recommend_chart_specs_returns_pie_count_fallback(sample_dataframe: pd.DataFrame) -> None:
    specs = recommend_chart_specs(sample_dataframe, chart_type="pie", preferred_x="category")

    assert specs
    assert any(spec.x_column == "category" and spec.y_column is None for spec in specs)


def test_recommend_chart_specs_returns_count_bar_when_no_numeric_columns() -> None:
    dataframe = pd.DataFrame({"segment": ["A", "A", "B"], "region": ["East", "West", "East"]})

    specs = recommend_chart_specs(dataframe, chart_type="bar", preferred_x="segment")

    assert specs
    assert specs[0].x_column == "segment"
    assert specs[0].y_column is None


def test_statistics_table_to_html_renders_table_markup(sample_dataframe: pd.DataFrame) -> None:
    stats = build_statistics_table(sample_dataframe, columns=["id"], statistics=["dtype", "count", "mean"])
    html = statistics_table_to_html(stats, title="Sample Table")

    assert "<table" in html
    assert "Sample Table" in html
    assert "mean" in html
