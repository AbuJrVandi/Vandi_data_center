from __future__ import annotations

from astrodata_tool import DataFilter, DataValidator, FilterCondition


def test_filter_applies_multiple_conditions(sample_dataset) -> None:
    data_filter = DataFilter()

    filtered = data_filter.apply_conditions(
        sample_dataset,
        [
            FilterCondition(column="category", operator="==", value="b"),
            FilterCondition(column="id", operator=">=", value="2"),
        ],
    )

    assert filtered.row_count == 2
    assert filtered.dataframe["category"].unique().tolist() == ["b"]


def test_validator_detects_type_range_and_outliers(sample_dataset) -> None:
    validator = DataValidator()

    report = validator.validate(
        sample_dataset,
        expected_types={"id": "int64", "event_date": "datetime64[ns]"},
        range_rules={"value": {"min": 0, "max": 100}},
        outlier_columns=["value"],
        outlier_z_threshold=0.5,
    )

    assert any(issue.category == "type_validation" for issue in report.issues)
    assert any(issue.category == "range_validation" for issue in report.issues)
    assert any(issue.category == "outlier_detection" for issue in report.issues)
