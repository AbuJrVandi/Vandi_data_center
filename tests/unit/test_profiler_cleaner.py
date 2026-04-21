from __future__ import annotations

from astrodata_tool import DataCleaner, DataProfiler


def test_profiler_reports_missing_values_and_duplicates(sample_dataset) -> None:
    profiler = DataProfiler()

    report = profiler.profile(sample_dataset)

    assert report.row_count == 4
    assert report.duplicate_rows == 1
    assert report.missing_values["value"] == 2


def test_cleaner_handles_missing_values_by_median(sample_dataset) -> None:
    cleaner = DataCleaner()

    cleaned = cleaner.handle_missing_values(sample_dataset, method="median", columns=["value"])

    assert cleaned.dataframe["value"].isna().sum() == 0
    assert cleaned.dataframe.loc[1, "value"] == 205.0


def test_cleaner_removes_duplicates(sample_dataset) -> None:
    cleaner = DataCleaner()

    deduplicated = cleaner.remove_duplicates(sample_dataset)

    assert deduplicated.row_count == 3
