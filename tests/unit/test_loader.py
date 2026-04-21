from __future__ import annotations

from io import BytesIO

import pandas as pd
import pytest

from astrodata_tool import DataLoader
from astrodata_tool.exceptions import DataLoadError


def test_loader_reads_csv_into_dataset(csv_buffer: BytesIO) -> None:
    loader = DataLoader()

    datasets = loader.load_file_object(file_name="sample.csv", file_object=csv_buffer)

    assert len(datasets) == 1
    assert datasets[0].name == "sample"
    assert list(datasets[0].dataframe.columns) == ["id", "value", "category", "event_date"]


def test_loader_reads_excel_first_sheet_only_by_default(sample_dataframe: pd.DataFrame) -> None:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        sample_dataframe.to_excel(writer, index=False, sheet_name="events")
        sample_dataframe.head(2).to_excel(writer, index=False, sheet_name="summary")
    buffer.seek(0)

    loader = DataLoader()
    datasets = loader.load_file_object(file_name="workbook.xlsx", file_object=buffer)

    assert len(datasets) == 1
    assert datasets[0].sheet_name == "events"


def test_loader_rejects_empty_payload() -> None:
    loader = DataLoader()

    with pytest.raises(DataLoadError):
        loader.load_file_object(file_name="empty.csv", file_object=BytesIO(b""))
