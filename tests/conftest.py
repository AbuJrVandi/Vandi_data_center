from __future__ import annotations

from io import BytesIO

import pandas as pd
import pytest

from astrodata_tool import DatasetArtifact


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 2, 4],
            "value": [10.0, None, None, 400.0],
            "category": ["a", "b", "b", "c"],
            "event_date": ["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"],
        }
    )


@pytest.fixture
def sample_dataset(sample_dataframe: pd.DataFrame) -> DatasetArtifact:
    return DatasetArtifact(
        name="sample",
        dataframe=sample_dataframe.copy(deep=True),
        source_name="sample.csv",
        source_type="csv",
    )


@pytest.fixture
def csv_buffer(sample_dataframe: pd.DataFrame) -> BytesIO:
    return BytesIO(sample_dataframe.to_csv(index=False).encode("utf-8"))
