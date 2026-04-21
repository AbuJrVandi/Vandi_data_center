from __future__ import annotations

from typing import Any

import pandas as pd

from .models import DatasetArtifact, ProfileReport


class DataProfiler:
    def profile(self, dataset: DatasetArtifact) -> ProfileReport:
        dataframe = dataset.dataframe
        try:
            summary_frame = dataframe.describe(include="all", datetime_is_numeric=True).fillna("")
        except TypeError:
            summary_frame = dataframe.describe(include="all").fillna("")
        summary_statistics: dict[str, dict[str, Any]] = {}
        for index, row in summary_frame.iterrows():
            summary_statistics[str(index)] = {column: self._coerce_value(value) for column, value in row.items()}

        return ProfileReport(
            dataset_name=dataset.name,
            row_count=int(dataframe.shape[0]),
            column_count=int(dataframe.shape[1]),
            dtypes={column: str(dtype) for column, dtype in dataframe.dtypes.items()},
            missing_values={column: int(value) for column, value in dataframe.isna().sum().items()},
            duplicate_rows=int(dataframe.duplicated().sum()),
            summary_statistics=summary_statistics,
            memory_usage_bytes=int(dataframe.memory_usage(deep=True).sum()),
        )

    @staticmethod
    def _coerce_value(value: Any) -> Any:
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            return str(value)
        if hasattr(value, "item"):
            try:
                return value.item()
            except ValueError:
                return str(value)
        return value
