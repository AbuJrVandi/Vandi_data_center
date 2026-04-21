from __future__ import annotations

from typing import Any

import pandas as pd

from .exceptions import DataValidationError
from .models import DatasetArtifact, OperationRecord


class DataCleaner:
    def remove_duplicates(
        self,
        dataset: DatasetArtifact,
        *,
        subset: list[str] | None = None,
        keep: str = "first",
    ) -> DatasetArtifact:
        self._validate_columns(dataset.dataframe, subset or [])
        dataframe = dataset.dataframe.drop_duplicates(subset=subset, keep=keep).copy(deep=True)
        result = dataset.clone(dataframe=dataframe, name=f"{dataset.name}_deduplicated")
        result.operation_history.append(
            OperationRecord(
                operation_name="remove_duplicates",
                parameters={"subset": subset, "keep": keep},
                summary=f"Removed {dataset.row_count - result.row_count} duplicate rows.",
                dataset_before=dataset.name,
                dataset_after=result.name,
            )
        )
        return result

    def handle_missing_values(
        self,
        dataset: DatasetArtifact,
        *,
        method: str,
        columns: list[str] | None = None,
        fill_value: Any = None,
    ) -> DatasetArtifact:
        target_columns = columns or list(dataset.dataframe.columns)
        self._validate_columns(dataset.dataframe, target_columns)

        dataframe = dataset.dataframe.copy(deep=True)
        original_missing = int(dataframe[target_columns].isna().sum().sum())
        if method == "drop":
            dataframe = dataframe.dropna(subset=target_columns).copy(deep=True)
        elif method == "mean":
            numeric_columns = list(dataframe[target_columns].select_dtypes(include="number").columns)
            dataframe[numeric_columns] = dataframe[numeric_columns].fillna(dataframe[numeric_columns].mean())
        elif method == "median":
            numeric_columns = list(dataframe[target_columns].select_dtypes(include="number").columns)
            dataframe[numeric_columns] = dataframe[numeric_columns].fillna(dataframe[numeric_columns].median())
        elif method == "mode":
            for column in target_columns:
                modes = dataframe[column].mode(dropna=True)
                if not modes.empty:
                    dataframe[column] = dataframe[column].fillna(modes.iloc[0])
        elif method == "forward_fill":
            dataframe[target_columns] = dataframe[target_columns].ffill()
        elif method == "constant":
            dataframe[target_columns] = dataframe[target_columns].fillna(fill_value)
        else:
            raise DataValidationError(f"Unsupported missing value strategy: {method}")

        remaining_missing = int(dataframe[target_columns].isna().sum().sum())
        result = dataset.clone(dataframe=dataframe, name=f"{dataset.name}_cleaned")
        result.operation_history.append(
            OperationRecord(
                operation_name="handle_missing_values",
                parameters={"method": method, "columns": target_columns, "fill_value": fill_value},
                summary=f"Missing values changed from {original_missing} to {remaining_missing}.",
                dataset_before=dataset.name,
                dataset_after=result.name,
            )
        )
        return result

    @staticmethod
    def _validate_columns(dataframe: pd.DataFrame, columns: list[str]) -> None:
        missing_columns = sorted(set(columns) - set(dataframe.columns))
        if missing_columns:
            raise DataValidationError(f"Unknown columns: {', '.join(missing_columns)}")
