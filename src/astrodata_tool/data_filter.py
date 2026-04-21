from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from .exceptions import DataValidationError
from .models import DatasetArtifact, FilterCondition, OperationRecord


class DataFilter:
    SUPPORTED_OPERATORS = {
        "==",
        "!=",
        ">",
        ">=",
        "<",
        "<=",
        "contains",
        "in",
        "between",
        "is_null",
        "not_null",
    }

    def apply_conditions(self, dataset: DatasetArtifact, conditions: list[FilterCondition]) -> DatasetArtifact:
        if not conditions:
            raise DataValidationError("At least one filter condition is required.")

        dataframe = dataset.dataframe.copy(deep=True)
        mask = pd.Series(True, index=dataframe.index)
        for condition in conditions:
            if condition.operator not in self.SUPPORTED_OPERATORS:
                raise DataValidationError(f"Unsupported filter operator: {condition.operator}")
            if condition.column not in dataframe.columns:
                raise DataValidationError(f"Unknown filter column: {condition.column}")

            series = dataframe[condition.column]
            typed_value = self._coerce_value(condition.value, series)
            typed_secondary = self._coerce_value(condition.secondary_value, series)

            if condition.operator == "==":
                mask &= series == typed_value
            elif condition.operator == "!=":
                mask &= series != typed_value
            elif condition.operator == ">":
                mask &= series > typed_value
            elif condition.operator == ">=":
                mask &= series >= typed_value
            elif condition.operator == "<":
                mask &= series < typed_value
            elif condition.operator == "<=":
                mask &= series <= typed_value
            elif condition.operator == "contains":
                mask &= series.astype("string").str.contains(str(condition.value), case=False, na=False)
            elif condition.operator == "in":
                values = condition.value if isinstance(condition.value, list) else [value.strip() for value in str(condition.value).split(",")]
                mask &= series.isin([self._coerce_value(value, series) for value in values])
            elif condition.operator == "between":
                if typed_value is None or typed_secondary is None:
                    raise DataValidationError("Between filters require both values.")
                mask &= series.between(typed_value, typed_secondary)
            elif condition.operator == "is_null":
                mask &= series.isna()
            elif condition.operator == "not_null":
                mask &= series.notna()

        filtered = dataframe.loc[mask].copy(deep=True)
        result = dataset.clone(dataframe=filtered, name=f"{dataset.name}_filtered")
        result.operation_history.append(
            OperationRecord(
                operation_name="filter_rows",
                parameters={"conditions": [asdict(condition) for condition in conditions]},
                summary=f"Filtered rows from {dataset.row_count} to {result.row_count}.",
                dataset_before=dataset.name,
                dataset_after=result.name,
            )
        )
        return result

    @staticmethod
    def _coerce_value(value: Any, series: pd.Series) -> Any:
        if value in (None, ""):
            return None
        try:
            if pd.api.types.is_numeric_dtype(series):
                return pd.to_numeric(value)
            if pd.api.types.is_datetime64_any_dtype(series):
                return pd.to_datetime(value)
        except (TypeError, ValueError) as exc:
            raise DataValidationError(f"Invalid filter value '{value}' for column '{series.name}'.") from exc
        return value
