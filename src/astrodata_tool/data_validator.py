from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .models import DatasetArtifact, ValidationIssue, ValidationReport


class DataValidator:
    def validate(
        self,
        dataset: DatasetArtifact,
        *,
        expected_types: Mapping[str, str] | None = None,
        range_rules: Mapping[str, dict[str, Any]] | None = None,
        reference_schema: Mapping[str, str] | None = None,
        outlier_columns: list[str] | None = None,
        outlier_z_threshold: float = 3.0,
    ) -> ValidationReport:
        issues: list[ValidationIssue] = []
        issues.extend(self.validate_types(dataset, expected_types or {}))
        issues.extend(self.validate_ranges(dataset, range_rules or {}))
        issues.extend(self.check_schema_consistency(dataset, reference_schema or {}))
        issues.extend(self.detect_outliers(dataset, columns=outlier_columns, z_threshold=outlier_z_threshold))
        return ValidationReport(dataset_name=dataset.name, issues=issues)

    def validate_types(self, dataset: DatasetArtifact, expected_types: Mapping[str, str]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for column, expected in expected_types.items():
            if column not in dataset.dataframe.columns:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        category="type_validation",
                        message=f"Expected column '{column}' is missing.",
                        columns=[column],
                    )
                )
                continue
            actual = str(dataset.dataframe[column].dtype)
            if actual != expected:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        category="type_validation",
                        message=f"Column '{column}' has dtype '{actual}' instead of '{expected}'.",
                        columns=[column],
                        details={"expected": expected, "actual": actual},
                    )
                )
        return issues

    def validate_ranges(self, dataset: DatasetArtifact, range_rules: Mapping[str, dict[str, Any]]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for column, rule in range_rules.items():
            if column not in dataset.dataframe.columns:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        category="range_validation",
                        message=f"Range validation column '{column}' is missing.",
                        columns=[column],
                    )
                )
                continue
            original_series = dataset.dataframe[column]
            series = pd.to_numeric(original_series, errors="coerce")
            minimum = rule.get("min")
            maximum = rule.get("max")
            invalid_mask = pd.Series(False, index=series.index)
            if minimum is not None:
                invalid_mask |= series < minimum
            if maximum is not None:
                invalid_mask |= series > maximum
            coercion_failures = int((original_series.notna() & series.isna()).sum())
            invalid_count = int(invalid_mask.fillna(False).sum())
            if coercion_failures:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        category="range_validation",
                        message=f"Column '{column}' contains {coercion_failures} non-numeric values that could not be range-checked.",
                        columns=[column],
                        details={"coercion_failures": coercion_failures},
                    )
                )
            if invalid_count:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        category="range_validation",
                        message=f"Column '{column}' has {invalid_count} values outside the accepted range.",
                        columns=[column],
                        details={"min": minimum, "max": maximum, "invalid_count": invalid_count},
                    )
                )
        return issues

    def check_schema_consistency(self, dataset: DatasetArtifact, reference_schema: Mapping[str, str]) -> list[ValidationIssue]:
        if not reference_schema:
            return []
        issues: list[ValidationIssue] = []
        current_schema = {column: str(dtype) for column, dtype in dataset.dataframe.dtypes.items()}
        missing_columns = sorted(set(reference_schema) - set(current_schema))
        extra_columns = sorted(set(current_schema) - set(reference_schema))
        mismatched_columns = sorted(
            column for column in set(reference_schema).intersection(current_schema) if reference_schema[column] != current_schema[column]
        )

        if missing_columns:
            issues.append(
                ValidationIssue(
                    severity="error",
                    category="schema_validation",
                    message=f"Missing schema columns: {', '.join(missing_columns)}.",
                    columns=missing_columns,
                )
            )
        if extra_columns:
            issues.append(
                ValidationIssue(
                    severity="info",
                    category="schema_validation",
                    message=f"Additional columns present: {', '.join(extra_columns)}.",
                    columns=extra_columns,
                )
            )
        if mismatched_columns:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    category="schema_validation",
                    message=f"Columns with unexpected dtypes: {', '.join(mismatched_columns)}.",
                    columns=mismatched_columns,
                    details={column: {"expected": reference_schema[column], "actual": current_schema[column]} for column in mismatched_columns},
                )
            )
        return issues

    def detect_outliers(
        self,
        dataset: DatasetArtifact,
        *,
        columns: list[str] | None = None,
        z_threshold: float = 3.0,
    ) -> list[ValidationIssue]:
        numeric_frame = dataset.dataframe.select_dtypes(include="number")
        target_columns = columns or list(numeric_frame.columns)
        issues: list[ValidationIssue] = []
        for column in target_columns:
            if column not in numeric_frame.columns:
                continue
            series = numeric_frame[column].dropna()
            if series.empty or series.std(ddof=0) == 0:
                continue
            z_scores = np.abs((series - series.mean()) / series.std(ddof=0))
            outlier_count = int((z_scores > z_threshold).sum())
            if outlier_count:
                issues.append(
                    ValidationIssue(
                        severity="info",
                        category="outlier_detection",
                        message=f"Detected {outlier_count} statistical outliers in '{column}' using z-score > {z_threshold}.",
                        columns=[column],
                        details={"outlier_count": outlier_count, "z_threshold": z_threshold},
                    )
                )
        return issues
