from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class OperationRecord:
    operation_name: str
    parameters: dict[str, Any]
    summary: str
    dataset_before: str | None = None
    dataset_after: str | None = None
    timestamp: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_name": self.operation_name,
            "parameters": self.parameters,
            "summary": self.summary,
            "dataset_before": self.dataset_before,
            "dataset_after": self.dataset_after,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(slots=True)
class DatasetArtifact:
    name: str
    dataframe: pd.DataFrame
    source_name: str
    source_type: str
    sheet_name: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)
    operation_history: list[OperationRecord] = field(default_factory=list)

    def clone(self, *, dataframe: pd.DataFrame | None = None, name: str | None = None) -> "DatasetArtifact":
        return DatasetArtifact(
            name=name or self.name,
            dataframe=(dataframe.copy(deep=True) if dataframe is not None else self.dataframe.copy(deep=True)),
            source_name=self.source_name,
            source_type=self.source_type,
            sheet_name=self.sheet_name,
            created_at=utc_now(),
            metadata=dict(self.metadata),
            operation_history=list(self.operation_history),
        )

    @property
    def row_count(self) -> int:
        return int(self.dataframe.shape[0])

    @property
    def column_count(self) -> int:
        return int(self.dataframe.shape[1])


@dataclass(slots=True)
class GeneratedColumnSchema:
    name: str
    data_type: str
    allow_duplicates: bool = True
    primary_key: bool = False
    sample_value: Any = None
    min_value: float | int | None = None
    max_value: float | int | None = None
    categories: list[str] = field(default_factory=list)
    pattern: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    true_probability: float = 0.5


@dataclass(slots=True)
class DatasetGenerationRequest:
    dataset_name: str
    row_count: int
    columns: list[GeneratedColumnSchema]
    random_seed: int | None = None


@dataclass(slots=True)
class ProfileReport:
    dataset_name: str
    row_count: int
    column_count: int
    dtypes: dict[str, str]
    missing_values: dict[str, int]
    duplicate_rows: int
    summary_statistics: dict[str, dict[str, Any]]
    memory_usage_bytes: int


@dataclass(slots=True)
class FilterCondition:
    column: str
    operator: str
    value: Any = None
    secondary_value: Any = None


@dataclass(slots=True)
class ValidationIssue:
    severity: str
    category: str
    message: str
    columns: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ValidationReport:
    dataset_name: str
    issues: list[ValidationIssue]
    checked_at: datetime = field(default_factory=utc_now)

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)

    def summary(self) -> dict[str, int]:
        counts = {"error": 0, "warning": 0, "info": 0}
        for issue in self.issues:
            counts[issue.severity] = counts.get(issue.severity, 0) + 1
        return counts


@dataclass(slots=True)
class MergeConfiguration:
    left_dataset_name: str
    right_dataset_name: str
    left_keys: list[str]
    right_keys: list[str]
    join_type: str
    suffixes: tuple[str, str] = ("_left", "_right")
    align_key_types: bool = True


@dataclass(slots=True)
class ExportArtifact:
    file_name: str
    bytes_data: bytes
    mime_type: str
    row_count: int
    column_count: int


@dataclass(slots=True)
class GeneratedFileArtifact:
    file_name: str
    file_path: Path
    mime_type: str
    row_count: int
    column_count: int
    file_size_bytes: int
    chunk_count: int
