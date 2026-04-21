from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from .data_cleaner import DataCleaner
from .data_generator import DataGenerator
from .data_exporter import DataExporter
from .data_filter import DataFilter
from .data_loader import DataLoader
from .data_merger import DataMerger
from .data_profiler import DataProfiler
from .data_transformer import DataTransformer
from .data_validator import DataValidator
from .logger import OperationLogger
from .models import DatasetArtifact, DatasetGenerationRequest, ExportArtifact, FilterCondition, GeneratedFileArtifact, MergeConfiguration, OperationRecord, ProfileReport, ValidationReport


class AutomationEngine:
    def __init__(self, log_path: str | Path = "logs/operations.jsonl") -> None:
        self.loader = DataLoader()
        self.profiler = DataProfiler()
        self.cleaner = DataCleaner()
        self.generator = DataGenerator()
        self.filterer = DataFilter()
        self.validator = DataValidator()
        self.transformer = DataTransformer()
        self.merger = DataMerger()
        self.exporter = DataExporter()
        self.logger = OperationLogger(log_path)

    def load_file(self, *, file_name: str, file_object: BinaryIO, load_all_sheets: bool = False) -> list[DatasetArtifact]:
        return self.loader.load_file_object(file_name=file_name, file_object=file_object, load_all_sheets=load_all_sheets)

    def profile(self, dataset: DatasetArtifact) -> ProfileReport:
        return self.profiler.profile(dataset)

    def generate_dataset(self, request: DatasetGenerationRequest) -> DatasetArtifact:
        artifact = self.generator.generate(request)
        self.logger.record(artifact.operation_history[-1])
        return artifact

    def generate_large_dataset_export(
        self,
        request: DatasetGenerationRequest,
        *,
        output_path: str | Path,
        chunk_size: int = 50_000,
    ) -> GeneratedFileArtifact:
        artifact = self.generator.generate_large_csv(request, output_path=output_path, chunk_size=chunk_size)
        self.logger.record(
            OperationRecord(
                operation_name="generate_large_dataset_export",
                parameters={
                    "row_count": request.row_count,
                    "chunk_size": chunk_size,
                    "file_name": artifact.file_name,
                    "columns": [column.name for column in request.columns],
                },
                summary=(
                    f"Prepared large dataset export '{artifact.file_name}' with "
                    f"{artifact.row_count} rows, {artifact.column_count} columns, and {artifact.chunk_count} chunk(s)."
                ),
                dataset_after=request.dataset_name,
            )
        )
        return artifact

    def clean_missing(self, dataset: DatasetArtifact, *, method: str, columns: list[str] | None = None, fill_value: str | None = None) -> DatasetArtifact:
        result = self.cleaner.handle_missing_values(dataset, method=method, columns=columns, fill_value=fill_value)
        self.logger.record(result.operation_history[-1])
        return result

    def deduplicate(self, dataset: DatasetArtifact, *, subset: list[str] | None = None) -> DatasetArtifact:
        result = self.cleaner.remove_duplicates(dataset, subset=subset)
        self.logger.record(result.operation_history[-1])
        return result

    def filter_rows(self, dataset: DatasetArtifact, conditions: list[FilterCondition]) -> DatasetArtifact:
        result = self.filterer.apply_conditions(dataset, conditions)
        self.logger.record(result.operation_history[-1])
        return result

    def validate(self, dataset: DatasetArtifact, **kwargs) -> ValidationReport:
        report = self.validator.validate(dataset, **kwargs)
        self.logger.record(
            OperationRecord(
                operation_name="validate_dataset",
                parameters=kwargs,
                summary=f"Validation completed with summary {report.summary()}.",
                dataset_before=dataset.name,
                dataset_after=dataset.name,
            )
        )
        return report

    def select_columns(self, dataset: DatasetArtifact, columns: list[str]) -> DatasetArtifact:
        result = self.transformer.select_columns(dataset, columns)
        self.logger.record(result.operation_history[-1])
        return result

    def rename_columns(self, dataset: DatasetArtifact, rename_map: dict[str, str]) -> DatasetArtifact:
        result = self.transformer.rename_columns(dataset, rename_map)
        self.logger.record(result.operation_history[-1])
        return result

    def derive_column(self, dataset: DatasetArtifact, *, new_column: str, expression: str) -> DatasetArtifact:
        result = self.transformer.derive_column(dataset, new_column=new_column, expression=expression)
        self.logger.record(result.operation_history[-1])
        return result

    def convert_types(self, dataset: DatasetArtifact, conversions: dict[str, str]) -> DatasetArtifact:
        result = self.transformer.convert_types(dataset, conversions)
        self.logger.record(result.operation_history[-1])
        return result

    def merge(self, left_dataset: DatasetArtifact, right_dataset: DatasetArtifact, configuration: MergeConfiguration) -> tuple[DatasetArtifact, list[str]]:
        result, warnings = self.merger.merge(left_dataset, right_dataset, configuration)
        self.logger.record(result.operation_history[-1])
        return result, warnings

    def export(self, dataset: DatasetArtifact, *, file_format: str, columns: list[str] | None = None) -> ExportArtifact:
        artifact = self.exporter.export(dataset, file_format=file_format, columns=columns)
        self.logger.record(
            OperationRecord(
                operation_name="export_dataset",
                parameters={"file_format": file_format, "file_name": artifact.file_name, "columns": columns},
                summary=f"Prepared export with {artifact.row_count} rows and {artifact.column_count} columns.",
                dataset_before=dataset.name,
                dataset_after=dataset.name,
            )
        )
        return artifact
