from __future__ import annotations

from io import BytesIO

import pandas as pd

from .exceptions import ExportError
from .models import DatasetArtifact, ExportArtifact


class DataExporter:
    def export(self, dataset: DatasetArtifact, *, file_format: str, columns: list[str] | None = None) -> ExportArtifact:
        dataframe = dataset.dataframe.copy(deep=True)
        if columns is not None:
            if not columns:
                raise ExportError("Select at least one column before exporting.")
            missing_columns = sorted(set(columns) - set(dataframe.columns))
            if missing_columns:
                raise ExportError(f"Export columns are missing from the dataset: {', '.join(missing_columns)}")
            dataframe = dataframe.loc[:, columns].copy(deep=True)
        if dataframe.columns.duplicated().any():
            raise ExportError("Cannot export a dataset with duplicate column names.")
        if dataframe.empty:
            raise ExportError("Cannot export an empty dataset.")

        file_format = file_format.lower()
        if file_format == "csv":
            payload = dataframe.to_csv(index=False).encode("utf-8")
            return ExportArtifact(
                file_name=f"{dataset.name}.csv",
                bytes_data=payload,
                mime_type="text/csv",
                row_count=int(dataframe.shape[0]),
                column_count=int(dataframe.shape[1]),
            )
        if file_format == "xlsx":
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                dataframe.to_excel(writer, index=False, sheet_name="processed_data")
            buffer.seek(0)
            return ExportArtifact(
                file_name=f"{dataset.name}.xlsx",
                bytes_data=buffer.getvalue(),
                mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                row_count=int(dataframe.shape[0]),
                column_count=int(dataframe.shape[1]),
            )
        raise ExportError(f"Unsupported export format: {file_format}")
