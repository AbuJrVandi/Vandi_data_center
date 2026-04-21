from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from .exceptions import DataLoadError
from .models import DatasetArtifact


class DataLoader:
    SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}

    def load_path(self, path: str | Path, *, load_all_sheets: bool = False) -> list[DatasetArtifact]:
        file_path = Path(path)
        if not file_path.exists():
            raise DataLoadError(f"File not found: {file_path}")
        with file_path.open("rb") as handle:
            return self.load_file_object(
                file_name=file_path.name,
                file_object=handle,
                load_all_sheets=load_all_sheets,
            )

    def load_file_object(
        self,
        *,
        file_name: str,
        file_object: BinaryIO,
        load_all_sheets: bool = False,
    ) -> list[DatasetArtifact]:
        if hasattr(file_object, "seek"):
            file_object.seek(0)
        extension = Path(file_name).suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise DataLoadError(f"Unsupported file type: {extension or 'unknown'}")

        payload = file_object.read()
        if not payload:
            raise DataLoadError(f"Uploaded file is empty: {file_name}")

        if extension == ".csv":
            dataframe = pd.read_csv(BytesIO(payload))
            self._validate_dataframe(dataframe, file_name)
            return [
                DatasetArtifact(
                    name=Path(file_name).stem,
                    dataframe=dataframe.copy(deep=True),
                    source_name=file_name,
                    source_type="csv",
                )
            ]

        workbook = pd.ExcelFile(BytesIO(payload))
        sheet_names = workbook.sheet_names
        if not sheet_names:
            raise DataLoadError(f"No worksheets found in workbook: {file_name}")

        target_sheets = sheet_names if load_all_sheets else [sheet_names[0]]
        datasets: list[DatasetArtifact] = []
        for sheet_name in target_sheets:
            dataframe = pd.read_excel(workbook, sheet_name=sheet_name)
            self._validate_dataframe(dataframe, f"{file_name}:{sheet_name}")
            datasets.append(
                DatasetArtifact(
                    name=f"{Path(file_name).stem}_{sheet_name}",
                    dataframe=dataframe.copy(deep=True),
                    source_name=file_name,
                    source_type="xlsx",
                    sheet_name=sheet_name,
                    metadata={"available_sheets": sheet_names},
                )
            )
        return datasets

    @staticmethod
    def _validate_dataframe(dataframe: pd.DataFrame, label: str) -> None:
        if dataframe.empty:
            raise DataLoadError(f"Dataset is empty: {label}")
        if dataframe.columns.empty:
            raise DataLoadError(f"Dataset has no columns: {label}")
        if dataframe.columns.duplicated().any():
            duplicated = dataframe.columns[dataframe.columns.duplicated()].tolist()
            raise DataLoadError(f"Dataset contains duplicate column names: {', '.join(duplicated)}")
