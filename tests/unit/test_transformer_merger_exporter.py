from __future__ import annotations

from io import BytesIO

import openpyxl
import pandas as pd

from astrodata_tool import DataExporter, DataMerger, DataTransformer, DatasetArtifact, MergeConfiguration


def test_transformer_derives_and_converts_columns(sample_dataset) -> None:
    transformer = DataTransformer()

    derived = transformer.derive_column(sample_dataset, new_column="value_x2", expression="value * 2")
    converted = transformer.convert_types(derived, {"event_date": "datetime64[ns]"})

    assert "value_x2" in converted.dataframe.columns
    assert str(converted.dataframe["event_date"].dtype).startswith("datetime64")


def test_merger_validates_and_merges_datasets() -> None:
    left = DatasetArtifact(
        name="orders",
        dataframe=pd.DataFrame({"customer_id": ["A", "B", "C"], "revenue": [10, 20, 30]}),
        source_name="orders.csv",
        source_type="csv",
    )
    right = DatasetArtifact(
        name="customers",
        dataframe=pd.DataFrame({"id": ["A", "B", "D"], "name": ["Ann", "Ben", "Dana"]}),
        source_name="customers.csv",
        source_type="csv",
    )
    merger = DataMerger()
    configuration = MergeConfiguration(
        left_dataset_name="orders",
        right_dataset_name="customers",
        left_keys=["customer_id"],
        right_keys=["id"],
        join_type="left",
    )

    merged, warnings = merger.merge(left, right, configuration)

    assert merged.row_count == 3
    assert "name" in merged.dataframe.columns
    assert warnings


def test_exporter_creates_csv_and_excel_outputs(sample_dataset) -> None:
    exporter = DataExporter()

    csv_artifact = exporter.export(sample_dataset, file_format="csv", columns=["id", "category"])
    excel_artifact = exporter.export(sample_dataset, file_format="xlsx", columns=["id", "category"])
    workbook = openpyxl.load_workbook(BytesIO(excel_artifact.bytes_data))
    worksheet = workbook["processed_data"]

    assert csv_artifact.file_name.endswith(".csv")
    assert excel_artifact.file_name.endswith(".xlsx")
    assert len(csv_artifact.bytes_data) > 0
    assert len(excel_artifact.bytes_data) > 0
    assert csv_artifact.column_count == 2
    assert "value" not in csv_artifact.bytes_data.decode("utf-8").splitlines()[0]
    assert [cell.value for cell in worksheet[1]] == ["id", "category"]
