from __future__ import annotations

from io import BytesIO

import pandas as pd

from astrodata_tool import AutomationEngine, DatasetGenerationRequest, FilterCondition, GeneratedColumnSchema, MergeConfiguration


def test_engine_executes_end_to_end_workflow(tmp_path) -> None:
    engine = AutomationEngine(log_path=tmp_path / "operations.jsonl")

    messy_csv = BytesIO(
        b"id,value,category\n1,10,a\n2,,b\n2,,b\n4,400,c\n"
    )
    customer_csv = BytesIO(
        b"id,name\n1,Alice\n2,Bob\n5,Eve\n"
    )

    loaded = engine.load_file(file_name="messy.csv", file_object=messy_csv)
    customers = engine.load_file(file_name="customers.csv", file_object=customer_csv)
    dataset = loaded[0]

    cleaned = engine.clean_missing(dataset, method="median", columns=["value"])
    deduplicated = engine.deduplicate(cleaned, subset=["id", "category"])
    typed = engine.convert_types(deduplicated, {"id": "string"})
    derived = engine.derive_column(typed, new_column="value_band", expression="value * 2")
    filtered = engine.filter_rows(derived, [FilterCondition(column="value", operator=">=", value="10")])
    report = engine.validate(filtered, expected_types={"id": "string"})

    customer_dataset = engine.convert_types(customers[0], {"id": "string"})
    merged, warnings = engine.merge(
        filtered,
        customer_dataset,
        MergeConfiguration(
            left_dataset_name=filtered.name,
            right_dataset_name=customer_dataset.name,
            left_keys=["id"],
            right_keys=["id"],
            join_type="left",
        ),
    )
    export_artifact = engine.export(merged, file_format="csv")

    assert not report.has_errors
    assert merged.row_count == 3
    assert "name" in merged.dataframe.columns
    assert warnings
    assert export_artifact.bytes_data.startswith(b"id,value,category")
    assert len(engine.logger.list_records()) >= 7


def test_engine_generates_then_cleans_and_exports_dataset(tmp_path) -> None:
    engine = AutomationEngine(log_path=tmp_path / "operations.jsonl")
    request = DatasetGenerationRequest(
        dataset_name="synthetic_orders",
        row_count=25,
        random_seed=7,
        columns=[
            GeneratedColumnSchema(name="order_id", data_type="integer", primary_key=True, min_value=1000, max_value=9999),
            GeneratedColumnSchema(name="customer_email", data_type="string", allow_duplicates=False, pattern="email"),
            GeneratedColumnSchema(name="amount", data_type="float", min_value=25, max_value=500),
            GeneratedColumnSchema(name="channel", data_type="category", categories=["Web", "Retail", "Partner"]),
        ],
    )

    generated = engine.generate_dataset(request)
    cleaned = engine.clean_missing(generated, method="mode", columns=["channel"])
    export_artifact = engine.export(cleaned, file_format="xlsx")

    assert generated.row_count == 25
    assert generated.dataframe["order_id"].is_unique
    assert cleaned.row_count == 25
    assert export_artifact.file_name.endswith(".xlsx")
    assert export_artifact.row_count == 25
    assert len(engine.logger.list_records()) >= 3
