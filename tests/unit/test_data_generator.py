from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from astrodata_tool import DataGenerator, DatasetGenerationRequest, GeneratedColumnSchema
from astrodata_tool.exceptions import DataGenerationError


def test_generator_builds_schema_driven_dataset_with_expected_types() -> None:
    generator = DataGenerator()
    request = DatasetGenerationRequest(
        dataset_name="synthetic_customers",
        row_count=40,
        random_seed=11,
        columns=[
            GeneratedColumnSchema(name="customer_id", data_type="integer", primary_key=True, min_value=1000, max_value=9999),
            GeneratedColumnSchema(name="email_address", data_type="string", allow_duplicates=False, pattern="email"),
            GeneratedColumnSchema(name="segment", data_type="category", categories=["Retail", "SMB", "Enterprise"]),
            GeneratedColumnSchema(name="credit_score", data_type="float", min_value=300, max_value=850),
            GeneratedColumnSchema(name="signup_date", data_type="date", start_date=date(2024, 1, 1), end_date=date(2024, 2, 29)),
            GeneratedColumnSchema(name="is_active", data_type="boolean", true_probability=0.7),
        ],
    )

    dataset = generator.generate(request)
    dataframe = dataset.dataframe

    assert dataset.source_type == "generated"
    assert dataframe.shape == (40, 6)
    assert dataframe["customer_id"].is_unique
    assert dataframe["email_address"].str.contains("@example.com").all()
    assert set(dataframe["segment"].astype(str).unique()) <= {"Retail", "SMB", "Enterprise"}
    assert pd.api.types.is_float_dtype(dataframe["credit_score"])
    assert dataframe["credit_score"].between(300, 850).all()
    assert pd.api.types.is_datetime64_any_dtype(dataframe["signup_date"])
    assert dataframe["signup_date"].between(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-29")).all()
    assert pd.api.types.is_bool_dtype(dataframe["is_active"])


def test_generator_uses_sample_values_to_anchor_output() -> None:
    generator = DataGenerator()
    request = DatasetGenerationRequest(
        dataset_name="sample_guided",
        row_count=6,
        random_seed=5,
        columns=[
            GeneratedColumnSchema(name="record_id", data_type="integer", primary_key=True, sample_value=7000),
            GeneratedColumnSchema(name="contact_email", data_type="string", allow_duplicates=False, pattern="email", sample_value="ops.team@acme.org"),
            GeneratedColumnSchema(name="phone_number", data_type="string", pattern="phone", sample_value="+23276888000"),
            GeneratedColumnSchema(name="age", data_type="integer", sample_value=34, min_value=21, max_value=60),
            GeneratedColumnSchema(name="gender", data_type="category", categories=["1=male", "2=female"], sample_value="2"),
            GeneratedColumnSchema(name="status", data_type="category", categories=["Pending", "Approved"], sample_value="Review"),
            GeneratedColumnSchema(name="event_date", data_type="date", sample_value=date(2024, 6, 15)),
            GeneratedColumnSchema(name="approved", data_type="boolean", sample_value=True),
        ],
    )

    dataset = generator.generate(request)
    dataframe = dataset.dataframe

    assert dataframe["record_id"].iloc[0] == 7000
    assert dataframe["contact_email"].iloc[0] == "ops.team@acme.org"
    assert dataframe["contact_email"].str.endswith("@acme.org").all()
    assert dataframe["phone_number"].iloc[0] == "+23276888000"
    assert dataframe["phone_number"].astype(str).str.startswith("+2327688").all()
    assert dataframe["age"].iloc[0] == 34
    assert dataframe["age"].between(21, 60).all()
    assert dataframe["gender"].iloc[0] == "female"
    assert set(dataframe["gender"].astype(str).unique()) <= {"male", "female"}
    assert "Review" in set(dataframe["status"].astype(str))
    assert dataframe["event_date"].iloc[0] == pd.Timestamp("2024-06-15")
    assert bool(dataframe["approved"].iloc[0]) is True


def test_generator_supports_coded_category_mappings_and_outputs_labels() -> None:
    generator = DataGenerator()
    request = DatasetGenerationRequest(
        dataset_name="coded_categories",
        row_count=20,
        random_seed=13,
        columns=[
            GeneratedColumnSchema(name="record_id", data_type="integer", primary_key=True),
            GeneratedColumnSchema(
                name="gender",
                data_type="category",
                categories=["1=male", "2=female"],
                sample_value="2",
            ),
        ],
    )

    dataset = generator.generate(request)
    dataframe = dataset.dataframe

    assert dataframe["gender"].iloc[0] == "female"
    assert set(dataframe["gender"].astype(str).unique()) <= {"male", "female"}


def test_generator_rejects_duplicate_category_mapping_codes() -> None:
    generator = DataGenerator()
    request = DatasetGenerationRequest(
        dataset_name="invalid_category_codes",
        row_count=5,
        columns=[
            GeneratedColumnSchema(
                name="gender",
                data_type="category",
                categories=["1=male", "1=female"],
            )
        ],
    )

    with pytest.raises(DataGenerationError, match="duplicate category code '1'"):
        generator.generate(request)


def test_generator_rejects_row_count_above_limit() -> None:
    generator = DataGenerator()
    request = DatasetGenerationRequest(
        dataset_name="too_large",
        row_count=501,
        columns=[GeneratedColumnSchema(name="id", data_type="integer", primary_key=True)],
    )

    with pytest.raises(DataGenerationError, match="cannot exceed 500"):
        generator.generate(request)


def test_generator_creates_large_csv_export_without_loading_full_dataset(tmp_path) -> None:
    generator = DataGenerator()
    request = DatasetGenerationRequest(
        dataset_name="large_export",
        row_count=2_000,
        random_seed=9,
        columns=[
            GeneratedColumnSchema(name="record_id", data_type="integer", primary_key=True),
            GeneratedColumnSchema(name="contact_email", data_type="string", allow_duplicates=False, pattern="email"),
            GeneratedColumnSchema(name="segment", data_type="category", categories=[f"segment_{index}" for index in range(2_000)], allow_duplicates=False),
            GeneratedColumnSchema(name="score", data_type="float", min_value=0, max_value=100),
        ],
    )

    artifact = generator.generate_large_csv(
        request,
        output_path=tmp_path / "large_export.csv",
        chunk_size=256,
    )

    exported = pd.read_csv(artifact.file_path)

    assert artifact.file_name == "large_export.csv"
    assert artifact.row_count == 2_000
    assert artifact.column_count == 4
    assert artifact.chunk_count == 8
    assert artifact.file_size_bytes > 0
    assert exported.shape == (2_000, 4)
    assert exported["record_id"].iloc[0] == 1
    assert exported["record_id"].iloc[-1] == 2_000
    assert exported["contact_email"].is_unique


def test_large_export_rejects_row_count_above_large_limit(tmp_path) -> None:
    generator = DataGenerator()
    request = DatasetGenerationRequest(
        dataset_name="too_large_for_export",
        row_count=1_000_001,
        columns=[GeneratedColumnSchema(name="id", data_type="integer", primary_key=True)],
    )

    with pytest.raises(DataGenerationError, match="cannot exceed 1000000"):
        generator.generate_large_csv(request, output_path=tmp_path / "too_large.csv")


def test_generator_rejects_non_unique_category_schema_when_duplicates_are_disabled() -> None:
    generator = DataGenerator()
    request = DatasetGenerationRequest(
        dataset_name="invalid_categories",
        row_count=5,
        columns=[
            GeneratedColumnSchema(
                name="status",
                data_type="category",
                allow_duplicates=False,
                categories=["new", "active"],
            )
        ],
    )

    with pytest.raises(DataGenerationError, match="needs at least 5 unique category values"):
        generator.generate(request)
