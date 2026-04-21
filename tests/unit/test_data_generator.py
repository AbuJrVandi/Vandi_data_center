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
    assert "Review" in set(dataframe["status"].astype(str))
    assert dataframe["event_date"].iloc[0] == pd.Timestamp("2024-06-15")
    assert bool(dataframe["approved"].iloc[0]) is True


def test_generator_rejects_row_count_above_limit() -> None:
    generator = DataGenerator()
    request = DatasetGenerationRequest(
        dataset_name="too_large",
        row_count=501,
        columns=[GeneratedColumnSchema(name="id", data_type="integer", primary_key=True)],
    )

    with pytest.raises(DataGenerationError, match="cannot exceed 500"):
        generator.generate(request)


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
