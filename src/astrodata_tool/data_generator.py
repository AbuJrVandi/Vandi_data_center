from __future__ import annotations

from dataclasses import asdict
from datetime import date, timedelta
import re

import numpy as np
import pandas as pd

try:
    from faker import Faker
except ModuleNotFoundError:  # pragma: no cover - exercised only when faker is unavailable
    Faker = None

from .exceptions import DataGenerationError
from .models import DatasetArtifact, DatasetGenerationRequest, GeneratedColumnSchema, OperationRecord


class DataGenerator:
    MAX_ROWS = 500
    SUPPORTED_TYPES = {"integer", "float", "string", "category", "date", "boolean"}
    SUPPORTED_PATTERNS = {"none", "email", "phone", "name", "company"}

    def generate(self, request: DatasetGenerationRequest) -> DatasetArtifact:
        self._validate_request(request)
        random_state = np.random.default_rng(request.random_seed)
        fake = self._create_faker(request.random_seed)
        columns = {
            column_schema.name: self._generate_column(column_schema, row_count=request.row_count, random_state=random_state, fake=fake)
            for column_schema in request.columns
        }
        dataframe = pd.DataFrame(columns)
        artifact = DatasetArtifact(
            name=request.dataset_name,
            dataframe=dataframe,
            source_name=f"{request.dataset_name}.generated",
            source_type="generated",
            metadata={
                "generator": "DataGenerator",
                "row_count": request.row_count,
                "random_seed": request.random_seed,
                "schema": [asdict(column) for column in request.columns],
            },
        )
        artifact.operation_history.append(
            OperationRecord(
                operation_name="generate_dataset",
                parameters={
                    "row_count": request.row_count,
                    "random_seed": request.random_seed,
                    "columns": [asdict(column) for column in request.columns],
                },
                summary=f"Generated dataset '{request.dataset_name}' with {request.row_count} rows and {len(request.columns)} columns.",
                dataset_after=request.dataset_name,
            )
        )
        return artifact

    def _validate_request(self, request: DatasetGenerationRequest) -> None:
        dataset_name = request.dataset_name.strip()
        if not dataset_name:
            raise DataGenerationError("Dataset name is required.")
        if request.row_count < 1:
            raise DataGenerationError("Row count must be at least 1.")
        if request.row_count > self.MAX_ROWS:
            raise DataGenerationError(f"Row count cannot exceed {self.MAX_ROWS}.")
        if not request.columns:
            raise DataGenerationError("Define at least one column in the schema.")

        column_names: list[str] = []
        for column in request.columns:
            name = column.name.strip()
            if not name:
                raise DataGenerationError("Every generated column must have a name.")
            if name in column_names:
                raise DataGenerationError(f"Duplicate column name in schema: {name}")
            column_names.append(name)

            if column.data_type not in self.SUPPORTED_TYPES:
                raise DataGenerationError(f"Unsupported generated column type: {column.data_type}")

            if column.data_type != "string" and column.pattern not in (None, "", "none"):
                raise DataGenerationError(f"Patterns are only supported for string columns: {name}")
            if column.data_type == "string" and (column.pattern or "none") not in self.SUPPORTED_PATTERNS:
                raise DataGenerationError(f"Unsupported string pattern for column '{name}': {column.pattern}")

            if column.data_type in {"integer", "float"}:
                if column.min_value is not None and column.max_value is not None and column.min_value > column.max_value:
                    raise DataGenerationError(f"Minimum value cannot exceed maximum value for column '{name}'.")
                if column.sample_value not in (None, ""):
                    try:
                        parsed_sample = int(column.sample_value) if column.data_type == "integer" else float(column.sample_value)
                    except (TypeError, ValueError) as exc:
                        raise DataGenerationError(f"Sample value for numeric column '{name}' is invalid.") from exc
                    if column.min_value is not None and parsed_sample < column.min_value:
                        raise DataGenerationError(f"Sample value for column '{name}' is below the minimum value.")
                    if column.max_value is not None and parsed_sample > column.max_value:
                        raise DataGenerationError(f"Sample value for column '{name}' is above the maximum value.")
            if column.data_type == "category":
                normalized_categories = self._normalized_categories(column)
                if not normalized_categories:
                    raise DataGenerationError(f"Category column '{name}' requires at least one category value.")
                if self._requires_unique_values(column) and len(set(normalized_categories)) < request.row_count:
                    raise DataGenerationError(
                        f"Category column '{name}' needs at least {request.row_count} unique category values when duplicates are disabled."
                    )
            if column.data_type == "date":
                sample_date = self._coerce_sample_date(column.sample_value, name=name)
                start_date = column.start_date or ((sample_date - timedelta(days=45)) if sample_date else (date.today() - timedelta(days=365)))
                end_date = column.end_date or ((sample_date + timedelta(days=45)) if sample_date else date.today())
                if start_date > end_date:
                    raise DataGenerationError(f"Date range is invalid for column '{name}'.")
                if sample_date is not None and not (start_date <= sample_date <= end_date):
                    raise DataGenerationError(f"Sample date for column '{name}' must fall within the date range.")
                if self._requires_unique_values(column):
                    available_days = (end_date - start_date).days + 1
                    if available_days < request.row_count:
                        raise DataGenerationError(
                            f"Date column '{name}' needs at least {request.row_count} distinct dates when duplicates are disabled."
                        )
            if column.data_type == "boolean" and self._requires_unique_values(column) and request.row_count > 2:
                raise DataGenerationError(
                    f"Boolean column '{name}' cannot generate more than 2 rows when duplicates are disabled."
                )
            if column.data_type == "boolean" and not 0 <= float(column.true_probability) <= 1:
                raise DataGenerationError(f"True probability for '{name}' must be between 0 and 1.")
            if column.data_type == "integer" and self._requires_unique_values(column):
                min_value = int(column.min_value) if column.min_value is not None else 1
                max_value = int(column.max_value) if column.max_value is not None else max(request.row_count * 10, min_value + request.row_count)
                if (max_value - min_value + 1) < request.row_count:
                    raise DataGenerationError(
                        f"Integer column '{name}' does not have enough unique values in the requested range."
                    )
            if column.data_type == "float" and self._requires_unique_values(column):
                min_value = float(column.min_value) if column.min_value is not None else 0.0
                max_value = float(column.max_value) if column.max_value is not None else max(float(request.row_count), 100.0)
                if request.row_count > 1 and min_value == max_value:
                    raise DataGenerationError(
                        f"Float column '{name}' requires a wider range when duplicates are disabled."
                    )

    def _generate_column(
        self,
        column: GeneratedColumnSchema,
        *,
        row_count: int,
        random_state: np.random.Generator,
        fake,
    ) -> pd.Series:
        if column.data_type == "integer":
            return self._generate_integer_series(column, row_count=row_count, random_state=random_state)
        if column.data_type == "float":
            return self._generate_float_series(column, row_count=row_count, random_state=random_state)
        if column.data_type == "string":
            return self._generate_string_series(column, row_count=row_count, random_state=random_state, fake=fake)
        if column.data_type == "category":
            return self._generate_category_series(column, row_count=row_count, random_state=random_state)
        if column.data_type == "date":
            return self._generate_date_series(column, row_count=row_count, random_state=random_state)
        return self._generate_boolean_series(column, row_count=row_count, random_state=random_state)

    def _generate_integer_series(self, column: GeneratedColumnSchema, *, row_count: int, random_state: np.random.Generator) -> pd.Series:
        sample_value = int(column.sample_value) if column.sample_value not in (None, "") else None
        if column.min_value is not None:
            min_value = int(column.min_value)
        elif sample_value is not None:
            min_value = sample_value if self._requires_unique_values(column) else max(0, sample_value - max(row_count, 10))
        else:
            min_value = 1
        if column.max_value is not None:
            max_value = int(column.max_value)
        elif sample_value is not None:
            max_value = max(sample_value + max(row_count * 2, 25), min_value + row_count + 4)
        else:
            max_value = max(row_count * 10, min_value + row_count + 24)
        if self._requires_unique_values(column):
            if column.primary_key or column.name.lower().endswith("id") or sample_value is not None:
                start_value = sample_value if sample_value is not None else min_value
                values = np.arange(start_value, start_value + row_count, dtype=np.int64)
            else:
                values = random_state.choice(np.arange(min_value, max_value + 1, dtype=np.int64), size=row_count, replace=False)
        else:
            values = random_state.integers(min_value, max_value + 1, size=row_count, dtype=np.int64)
        return pd.Series(values, dtype="int64")

    def _generate_float_series(self, column: GeneratedColumnSchema, *, row_count: int, random_state: np.random.Generator) -> pd.Series:
        sample_value = float(column.sample_value) if column.sample_value not in (None, "") else None
        if column.min_value is not None:
            min_value = float(column.min_value)
        elif sample_value is not None:
            min_value = max(0.0, sample_value - max(float(row_count * 2), 15.0))
        else:
            min_value = 0.0
        if column.max_value is not None:
            max_value = float(column.max_value)
        elif sample_value is not None:
            max_value = sample_value + max(float(row_count * 2), 15.0)
        else:
            max_value = max(float(row_count * 12), 250.0)
        if self._requires_unique_values(column):
            if row_count == 1:
                values = np.array([sample_value if sample_value is not None else min_value], dtype=float)
            else:
                values = np.linspace(min_value, max_value, num=row_count, dtype=float)
                random_state.shuffle(values)
        else:
            if sample_value is not None:
                deviation = max((max_value - min_value) / 6, 1.0)
                values = random_state.normal(loc=sample_value, scale=deviation, size=row_count)
                values = np.clip(values, min_value, max_value)
            else:
                values = random_state.uniform(min_value, max_value, size=row_count)
            values = np.round(values, 4)
        return pd.Series(values, dtype="float64")

    def _generate_string_series(
        self,
        column: GeneratedColumnSchema,
        *,
        row_count: int,
        random_state: np.random.Generator,
        fake,
    ) -> pd.Series:
        pattern = self._resolve_pattern(column)
        values = [
            self._generate_string_value(column, pattern, index, fake=fake, random_state=random_state)
            for index in range(row_count)
        ]
        if self._requires_unique_values(column):
            values = [self._force_unique_string(column.name, pattern, value, index) for index, value in enumerate(values)]
        return pd.Series(values, dtype="string")

    def _generate_category_series(self, column: GeneratedColumnSchema, *, row_count: int, random_state: np.random.Generator) -> pd.Series:
        categories = self._normalized_categories(column)
        if self._requires_unique_values(column):
            values = random_state.choice(np.array(categories, dtype=object), size=row_count, replace=False)
        else:
            values = random_state.choice(np.array(categories, dtype=object), size=row_count, replace=True)
            if column.sample_value not in (None, ""):
                values[0] = str(column.sample_value).strip()
        return pd.Series(pd.Categorical(values, categories=categories))

    def _generate_date_series(self, column: GeneratedColumnSchema, *, row_count: int, random_state: np.random.Generator) -> pd.Series:
        sample_date = self._coerce_sample_date(column.sample_value, name=column.name)
        start_date = pd.Timestamp(column.start_date or ((sample_date - timedelta(days=45)) if sample_date else (date.today() - timedelta(days=365))))
        end_date = pd.Timestamp(column.end_date or ((sample_date + timedelta(days=45)) if sample_date else date.today()))
        day_count = (end_date - start_date).days + 1
        if self._requires_unique_values(column):
            offsets = random_state.choice(np.arange(day_count, dtype=np.int64), size=row_count, replace=False)
        else:
            if sample_date is not None:
                center_offset = (pd.Timestamp(sample_date) - start_date).days
                offsets = np.round(random_state.normal(loc=center_offset, scale=max(day_count / 8, 1), size=row_count)).astype(np.int64)
                offsets = np.clip(offsets, 0, day_count - 1)
            else:
                offsets = random_state.integers(0, day_count, size=row_count, dtype=np.int64)
        values = pd.Series(pd.to_datetime(start_date + pd.to_timedelta(offsets, unit="D")))
        if sample_date is not None and row_count > 0:
            values.iloc[0] = pd.Timestamp(sample_date)
        return values

    def _generate_boolean_series(self, column: GeneratedColumnSchema, *, row_count: int, random_state: np.random.Generator) -> pd.Series:
        if self._requires_unique_values(column):
            values = np.array([True, False][:row_count], dtype=bool)
            random_state.shuffle(values)
            return pd.Series(values, dtype="bool")
        values = random_state.random(row_count) < float(column.true_probability)
        if column.sample_value not in (None, "") and row_count > 0:
            values[0] = bool(column.sample_value)
        return pd.Series(values, dtype="bool")

    @staticmethod
    def _requires_unique_values(column: GeneratedColumnSchema) -> bool:
        return column.primary_key or not column.allow_duplicates

    def _resolve_pattern(self, column: GeneratedColumnSchema) -> str:
        if column.pattern and column.pattern != "none":
            return column.pattern
        lowered_name = column.name.lower()
        if "email" in lowered_name:
            return "email"
        if "phone" in lowered_name or "mobile" in lowered_name:
            return "phone"
        if "company" in lowered_name or "organization" in lowered_name:
            return "company"
        if "name" in lowered_name:
            return "name"
        return "none"

    def _generate_string_value(
        self,
        column: GeneratedColumnSchema,
        pattern: str,
        index: int,
        *,
        fake,
        random_state: np.random.Generator,
    ) -> str:
        sample_value = str(column.sample_value).strip() if column.sample_value not in (None, "") else ""
        if sample_value:
            if "@" in sample_value or pattern == "email":
                return self._email_from_sample(sample_value, column.name, index, random_state=random_state)
            if pattern == "phone":
                return self._phone_from_sample(sample_value, index, random_state=random_state)
            if pattern in {"name", "company"}:
                return sample_value
            if pattern == "none":
                return self._text_from_sample(sample_value, index, random_state=random_state)
        if pattern == "email":
            return fake.email() if fake is not None else f"user{random_state.integers(1000, 999999)}@example.com"
        if pattern == "phone":
            return fake.phone_number() if fake is not None else f"+1-202-555-{random_state.integers(1000, 9999):04d}"
        if pattern == "name":
            return fake.name() if fake is not None else self._fallback_name(random_state)
        if pattern == "company":
            return fake.company() if fake is not None else self._fallback_company(random_state)
        if fake is not None:
            return fake.sentence(nb_words=3).replace(".", "")
        column_slug = self._slugify(column_name)
        return f"{column_slug}_value_{index + 1}"

    def _force_unique_string(self, column_name: str, pattern: str, value: str, index: int) -> str:
        if index == 0 and value:
            return value
        sequence = index + 1
        if pattern == "email":
            if "@" in value:
                local_part, domain = value.split("@", 1)
                slug = self._slugify(local_part) or self._slugify(column_name) or "user"
                return f"{slug}{sequence:04d}@{domain}"
            slug = self._slugify(column_name) or "user"
            return f"{slug}{sequence:04d}@example.com"
        if pattern == "phone":
            return f"+1-202-555-{sequence + 999:04d}"
        if pattern in {"name", "company"}:
            return f"{value} {sequence}"
        slug = self._slugify(column_name) or "value"
        return f"{slug.upper()}-{sequence:04d}"

    @staticmethod
    def _normalized_categories(column: GeneratedColumnSchema) -> list[str]:
        categories = [value.strip() for value in column.categories if str(value).strip()]
        if column.sample_value not in (None, ""):
            sample_category = str(column.sample_value).strip()
            if sample_category and sample_category not in categories:
                categories.insert(0, sample_category)
        return categories

    @staticmethod
    def _coerce_sample_date(value: object, *, name: str) -> date | None:
        if value in (None, ""):
            return None
        if isinstance(value, date):
            return value
        try:
            return pd.to_datetime(value).date()
        except (TypeError, ValueError) as exc:
            raise DataGenerationError(f"Sample date for column '{name}' is invalid.") from exc

    def _email_from_sample(self, sample_value: str, column_name: str, index: int, *, random_state: np.random.Generator) -> str:
        if "@" not in sample_value:
            slug = self._slugify(sample_value or column_name) or "user"
            return f"{slug}{index + 1:04d}@example.com"
        local_part, domain = sample_value.split("@", 1)
        local_slug = self._slugify(local_part) or self._slugify(column_name) or "user"
        domain = domain.strip() or "example.com"
        if index == 0:
            return sample_value
        return f"{local_slug}{index + 1:04d}@{domain}"

    def _phone_from_sample(self, sample_value: str, index: int, *, random_state: np.random.Generator) -> str:
        digits = "".join(character for character in sample_value if character.isdigit())
        if len(digits) >= 7:
            prefix = digits[:-4] or "1202555"
            return f"+{prefix}{index + 1000:04d}" if sample_value.startswith("+") else f"{prefix}{index + 1000:04d}"
        return f"+1-202-555-{random_state.integers(1000, 9999):04d}"

    def _text_from_sample(self, sample_value: str, index: int, *, random_state: np.random.Generator) -> str:
        if index == 0:
            return sample_value
        if re.fullmatch(r"[A-Za-z_ -]+", sample_value):
            return f"{sample_value} {index + 1}"
        return f"{sample_value}_{index + 1}"

    @staticmethod
    def _slugify(value: str) -> str:
        return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")

    @staticmethod
    def _create_faker(seed: int | None):
        if Faker is None:
            return None
        fake = Faker()
        if seed is not None:
            fake.seed_instance(seed)
        return fake

    @staticmethod
    def _fallback_name(random_state: np.random.Generator) -> str:
        first_names = np.array(["Ava", "Noah", "Mia", "Liam", "Ella", "Mason", "Sophia", "Lucas"], dtype=object)
        last_names = np.array(["Stone", "Brooks", "Carter", "Hill", "Wright", "Turner", "Cole", "Hayes"], dtype=object)
        return f"{random_state.choice(first_names)} {random_state.choice(last_names)}"

    @staticmethod
    def _fallback_company(random_state: np.random.Generator) -> str:
        descriptors = np.array(["North", "Blue", "Prime", "Nova", "Summit", "Urban", "Vertex", "Bright"], dtype=object)
        nouns = np.array(["Systems", "Labs", "Logistics", "Analytics", "Works", "Group", "Partners", "Ventures"], dtype=object)
        return f"{random_state.choice(descriptors)} {random_state.choice(nouns)}"
