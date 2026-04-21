from __future__ import annotations

from datetime import date, timedelta
import inspect
from io import BytesIO
from pathlib import Path
import re
import sys

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from astrodata_tool import (
    AutomationEngine,
    DatasetArtifact,
    DatasetGenerationRequest,
    FilterCondition,
    GeneratedColumnSchema,
    MergeConfiguration,
)
from astrodata_tool.analytics import (
    ChartSpec,
    SUPPORTED_AGGREGATIONS,
    SUPPORTED_STATISTICS,
    build_chart_artifact,
    build_statistics_table,
    build_word_copy_component,
    recommend_chart_specs,
    statistics_table_to_html,
)
from astrodata_tool.exceptions import DataAutomationError


st.set_page_config(
    page_title="AstroData Automation Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_CSS = """
<style>
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(73,107,255,0.18), transparent 26%),
            radial-gradient(circle at bottom left, rgba(33,214,151,0.10), transparent 24%),
            linear-gradient(180deg, #08101f 0%, #07111d 100%);
    }
    .app-shell {
        padding: 0.75rem 0 1rem 0;
    }
    .hero {
        border: 1px solid rgba(255,255,255,0.08);
        background: linear-gradient(180deg, rgba(13,21,41,0.94), rgba(10,18,34,0.96));
        border-radius: 18px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 18px 50px rgba(0,0,0,0.24);
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
        line-height: 1.2;
    }
    .hero p {
        margin: 0.55rem 0 0 0;
        color: #9fb0d0;
        max-width: 60rem;
    }
    .metric-card {
        border: 1px solid rgba(255,255,255,0.07);
        background: rgba(16, 26, 48, 0.92);
        border-radius: 16px;
        padding: 1rem 1rem 0.85rem 1rem;
        min-height: 120px;
    }
    .metric-label {
        color: #91a2c1;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .metric-value {
        color: #f4f7ff;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }
    .metric-detail {
        color: #93a4c4;
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }
    .section-card {
        border: 1px solid rgba(255,255,255,0.07);
        background: rgba(11, 19, 36, 0.92);
        border-radius: 16px;
        padding: 1rem 1rem 1.1rem 1rem;
        margin-bottom: 1rem;
    }
    .section-title {
        margin: 0;
        font-size: 1.45rem;
        color: #f5f8ff;
    }
    .section-lead {
        margin: 0.35rem 0 0 0;
        color: #9fb0d0;
        max-width: 56rem;
    }
    .small-note {
        color: #91a2c1;
        font-size: 0.92rem;
    }
</style>
"""


def init_state() -> None:
    if "engine" not in st.session_state or _engine_requires_refresh(st.session_state.engine):
        st.session_state.engine = AutomationEngine()
    if "datasets" not in st.session_state:
        st.session_state.datasets = {}
    if "input_mode" not in st.session_state:
        st.session_state.input_mode = "Upload Dataset"
    if "generator_column_count" not in st.session_state:
        st.session_state.generator_column_count = 4
    if "mode" not in st.session_state:
        st.session_state.mode = None
    if "selected_dataset" not in st.session_state:
        st.session_state.selected_dataset = None
    if "last_validation" not in st.session_state:
        st.session_state.last_validation = None
    if "merge_preview" not in st.session_state:
        st.session_state.merge_preview = None
    if "export_artifact" not in st.session_state:
        st.session_state.export_artifact = None
    if "export_context" not in st.session_state:
        st.session_state.export_context = None


def dataset_names() -> list[str]:
    return list(st.session_state.datasets.keys())


def _engine_requires_refresh(engine: object) -> bool:
    export_method = getattr(engine, "export", None)
    generate_method = getattr(engine, "generate_dataset", None)
    if export_method is None or generate_method is None:
        return True
    try:
        return "columns" not in inspect.signature(export_method).parameters
    except (TypeError, ValueError):
        return True


def get_selected_dataset() -> DatasetArtifact | None:
    selected_name = st.session_state.selected_dataset
    if selected_name and selected_name in st.session_state.datasets:
        return st.session_state.datasets[selected_name]
    names = dataset_names()
    if names:
        st.session_state.selected_dataset = names[0]
        return st.session_state.datasets[names[0]]
    return None


def replace_dataset(old_key: str, new_dataset: DatasetArtifact) -> None:
    datasets = st.session_state.datasets
    datasets.pop(old_key, None)
    datasets[new_dataset.name] = new_dataset
    st.session_state.selected_dataset = new_dataset.name
    st.session_state.export_artifact = None
    st.session_state.export_context = None


def add_dataset(dataset: DatasetArtifact) -> None:
    st.session_state.datasets[dataset.name] = dataset
    st.session_state.selected_dataset = dataset.name
    st.session_state.export_artifact = None
    st.session_state.export_context = None


def render_metric_card(label: str, value: str, detail: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="section-card">
            <h2 class="section-title">{title}</h2>
            <p class="section-lead">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_preview(dataset: DatasetArtifact, *, title: str = "Data Preview", rows: int = 15) -> None:
    st.markdown(f"### {title}")
    st.caption(f"Current artifact: `{dataset.name}` | Source: `{dataset.source_name}`")
    st.dataframe(dataset.dataframe.head(rows), width="stretch", hide_index=True)


def render_profile(dataset: DatasetArtifact) -> None:
    profile = st.session_state.engine.profile(dataset)
    missing_total = sum(profile.missing_values.values())
    duplicate_rows = profile.duplicate_rows
    metric_columns = st.columns(4)
    with metric_columns[0]:
        render_metric_card("Rows", f"{profile.row_count:,}", "Active row count")
    with metric_columns[1]:
        render_metric_card("Columns", f"{profile.column_count:,}", "Detected schema width")
    with metric_columns[2]:
        render_metric_card("Missing Values", f"{missing_total:,}", "Explicitly tracked null cells")
    with metric_columns[3]:
        render_metric_card("Duplicates", f"{duplicate_rows:,}", "Exact duplicate rows")

    summary_columns = st.columns([1.15, 1])
    with summary_columns[0]:
        st.markdown("### Profiling Summary")
        schema_frame = pd.DataFrame(
            {
                "column": list(profile.dtypes.keys()),
                "dtype": list(profile.dtypes.values()),
                "missing_values": [profile.missing_values[column] for column in profile.dtypes],
            }
        )
        st.dataframe(schema_frame, width="stretch", hide_index=True)
    with summary_columns[1]:
        st.markdown("### Summary Statistics")
        summary_frame = pd.DataFrame(profile.summary_statistics).T.reset_index().rename(columns={"index": "statistic"})
        st.dataframe(summary_frame, width="stretch", hide_index=True)


def reset_workspace() -> None:
    st.session_state.datasets = {}
    st.session_state.selected_dataset = None
    st.session_state.last_validation = None
    st.session_state.merge_preview = None
    st.session_state.generator_column_count = 4
    st.session_state.engine.logger.clear()
    st.session_state.export_artifact = None
    st.session_state.export_context = None


def upload_phase() -> None:
    st.markdown("## Upload Phase")
    st.caption("CSV and XLSX only. In single mode, Excel uploads use the first worksheet. In multiple mode, all worksheets are loaded as datasets.")
    mode = st.session_state.mode
    accept_multiple = mode == "Multiple Dataset"
    uploaded_files = st.file_uploader(
        "Upload dataset files",
        type=["csv", "xlsx"],
        accept_multiple_files=accept_multiple,
        help="Single mode accepts one file. Multiple mode accepts several files and multi-sheet workbooks.",
    )

    load_clicked = st.button("Load Dataset Files", type="primary", width="stretch")
    if not load_clicked:
        return

    if not uploaded_files:
        st.error("Upload at least one dataset before loading.")
        return

    files = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
    if mode == "Single Dataset" and len(files) != 1:
        st.error("Single Dataset mode requires exactly one uploaded file.")
        return

    engine: AutomationEngine = st.session_state.engine
    try:
        with st.spinner("Loading datasets..."):
            loaded_datasets: list[DatasetArtifact] = []
            for uploaded_file in files:
                uploaded_file.seek(0)
                loaded_datasets.extend(
                    engine.load_file(
                        file_name=uploaded_file.name,
                        file_object=BytesIO(uploaded_file.getvalue()),
                        load_all_sheets=(mode == "Multiple Dataset"),
                    )
                )
    except DataAutomationError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(f"Unexpected upload failure: {exc}")
        return

    if mode == "Single Dataset" and len(loaded_datasets) != 1:
        st.error("Single Dataset mode must resolve to exactly one dataset after loading.")
        return

    for dataset in loaded_datasets:
        add_dataset(dataset)

    st.success(f"Loaded {len(loaded_datasets)} dataset(s) into the workspace.")


def _parse_optional_numeric(value: str, *, label: str, integer: bool = False) -> float | int | None:
    cleaned_value = value.strip()
    if not cleaned_value:
        return None
    try:
        return int(cleaned_value) if integer else float(cleaned_value)
    except ValueError as exc:
        target_type = "integer" if integer else "number"
        raise ValueError(f"{label} must be a valid {target_type}.") from exc


def _parse_optional_date(value: str, *, label: str) -> date | None:
    cleaned_value = value.strip()
    if not cleaned_value:
        return None
    try:
        return pd.to_datetime(cleaned_value).date()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a valid date, for example 2024-01-15.") from exc


def generator_phase() -> None:
    st.markdown("## Data Generator")
    st.caption(
        "Define a schema, generate up to 500 rows, and add the result directly to the active workspace. "
        "Generated datasets move through the same profiling, cleaning, validation, merge, analytics, and export pipeline as uploaded files."
    )

    generator_controls = st.columns([0.95, 0.7, 0.7, 1.15])
    with generator_controls[0]:
        st.markdown(f"**Schema Columns:** `{st.session_state.generator_column_count}`")
    with generator_controls[1]:
        if st.button("Add Column", width="stretch"):
            st.session_state.generator_column_count = min(st.session_state.generator_column_count + 1, 30)
    with generator_controls[2]:
        if st.button("Remove Column", width="stretch", disabled=(st.session_state.generator_column_count <= 1)):
            st.session_state.generator_column_count = max(st.session_state.generator_column_count - 1, 1)
    with generator_controls[3]:
        configured_count = st.number_input(
            "Total Schema Columns",
            min_value=1,
            max_value=30,
            value=int(st.session_state.generator_column_count),
            step=1,
            help="Increase this when you need more than the default four generated columns.",
        )
        st.session_state.generator_column_count = int(configured_count)

    suggested_name = f"generated_dataset_{len(dataset_names()) + 1}"
    with st.form("generator_form"):
        top_controls = st.columns([1.2, 0.8, 0.8, 0.8])
        with top_controls[0]:
            dataset_name = st.text_input("Dataset Name", value=suggested_name)
        with top_controls[1]:
            row_count = st.number_input("Rows", min_value=1, max_value=500, value=50, step=1)
        with top_controls[2]:
            st.metric("Columns", st.session_state.generator_column_count)
        with top_controls[3]:
            use_seed = st.checkbox("Use Random Seed", value=False)
            random_seed = st.number_input("Seed", min_value=0, value=42, step=1, disabled=not use_seed)

        st.caption(
            "Supported types: integer, float, string, category, date, boolean. "
            "Use `Primary Identifier` or disable duplicates when a column must be unique. "
            "Each column also supports a sample value so generated data can follow a real example."
        )

        column_configs: list[dict[str, object]] = []
        for index in range(int(st.session_state.generator_column_count)):
            with st.expander(f"Column {index + 1}", expanded=(index < 3)):
                identity_columns = st.columns([1.2, 1, 0.8, 0.9])
                with identity_columns[0]:
                    column_name = st.text_input("Column Name", key=f"generator_name_{index}", value=f"column_{index + 1}")
                with identity_columns[1]:
                    data_type = st.selectbox(
                        "Type",
                        options=["integer", "float", "string", "category", "date", "boolean"],
                        key=f"generator_type_{index}",
                    )
                with identity_columns[2]:
                    primary_key = st.checkbox("Primary Identifier", key=f"generator_primary_{index}", value=(index == 0))
                with identity_columns[3]:
                    allow_duplicates = st.checkbox(
                        "Allow Duplicates",
                        key=f"generator_duplicates_{index}",
                        value=not primary_key,
                        disabled=primary_key,
                    )

                min_input = ""
                max_input = ""
                sample_input = ""
                category_input = ""
                pattern = "none"
                start_date = None
                end_date = None
                true_probability = 0.5
                boolean_sample = ""

                if data_type in {"integer", "float"}:
                    numeric_columns = st.columns(3)
                    with numeric_columns[0]:
                        min_input = st.text_input("Minimum Value", key=f"generator_min_{index}")
                    with numeric_columns[1]:
                        max_input = st.text_input("Maximum Value", key=f"generator_max_{index}")
                    with numeric_columns[2]:
                        sample_input = st.text_input(
                            "Sample Value",
                            key=f"generator_sample_numeric_{index}",
                            help="Example numeric value used as a reference during generation.",
                        )
                elif data_type == "string":
                    string_columns = st.columns(2)
                    with string_columns[0]:
                        pattern = st.selectbox(
                            "Pattern",
                            options=["none", "email", "phone", "name", "company"],
                            key=f"generator_pattern_{index}",
                            help="Email and phone enforce realistic contact formats. Name and company use faker-style records.",
                        )
                    with string_columns[1]:
                        sample_input = st.text_input(
                            "Sample Value",
                            key=f"generator_sample_string_{index}",
                            help="Example text used to shape the generated values.",
                        )
                elif data_type == "category":
                    category_columns = st.columns(2)
                    with category_columns[0]:
                        category_input = st.text_input(
                            "Category Values",
                            key=f"generator_categories_{index}",
                            value="A, B, C" if index == 0 else "",
                            help="Comma-separated category values.",
                        )
                    with category_columns[1]:
                        sample_input = st.text_input(
                            "Sample Category",
                            key=f"generator_sample_category_{index}",
                            help="Example category value to include in the generated output.",
                        )
                elif data_type == "date":
                    date_columns = st.columns(3)
                    default_end = date.today()
                    default_start = default_end - timedelta(days=90)
                    with date_columns[0]:
                        start_date = st.date_input("Start Date", key=f"generator_start_date_{index}", value=default_start)
                    with date_columns[1]:
                        end_date = st.date_input("End Date", key=f"generator_end_date_{index}", value=default_end)
                    with date_columns[2]:
                        sample_input = st.text_input(
                            "Sample Date",
                            key=f"generator_sample_date_{index}",
                            placeholder="2024-01-15",
                            help="Optional example date used to anchor generated dates.",
                        )
                elif data_type == "boolean":
                    boolean_columns = st.columns(2)
                    with boolean_columns[0]:
                        true_probability = st.slider(
                            "True Probability",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.5,
                            step=0.05,
                            key=f"generator_probability_{index}",
                        )
                    with boolean_columns[1]:
                        boolean_sample = st.selectbox(
                            "Sample Value",
                            options=["", "True", "False"],
                            key=f"generator_sample_boolean_{index}",
                            help="Optional example boolean value to seed the generated column.",
                        )

                column_configs.append(
                    {
                        "name": column_name.strip(),
                        "data_type": data_type,
                        "allow_duplicates": (False if primary_key else allow_duplicates),
                        "primary_key": primary_key,
                        "min_input": min_input,
                        "max_input": max_input,
                        "sample_input": sample_input,
                        "boolean_sample": boolean_sample,
                        "category_input": category_input,
                        "pattern": pattern,
                        "start_date": start_date,
                        "end_date": end_date,
                        "true_probability": true_probability,
                    }
                )

        submitted = st.form_submit_button("Generate Dataset", type="primary")

    if submitted:
        try:
            column_schemas: list[GeneratedColumnSchema] = []
            for index, column_config in enumerate(column_configs, start=1):
                data_type = str(column_config["data_type"])
                column_name = str(column_config["name"]).strip()
                min_value = _parse_optional_numeric(
                    str(column_config["min_input"]),
                    label=f"Minimum value for column '{column_name or index}'",
                    integer=(data_type == "integer"),
                )
                max_value = _parse_optional_numeric(
                    str(column_config["max_input"]),
                    label=f"Maximum value for column '{column_name or index}'",
                    integer=(data_type == "integer"),
                )
                if data_type in {"integer", "float"}:
                    sample_value = _parse_optional_numeric(
                        str(column_config["sample_input"]),
                        label=f"Sample value for column '{column_name or index}'",
                        integer=(data_type == "integer"),
                    )
                elif data_type == "date":
                    sample_value = _parse_optional_date(
                        str(column_config["sample_input"]),
                        label=f"Sample date for column '{column_name or index}'",
                    )
                elif data_type == "boolean":
                    boolean_sample = str(column_config["boolean_sample"])
                    sample_value = None if not boolean_sample else (boolean_sample == "True")
                else:
                    sample_value = str(column_config["sample_input"]).strip() or None
                categories = [value.strip() for value in str(column_config["category_input"]).split(",") if value.strip()]
                column_schemas.append(
                    GeneratedColumnSchema(
                        name=column_name,
                        data_type=data_type,
                        allow_duplicates=bool(column_config["allow_duplicates"]),
                        primary_key=bool(column_config["primary_key"]),
                        sample_value=sample_value,
                        min_value=min_value,
                        max_value=max_value,
                        categories=categories,
                        pattern=str(column_config["pattern"]) if data_type == "string" else None,
                        start_date=column_config["start_date"],
                        end_date=column_config["end_date"],
                        true_probability=float(column_config["true_probability"]),
                    )
                )
            generation_request = DatasetGenerationRequest(
                dataset_name=dataset_name.strip(),
                row_count=int(row_count),
                columns=column_schemas,
                random_seed=int(random_seed) if use_seed else None,
            )
            generated_dataset = st.session_state.engine.generate_dataset(generation_request)
            add_dataset(generated_dataset)
            st.success(
                f"Generated dataset `{generated_dataset.name}` with {generated_dataset.row_count} rows and "
                f"{generated_dataset.column_count} columns."
            )
        except ValueError as exc:
            st.error(str(exc))
        except DataAutomationError as exc:
            st.error(str(exc))

    selected = get_selected_dataset()
    if selected and selected.source_type == "generated":
        st.markdown("### Generated Dataset Downloads")
        csv_artifact = st.session_state.engine.exporter.export(selected, file_format="csv")
        xlsx_artifact = st.session_state.engine.exporter.export(selected, file_format="xlsx")
        download_columns = st.columns(2)
        with download_columns[0]:
            st.download_button(
                "Download Generated CSV",
                data=csv_artifact.bytes_data,
                file_name=csv_artifact.file_name,
                mime=csv_artifact.mime_type,
                width="stretch",
            )
        with download_columns[1]:
            st.download_button(
                "Download Generated Excel",
                data=xlsx_artifact.bytes_data,
                file_name=xlsx_artifact.file_name,
                mime=xlsx_artifact.mime_type,
                width="stretch",
            )


def render_dashboard() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Initialize Workflow</h1>
            <p>Select whether you want to upload a dataset or generate one from a structured schema, then move it through the same profiling, cleaning, filtering, validation, transformation, merging, analytics, and export workflow.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    input_mode = st.radio(
        "Select Mode",
        options=["Upload Dataset", "Generate Dataset"],
        horizontal=True,
        index=0 if st.session_state.input_mode == "Upload Dataset" else 1,
    )
    st.session_state.input_mode = input_mode

    if input_mode == "Upload Dataset":
        mode = st.radio(
            "Workspace Scope",
            options=["Single Dataset", "Multiple Dataset"],
            horizontal=True,
            index=(0 if st.session_state.mode == "Single Dataset" else 1 if st.session_state.mode == "Multiple Dataset" else None),
        )
        if mode is None:
            st.info("Select a workspace scope to enable dataset upload.")
        elif mode != st.session_state.mode and st.session_state.datasets:
            st.warning("Switching modes after datasets are loaded can invalidate the workflow. Reset the workspace to start in a different mode.")
        else:
            st.session_state.mode = mode
        if st.session_state.mode is not None:
            upload_phase()
    else:
        generator_phase()

    selected = get_selected_dataset()
    if not selected:
        st.info("No datasets loaded yet. Upload a file above, generate a dataset from schema, or use the sample CSV files in `sample_data/`.")
        return

    dataset_options = dataset_names()
    if dataset_options:
        selected_name = st.selectbox("Active Dataset", options=dataset_options, index=dataset_options.index(st.session_state.selected_dataset))
        st.session_state.selected_dataset = selected_name
        selected = st.session_state.datasets[selected_name]

    render_profile(selected)
    render_preview(selected)

    if st.session_state.engine.logger.list_records():
        st.markdown("### Recent Operations")
        recent_records = pd.DataFrame(record.to_dict() for record in st.session_state.engine.logger.list_records()[-8:])
        st.dataframe(recent_records, width="stretch", hide_index=True)


def render_cleaning() -> None:
    dataset = get_selected_dataset()
    st.markdown("## Cleaning")
    if not dataset:
        st.info("Load a dataset before using cleaning operations.")
        return

    with st.form("cleaning_form"):
        remove_duplicates = st.checkbox("Remove duplicate rows")
        duplicate_subset = st.multiselect("Duplicate key columns", options=list(dataset.dataframe.columns))
        missing_method = st.selectbox(
            "Missing value strategy",
            options=["none", "drop", "mean", "median", "mode", "forward_fill", "constant"],
        )
        missing_columns = st.multiselect("Columns to clean", options=list(dataset.dataframe.columns), default=list(dataset.dataframe.columns))
        fill_value = st.text_input("Constant fill value", disabled=(missing_method != "constant"))
        submitted = st.form_submit_button("Apply Cleaning", type="primary")

    if submitted:
        engine: AutomationEngine = st.session_state.engine
        try:
            working_dataset = dataset
            if remove_duplicates:
                working_dataset = engine.deduplicate(working_dataset, subset=duplicate_subset or None)
            if missing_method != "none":
                working_dataset = engine.clean_missing(
                    working_dataset,
                    method=missing_method,
                    columns=missing_columns or None,
                    fill_value=fill_value if missing_method == "constant" else None,
                )
            if working_dataset is dataset:
                st.warning("No cleaning operation was selected.")
            else:
                replace_dataset(dataset.name, working_dataset)
                st.success("Cleaning completed successfully.")
        except DataAutomationError as exc:
            st.error(str(exc))

    render_profile(get_selected_dataset())
    render_preview(get_selected_dataset(), title="Cleaned Dataset Preview")


def render_filtering() -> None:
    dataset = get_selected_dataset()
    st.markdown("## Filtering")
    if not dataset:
        st.info("Load a dataset before using filtering.")
        return

    condition_count = st.number_input("Number of filter conditions", min_value=1, max_value=5, value=1, step=1)
    conditions: list[FilterCondition] = []
    for index in range(int(condition_count)):
        st.markdown(f"### Condition {index + 1}")
        condition_columns = st.columns(4)
        with condition_columns[0]:
            column = st.selectbox(f"Column {index + 1}", options=list(dataset.dataframe.columns), key=f"filter_column_{index}")
        with condition_columns[1]:
            operator = st.selectbox(
                f"Operator {index + 1}",
                options=["==", "!=", ">", ">=", "<", "<=", "contains", "in", "between", "is_null", "not_null"],
                key=f"filter_operator_{index}",
            )
        with condition_columns[2]:
            value = st.text_input(f"Value {index + 1}", key=f"filter_value_{index}", disabled=operator in {"is_null", "not_null"})
        with condition_columns[3]:
            secondary = st.text_input(f"Second value {index + 1}", key=f"filter_second_{index}", disabled=(operator != "between"))
        conditions.append(FilterCondition(column=column, operator=operator, value=value, secondary_value=secondary))

    if st.button("Apply Filters", type="primary", width="stretch"):
        try:
            filtered = st.session_state.engine.filter_rows(dataset, conditions)
            replace_dataset(dataset.name, filtered)
            st.success("Filtering completed successfully.")
        except DataAutomationError as exc:
            st.error(str(exc))

    render_preview(get_selected_dataset(), title="Filtered Dataset Preview")


def render_validation() -> None:
    dataset = get_selected_dataset()
    st.markdown("## Validation")
    if not dataset:
        st.info("Load a dataset before using validation.")
        return

    numeric_columns = list(dataset.dataframe.select_dtypes(include="number").columns)
    schema_reference_options = [name for name in dataset_names() if name != dataset.name]
    with st.form("validation_form"):
        selected_type_columns = st.multiselect("Columns for dtype validation", options=list(dataset.dataframe.columns))
        expected_types = {
            column: st.selectbox(
                f"Expected dtype for {column}",
                options=["int64", "float64", "object", "string", "bool", "datetime64[ns]"],
                key=f"dtype_{column}",
            )
            for column in selected_type_columns
        }
        range_column = st.selectbox("Numeric column for range validation", options=[""] + numeric_columns)
        min_value = st.text_input("Minimum value")
        max_value = st.text_input("Maximum value")
        reference_dataset_name = st.selectbox("Reference schema dataset", options=[""] + schema_reference_options)
        outlier_columns = st.multiselect("Columns for outlier detection", options=numeric_columns, default=numeric_columns[: min(3, len(numeric_columns))])
        validate_clicked = st.form_submit_button("Run Validation", type="primary")

    if validate_clicked:
        range_rules = {}
        if range_column:
            try:
                parsed_min = float(min_value) if min_value else None
                parsed_max = float(max_value) if max_value else None
            except ValueError:
                st.error("Range validation requires numeric min and max values.")
                parsed_min = None
                parsed_max = None
            else:
                range_rules[range_column] = {"min": parsed_min, "max": parsed_max}

        reference_schema = {}
        if reference_dataset_name:
            reference_dataset = st.session_state.datasets[reference_dataset_name]
            reference_schema = {column: str(dtype) for column, dtype in reference_dataset.dataframe.dtypes.items()}

        try:
            report = st.session_state.engine.validate(
                dataset,
                expected_types=expected_types,
                range_rules=range_rules,
                reference_schema=reference_schema,
                outlier_columns=outlier_columns or None,
            )
            st.session_state.last_validation = report
            if report.has_errors:
                st.error("Validation completed with errors.")
            else:
                st.success("Validation completed.")
        except DataAutomationError as exc:
            st.error(str(exc))

    report = st.session_state.last_validation
    if report and report.dataset_name == dataset.name:
        summary = report.summary()
        summary_columns = st.columns(3)
        with summary_columns[0]:
            render_metric_card("Errors", str(summary.get("error", 0)), "Blocking issues")
        with summary_columns[1]:
            render_metric_card("Warnings", str(summary.get("warning", 0)), "Requires review")
        with summary_columns[2]:
            render_metric_card("Info", str(summary.get("info", 0)), "Observational checks")
        issues_frame = pd.DataFrame(
            {
                "severity": [issue.severity for issue in report.issues],
                "category": [issue.category for issue in report.issues],
                "message": [issue.message for issue in report.issues],
                "columns": [", ".join(issue.columns) for issue in report.issues],
                "details": [issue.details for issue in report.issues],
            }
        )
        st.dataframe(issues_frame, width="stretch", hide_index=True)
    render_preview(dataset, title="Validated Dataset Preview")


def render_transformation() -> None:
    dataset = get_selected_dataset()
    st.markdown("## Transformation")
    if not dataset:
        st.info("Load a dataset before using transformations.")
        return

    with st.form("transform_form"):
        selected_columns = st.multiselect("Select columns to retain", options=list(dataset.dataframe.columns), default=list(dataset.dataframe.columns))
        rename_targets = st.multiselect("Columns to rename", options=list(dataset.dataframe.columns))
        rename_map = {column: st.text_input(f"Rename {column} to", value=column, key=f"rename_{column}") for column in rename_targets}
        derived_name = st.text_input("Derived column name")
        derived_expression = st.text_input("Derived column expression", placeholder="revenue - cost")
        conversion_columns = st.multiselect("Columns to type-convert", options=list(dataset.dataframe.columns))
        conversions = {
            column: st.selectbox(
                f"Convert {column} to",
                options=["string", "int64", "float64", "bool", "datetime64[ns]"],
                key=f"convert_{column}",
            )
            for column in conversion_columns
        }
        transform_clicked = st.form_submit_button("Apply Transformations", type="primary")

    if transform_clicked:
        engine: AutomationEngine = st.session_state.engine
        try:
            working_dataset = dataset
            if selected_columns and selected_columns != list(dataset.dataframe.columns):
                working_dataset = engine.select_columns(working_dataset, selected_columns)
            if conversions:
                working_dataset = engine.convert_types(working_dataset, conversions)
            if derived_name and derived_expression:
                working_dataset = engine.derive_column(working_dataset, new_column=derived_name, expression=derived_expression)
            active_rename_map = {old: new for old, new in rename_map.items() if new and new != old}
            if active_rename_map:
                working_dataset = engine.rename_columns(working_dataset, active_rename_map)
            if working_dataset is dataset:
                st.warning("No transformation was selected.")
            else:
                replace_dataset(dataset.name, working_dataset)
                st.success("Transformations completed successfully.")
        except DataAutomationError as exc:
            st.error(str(exc))

    render_preview(get_selected_dataset(), title="Transformed Dataset Preview")


def render_merging() -> None:
    st.markdown("## Merging")
    names = dataset_names()
    if len(names) < 2:
        st.error("Merge is blocked until at least two datasets are loaded.")
        return

    left_column, config_column, right_column = st.columns([1, 1.2, 1])
    with left_column:
        left_name = st.selectbox("Left dataset", options=names, key="merge_left")
        left_dataset = st.session_state.datasets[left_name]
        st.dataframe(left_dataset.dataframe.head(8), width="stretch", hide_index=True)
    with right_column:
        right_options = [name for name in names if name != left_name]
        right_name = st.selectbox("Right dataset", options=right_options, key="merge_right")
        right_dataset = st.session_state.datasets[right_name]
        st.dataframe(right_dataset.dataframe.head(8), width="stretch", hide_index=True)
    with config_column:
        join_type = st.selectbox("Join type", options=["inner", "left", "right", "outer"])
        common_columns = sorted(set(left_dataset.dataframe.columns).intersection(right_dataset.dataframe.columns))
        default_left_keys = common_columns[:1] if common_columns else []
        default_right_keys = common_columns[:1] if common_columns else []
        left_keys = st.multiselect("Left join keys", options=list(left_dataset.dataframe.columns), default=default_left_keys)
        right_keys = st.multiselect("Right join keys", options=list(right_dataset.dataframe.columns), default=default_right_keys)
        align_key_types = st.checkbox("Align mismatched key dtypes automatically", value=True)
        if common_columns:
            st.caption(f"Suggested shared columns: {', '.join(common_columns[:6])}")
        else:
            st.caption("No identical column names were found across the selected datasets. Pick matching columns manually.")

        merge_validation_error = None
        if not left_keys or not right_keys:
            merge_validation_error = "Select at least one join key on both the left and right datasets."
        elif len(left_keys) != len(right_keys):
            merge_validation_error = "The number of left join keys must match the number of right join keys."
        elif left_name == right_name:
            merge_validation_error = "Select two different datasets to merge."

        if merge_validation_error:
            st.warning(merge_validation_error)
        if st.button("Preview Merge Risk", width="stretch"):
            if merge_validation_error:
                st.error(merge_validation_error)
            else:
                try:
                    warnings = st.session_state.engine.merger.analyse_merge_risk(
                        left_dataset,
                        right_dataset,
                        left_keys=left_keys,
                        right_keys=right_keys,
                    )
                    st.session_state.merge_preview = warnings
                except DataAutomationError as exc:
                    st.error(str(exc))
        if st.button("Execute Merge", type="primary", width="stretch"):
            if merge_validation_error:
                st.error(merge_validation_error)
            else:
                try:
                    configuration = MergeConfiguration(
                        left_dataset_name=left_name,
                        right_dataset_name=right_name,
                        left_keys=left_keys,
                        right_keys=right_keys,
                        join_type=join_type,
                        align_key_types=align_key_types,
                    )
                    merged, warnings = st.session_state.engine.merge(left_dataset, right_dataset, configuration)
                    add_dataset(merged)
                    st.session_state.merge_preview = warnings
                    st.success(f"Merge completed. Result dataset `{merged.name}` added to the workspace.")
                except DataAutomationError as exc:
                    st.error(str(exc))

    if st.session_state.merge_preview:
        st.markdown("### Merge Risk Assessment")
        for warning in st.session_state.merge_preview:
            st.warning(warning)

    selected = get_selected_dataset()
    if selected:
        render_preview(selected, title="Merged Dataset Preview")


def render_export() -> None:
    dataset = get_selected_dataset()
    st.markdown("## Export")
    if not dataset:
        st.info("Load a dataset before exporting.")
        return

    export_columns = st.multiselect(
        "Columns to export",
        options=list(dataset.dataframe.columns),
        default=list(dataset.dataframe.columns),
        help="Choose only the columns you want in the exported file. This does not modify the working dataset.",
    )
    file_format = st.selectbox("Export format", options=["csv", "xlsx"])
    if export_columns:
        export_preview_dataset = dataset.clone(
            dataframe=dataset.dataframe.loc[:, export_columns].copy(deep=True),
            name=f"{dataset.name}_export_preview",
        )
        st.caption(f"Export selection: {len(export_columns)} of {dataset.column_count} columns")
        render_preview(export_preview_dataset, title="Export Preview", rows=20)
    else:
        st.warning("Select at least one column to prepare an export.")
        render_preview(dataset, title="Current Dataset Preview", rows=20)

    if st.button("Prepare Export", type="primary", width="stretch"):
        try:
            engine = st.session_state.engine
            export_signature = inspect.signature(engine.export)
            export_kwargs = {"file_format": file_format}
            if "columns" in export_signature.parameters:
                export_kwargs["columns"] = export_columns
            export_artifact = engine.export(dataset, **export_kwargs)
            st.session_state.export_artifact = export_artifact
            st.session_state.export_context = {
                "dataset_name": dataset.name,
                "file_format": file_format,
                "columns": export_columns,
            }
            st.success("Export is ready for download.")
        except DataAutomationError as exc:
            st.error(str(exc))

    export_context = st.session_state.export_context
    export_artifact = st.session_state.export_artifact
    if (
        export_artifact is not None
        and export_context is not None
        and export_context.get("dataset_name") == dataset.name
        and export_context.get("file_format") == file_format
        and export_context.get("columns") == export_columns
    ):
        st.download_button(
            label=f"Download {export_artifact.file_name}",
            data=export_artifact.bytes_data,
            file_name=export_artifact.file_name,
            mime=export_artifact.mime_type,
            width="stretch",
        )
        st.caption(
            f"Prepared export contains {export_artifact.column_count} columns and {export_artifact.row_count} rows."
        )


def _altair_field_type(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "T"
    if pd.api.types.is_numeric_dtype(series):
        return "Q"
    return "N"


def _build_altair_chart(chart_artifact, spec: ChartSpec) -> alt.Chart:
    chart_type = spec.chart_type.lower()
    if chart_type == "heatmap":
        return (
            alt.Chart(chart_artifact.dataframe)
            .mark_rect(cornerRadius=8)
            .encode(
                x=alt.X("x:N", title=spec.x_column, sort="-y"),
                y=alt.Y("y:N", title=spec.y_column),
                color=alt.Color("value:Q", title="Count", scale=alt.Scale(scheme="teals")),
                tooltip=[
                    alt.Tooltip("x:N", title=spec.x_column),
                    alt.Tooltip("y:N", title=spec.y_column),
                    alt.Tooltip("value:Q", title="Count"),
                ],
            )
            .properties(height=460)
        )

    if chart_type == "pie":
        return (
            alt.Chart(chart_artifact.dataframe)
            .mark_arc(innerRadius=72, outerRadius=170)
            .encode(
                theta=alt.Theta("y:Q", title=spec.y_column or "Count"),
                color=alt.Color("x:N", title=spec.x_column, scale=alt.Scale(scheme="tableau20")),
                tooltip=[
                    alt.Tooltip("x:N", title=spec.x_column),
                    alt.Tooltip("y:Q", title=spec.y_column or "Count"),
                ],
            )
            .properties(height=460)
        )

    x_type = _altair_field_type(chart_artifact.dataframe["x"])
    x_encoding = alt.X(f"x:{x_type}", title=spec.x_column, sort=None)
    y_encoding = alt.Y("y:Q", title=spec.y_column)
    tooltip = [
        alt.Tooltip(f"x:{x_type}", title=spec.x_column),
        alt.Tooltip("y:Q", title=spec.y_column),
    ]

    if chart_type == "line":
        base = alt.Chart(chart_artifact.dataframe).encode(x=x_encoding, y=y_encoding, tooltip=tooltip)
        chart = base.mark_line(strokeWidth=3.5, interpolate="monotone", color="#57d5ff") + base.mark_point(
            size=95,
            filled=True,
            color="#e8f7ff",
            stroke="#57d5ff",
            strokeWidth=2,
        )
        return chart.properties(height=460)

    if chart_type == "scatter":
        return (
            alt.Chart(chart_artifact.dataframe)
            .mark_circle(size=120, opacity=0.84, color="#59d6d6", stroke="#eff8ff", strokeWidth=1.6)
            .encode(x=x_encoding, y=y_encoding, tooltip=tooltip)
            .properties(height=460)
        )

    return (
        alt.Chart(chart_artifact.dataframe)
        .mark_bar(cornerRadiusTopLeft=7, cornerRadiusTopRight=7, color="#57d5ff")
        .encode(x=x_encoding, y=y_encoding, tooltip=tooltip)
        .properties(height=460)
    )


def _apply_chart_theme(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_view(stroke=None)
        .configure_axis(
            labelColor="#b7c4dc",
            titleColor="#eef4ff",
            gridColor="rgba(179,197,226,0.15)",
            domainColor="rgba(179,197,226,0.20)",
            tickColor="rgba(179,197,226,0.20)",
        )
        .configure_title(
            color="#f5f8ff",
            fontSize=20,
            subtitleColor="#9fb0d0",
            anchor="start",
        )
        .configure_legend(
            titleColor="#eef4ff",
            labelColor="#b7c4dc",
            orient="bottom",
        )
    )


def render_visual_analytics() -> None:
    dataset = get_selected_dataset()
    render_section_header(
        "Visualization Studio",
        "Build publication-grade charts from the active dataset, switch between line, scatter, bar, pie, and heatmap views, and download the result as a polished vector image.",
    )
    if not dataset:
        st.info("Load a dataset before using the visualization studio.")
        return

    dataframe = dataset.dataframe
    numeric_columns = list(dataframe.select_dtypes(include="number").columns)
    all_columns = list(dataframe.columns)
    chart_type_labels = {
        "line": "Line",
        "scatter": "Scatter",
        "bar": "Bar",
        "pie": "Pie",
        "heatmap": "Heatmap",
    }
    fallback_chart_types = ["bar", "pie", "heatmap", "line", "scatter"]
    current_chart_type = st.session_state.get("chart_type", "line")
    current_aggregation = st.session_state.get("chart_aggregation", "sum")
    current_top_n = int(st.session_state.get("chart_top_n", 12))
    current_x = st.session_state.get("chart_x_column")
    current_y = st.session_state.get("chart_y_column")
    current_heatmap_y = st.session_state.get("chart_heatmap_y_column")
    current_pie_metric = st.session_state.get("chart_pie_value_column")
    preferred_y = None if current_pie_metric == "(Count Rows)" else (current_heatmap_y if current_chart_type == "heatmap" else current_y)
    initial_candidates = recommend_chart_specs(
        dataframe,
        chart_type=current_chart_type,
        preferred_x=current_x,
        preferred_y=preferred_y,
        aggregation=current_aggregation,
        top_n=current_top_n,
    )
    default_spec = initial_candidates[0] if initial_candidates else None

    control_columns = st.columns([1.15, 1.1, 0.9, 0.85])
    with control_columns[0]:
        chart_type = st.selectbox("Graph Type", options=list(chart_type_labels), format_func=chart_type_labels.get, key="chart_type")
        default_x = default_spec.x_column if default_spec is not None else all_columns[0]
        x_column = st.selectbox("Primary Column", options=all_columns, index=all_columns.index(default_x), key="chart_x_column")
    with control_columns[1]:
        if chart_type in {"line", "scatter", "bar"}:
            if chart_type == "scatter":
                if numeric_columns:
                    candidates = recommend_chart_specs(
                        dataframe,
                        chart_type=chart_type,
                        preferred_x=x_column,
                        preferred_y=st.session_state.get("chart_y_column"),
                        aggregation=current_aggregation,
                        top_n=current_top_n,
                    )
                    default_y = next(
                        (candidate.y_column for candidate in candidates if candidate.x_column == x_column and candidate.y_column in numeric_columns),
                        numeric_columns[0],
                    )
                    y_column = st.selectbox("Value Column", options=numeric_columns, index=numeric_columns.index(default_y), key="chart_y_column")
                else:
                    y_column = None
                    st.caption("Scatter needs numeric fields. If none are available, the module will auto-switch to another chart.")
            else:
                value_options = ["(Count Rows)"] + numeric_columns
                candidates = recommend_chart_specs(
                    dataframe,
                    chart_type=chart_type,
                    preferred_x=x_column,
                    preferred_y=(None if st.session_state.get("chart_y_column") == "(Count Rows)" else st.session_state.get("chart_y_column")),
                    aggregation=current_aggregation,
                    top_n=current_top_n,
                )
                default_metric = next(
                    (
                        "(Count Rows)" if candidate.y_column is None else candidate.y_column
                        for candidate in candidates
                        if candidate.x_column == x_column and (candidate.y_column in numeric_columns or candidate.y_column is None)
                    ),
                    "(Count Rows)",
                )
                selected_metric = st.selectbox(
                    "Value Column",
                    options=value_options,
                    index=value_options.index(default_metric),
                    key="chart_y_column",
                )
                y_column = None if selected_metric == "(Count Rows)" else selected_metric
        elif chart_type == "pie":
            pie_options = ["(Count Rows)"] + numeric_columns
            candidates = recommend_chart_specs(
                dataframe,
                chart_type="pie",
                preferred_x=x_column,
                preferred_y=(None if st.session_state.get("chart_pie_value_column") == "(Count Rows)" else st.session_state.get("chart_pie_value_column")),
                aggregation=current_aggregation,
                top_n=current_top_n,
            )
            default_pie_metric = next(
                (
                    "(Count Rows)" if candidate.y_column is None else candidate.y_column
                    for candidate in candidates
                    if candidate.x_column == x_column and ((candidate.y_column in numeric_columns) or candidate.y_column is None)
                ),
                "(Count Rows)",
            )
            pie_selection = st.selectbox(
                "Slice Metric",
                options=pie_options,
                index=pie_options.index(default_pie_metric),
                key="chart_pie_value_column",
            )
            y_column = None if pie_selection == "(Count Rows)" else pie_selection
        else:
            secondary_options = [column for column in all_columns if column != x_column] or all_columns
            candidates = recommend_chart_specs(
                dataframe,
                chart_type="heatmap",
                preferred_x=x_column,
                preferred_y=st.session_state.get("chart_heatmap_y_column"),
                aggregation=current_aggregation,
                top_n=current_top_n,
            )
            default_heatmap_y = next(
                (candidate.y_column for candidate in candidates if candidate.x_column == x_column and candidate.y_column in secondary_options),
                secondary_options[0],
            )
            y_column = st.selectbox(
                "Secondary Column",
                options=secondary_options,
                index=secondary_options.index(default_heatmap_y),
                key="chart_heatmap_y_column",
            )
    with control_columns[2]:
        aggregation_options = list(SUPPORTED_AGGREGATIONS)
        aggregation = st.selectbox(
            "Aggregation",
            options=aggregation_options,
            index=aggregation_options.index(current_aggregation) if current_aggregation in aggregation_options else 0,
            disabled=chart_type in {"scatter", "heatmap"},
            key="chart_aggregation",
        )
        top_n = st.slider(
            "Display Limit",
            min_value=4,
            max_value=20,
            value=current_top_n if 4 <= current_top_n <= 20 else 12,
            disabled=chart_type in {"line", "scatter"},
            key="chart_top_n",
        )
    with control_columns[3]:
        st.markdown("### Output")
        st.caption("Charts render live in the app and download as SVG for crisp reports and presentations.")
        st.caption("Heatmaps summarize pair-frequency across the selected columns.")

    candidate_specs = recommend_chart_specs(
        dataframe,
        chart_type=chart_type,
        preferred_x=x_column,
        preferred_y=y_column,
        aggregation=aggregation,
        top_n=top_n,
    )
    if not candidate_specs:
        candidate_specs = []
        for fallback_chart_type in fallback_chart_types:
            if fallback_chart_type == chart_type:
                continue
            candidate_specs.extend(
                recommend_chart_specs(
                    dataframe,
                    chart_type=fallback_chart_type,
                    preferred_x=x_column,
                    preferred_y=y_column,
                    aggregation=aggregation,
                    top_n=top_n,
                )
            )
        if not candidate_specs:
            st.error("No compatible columns are available for plotting in this dataset.")
            return

    spec = None
    chart_artifact = None
    errors: list[str] = []
    for candidate in candidate_specs:
        try:
            chart_artifact = build_chart_artifact(dataframe, candidate, dataset_name=dataset.name)
            spec = candidate
            break
        except ValueError as exc:
            errors.append(str(exc))

    if chart_artifact is None or spec is None:
        st.error(errors[0] if errors else "No compatible chart could be created from this dataset.")
        return

    if spec.chart_type != chart_type or spec.x_column != x_column or spec.y_column != y_column:
        resolved_y = spec.y_column if spec.y_column is not None else "Count Rows"
        st.info(
            f"Automatically selected a compatible plot: `{chart_type_labels[spec.chart_type]}` using "
            f"`{spec.x_column}` and `{resolved_y}`."
        )

    chart = _apply_chart_theme(_build_altair_chart(chart_artifact, spec)).properties(
        title=alt.TitleParams(
            text=chart_artifact.title,
            subtitle=[f"Dataset: {dataset.name}"],
        )
    )

    metric_columns = st.columns(3)
    with metric_columns[0]:
        render_metric_card("Plotted Rows", f"{len(chart_artifact.dataframe):,}", "Rows used in the current visualization")
    with metric_columns[1]:
        render_metric_card("X Axis", chart_artifact.x_label, "Primary dimension")
    with metric_columns[2]:
        render_metric_card("Y Axis", chart_artifact.y_label or "Count", "Measured value")

    st.altair_chart(chart, width="stretch", theme=None)
    st.download_button(
        "Download Graph as SVG",
        data=chart_artifact.svg_bytes,
        file_name=f"{dataset.name}_{chart_type}.svg",
        mime="image/svg+xml",
        width="stretch",
    )
    st.caption("SVG is a vector image format, which keeps chart text and lines sharp in Word, PowerPoint, and print exports.")

    st.markdown("### Plot Data")
    st.dataframe(chart_artifact.dataframe, width="stretch", hide_index=True)


def render_statistical_tables() -> None:
    dataset = get_selected_dataset()
    render_section_header(
        "Statistical Tables",
        "Select any columns, generate a presentation-ready summary table with key statistics, and copy the formatted output directly into Word while keeping table structure intact.",
    )
    if not dataset:
        st.info("Load a dataset before generating statistical tables.")
        return

    dataframe = dataset.dataframe
    default_columns = list(dataframe.columns[: min(6, len(dataframe.columns))])
    default_statistics = ["dtype", "count", "nulls", "unique", "mean", "median", "std", "min", "max", "q1", "q3", "iqr", "mode"]

    table_control_columns = st.columns([1.4, 1.2])
    with table_control_columns[0]:
        selected_columns = st.multiselect(
            "Columns",
            options=list(dataframe.columns),
            default=default_columns,
            help="The table will include one summary row per selected column.",
        )
    with table_control_columns[1]:
        selected_statistics = st.multiselect(
            "Statistics",
            options=SUPPORTED_STATISTICS,
            default=default_statistics,
            help="Choose the metrics to include in the output table.",
        )

    if not selected_columns or not selected_statistics:
        st.warning("Select at least one column and one statistic.")
        return

    try:
        stats_frame = build_statistics_table(dataframe, columns=selected_columns, statistics=selected_statistics)
    except ValueError as exc:
        st.error(str(exc))
        return

    metric_columns = st.columns(3)
    with metric_columns[0]:
        render_metric_card("Columns Analysed", f"{len(selected_columns):,}", "Included in the summary table")
    with metric_columns[1]:
        render_metric_card("Metrics Included", f"{len(selected_statistics):,}", "Statistics per column")
    with metric_columns[2]:
        render_metric_card("Source Rows", f"{len(dataframe):,}", "Rows scanned for the summary")

    st.markdown("### Summary Matrix")
    st.dataframe(stats_frame, width="stretch", hide_index=True)

    html_table = statistics_table_to_html(stats_frame, title=f"{dataset.name} Statistical Summary")
    component_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", f"word-copy-{dataset.name.lower()}").strip("-") or "word-copy-table"
    component_html = build_word_copy_component(html_table, component_id=component_id)
    component_height = min(860, 220 + (len(stats_frame) * 48))
    components.html(component_html, height=component_height, scrolling=True)

    st.download_button(
        "Download Table as HTML",
        data=html_table.encode("utf-8"),
        file_name=f"{dataset.name}_summary_table.html",
        mime="text/html",
        width="stretch",
    )
    st.caption("If your browser blocks clipboard access, download the HTML file and open it in Word to keep the table styling.")

    st.markdown("### Selected Data Preview")
    st.dataframe(dataframe.loc[:, selected_columns].head(50), width="stretch", hide_index=True)


def render_history() -> None:
    st.markdown("## Operation History")
    records = st.session_state.engine.logger.list_records()
    if not records:
        st.info("No logged operations yet.")
        return
    history_frame = pd.DataFrame(record.to_dict() for record in records)
    st.dataframe(history_frame, width="stretch", hide_index=True)


def sidebar() -> str:
    with st.sidebar:
        st.markdown("## VANDI_DATA_CENTER")
        st.caption("Automation Engine | Local Instance")
        if st.button("Reset Workspace", width="stretch"):
            reset_workspace()
            st.rerun()
        current_names = dataset_names()
        if current_names:
            selected_name = st.selectbox(
                "Workspace Dataset",
                options=current_names,
                index=current_names.index(st.session_state.selected_dataset) if st.session_state.selected_dataset in current_names else 0,
            )
            st.session_state.selected_dataset = selected_name
        st.markdown("---")
        return st.radio(
            "Workflow Modules",
            options=["Dashboard", "Cleaning", "Filtering", "Validation", "Transformation", "Merging", "Visual Analytics", "Statistical Tables", "Export", "History"],
        )


def main() -> None:
    init_state()
    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.markdown('<div class="app-shell">', unsafe_allow_html=True)
    page = sidebar()

    if page == "Dashboard":
        render_dashboard()
    elif page == "Cleaning":
        render_cleaning()
    elif page == "Filtering":
        render_filtering()
    elif page == "Validation":
        render_validation()
    elif page == "Transformation":
        render_transformation()
    elif page == "Merging":
        render_merging()
    elif page == "Visual Analytics":
        render_visual_analytics()
    elif page == "Statistical Tables":
        render_statistical_tables()
    elif page == "Export":
        render_export()
    elif page == "History":
        render_history()

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
