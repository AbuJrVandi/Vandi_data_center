from __future__ import annotations

import base64
from datetime import date, datetime, timedelta
from html import escape
import inspect
from io import BytesIO
import numbers
from pathlib import Path
import re
import sys

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import qrcode
except ImportError:  # pragma: no cover - runtime fallback when optional dependency is missing
    qrcode = None

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
GENERATED_EXPORT_DIR = ROOT_DIR / "generated_exports"
AUTHOR_PHOTO_CANDIDATES = (
    "WhatsApp Image 2026-04-21 at 6.40.46 PM.jpeg",
)
AUTHOR_PORTFOLIO_URL = "https://abujuniorvandi.vercel.app/"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def resolve_author_photo_path() -> Path | None:
    for relative_path in AUTHOR_PHOTO_CANDIDATES:
        candidate = ROOT_DIR / relative_path
        if candidate.is_file():
            return candidate
    return None


AUTHOR_PHOTO_PATH = resolve_author_photo_path()

from astrodata_tool import (
    AutomationEngine,
    DatasetArtifact,
    DatasetGenerationRequest,
    FilterCondition,
    GeneratedFileArtifact,
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
    page_title="Vandi Data Center Automation Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_CSS = """
<style>
    :root {
        --app-bg-top: #f9fbff;
        --app-bg-bottom: #eef3fb;
        --card-bg: rgba(255, 255, 255, 0.92);
        --card-bg-strong: rgba(255, 255, 255, 0.98);
        --card-border: rgba(27, 46, 94, 0.10);
        --text-strong: #172033;
        --text-body: #53627f;
        --text-muted: #6f7f9b;
        --accent-soft: rgba(47, 107, 255, 0.10);
        --accent: #2f6bff;
        --shadow-soft: 0 16px 38px rgba(26, 43, 77, 0.08);
    }
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(47,107,255,0.14), transparent 24%),
            radial-gradient(circle at bottom left, rgba(32,184,133,0.09), transparent 22%),
            linear-gradient(180deg, var(--app-bg-top) 0%, var(--app-bg-bottom) 100%);
        color: var(--text-strong);
    }
    .app-shell {
        padding: 0.75rem 0 1rem 0;
    }
    .hero {
        border: 1px solid var(--card-border);
        background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(247,250,255,0.98));
        border-radius: 18px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-soft);
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
        line-height: 1.2;
        color: var(--text-strong);
    }
    .hero p {
        margin: 0.55rem 0 0 0;
        color: var(--text-body);
        max-width: 60rem;
    }
    .metric-card {
        border: 1px solid var(--card-border);
        background: var(--card-bg);
        border-radius: 16px;
        padding: 1rem 1rem 0.85rem 1rem;
        min-height: 120px;
        box-shadow: 0 10px 26px rgba(29, 48, 86, 0.05);
    }
    .metric-label {
        color: var(--text-muted);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .metric-value {
        color: var(--text-strong);
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }
    .metric-detail {
        color: var(--text-body);
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }
    .section-card {
        border: 1px solid var(--card-border);
        background: var(--card-bg);
        border-radius: 16px;
        padding: 1rem 1rem 1.1rem 1rem;
        margin-bottom: 1rem;
    }
    .module-header {
        border: 1px solid var(--card-border);
        background:
            radial-gradient(circle at top right, rgba(47, 107, 255, 0.10), transparent 24%),
            linear-gradient(180deg, rgba(255,255,255,0.96), rgba(244,248,255,0.98));
        border-radius: 18px;
        padding: 1.1rem 1.2rem 1.15rem 1.2rem;
        margin-bottom: 0.9rem;
        box-shadow: var(--shadow-soft);
    }
    .module-kicker {
        display: inline-block;
        margin-bottom: 0.55rem;
        padding: 0.22rem 0.55rem;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent);
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .module-header h1 {
        margin: 0;
        font-size: 1.7rem;
        line-height: 1.1;
        color: var(--text-strong);
    }
    .module-header p {
        margin: 0.45rem 0 0 0;
        color: var(--text-body);
        max-width: 58rem;
        font-size: 0.97rem;
    }
    .workspace-card {
        border: 1px solid var(--card-border);
        background: var(--card-bg);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        margin-bottom: 1rem;
    }
    .workspace-label {
        color: var(--text-muted);
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
    }
    .workspace-value {
        color: var(--text-strong);
        font-size: 1rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .workspace-detail {
        color: var(--text-body);
        font-size: 0.84rem;
        margin-top: 0.25rem;
    }
    .stage-block {
        margin: 0.15rem 0 0.8rem 0;
    }
    .stage-block h3 {
        margin: 0;
        font-size: 1.02rem;
        color: var(--text-strong);
    }
    .stage-block p {
        margin: 0.25rem 0 0 0;
        color: var(--text-body);
        font-size: 0.9rem;
    }
    .section-title {
        margin: 0;
        font-size: 1.45rem;
        color: var(--text-strong);
    }
    .section-lead {
        margin: 0.35rem 0 0 0;
        color: var(--text-body);
        max-width: 56rem;
    }
    .small-note {
        color: var(--text-muted);
        font-size: 0.92rem;
    }
    [data-testid="stSidebar"] {
        background:
            radial-gradient(circle at top right, rgba(47, 107, 255, 0.10), transparent 28%),
            linear-gradient(180deg, rgba(248,250,255,0.98), rgba(239,244,251,0.99));
        border-right: 1px solid rgba(27, 46, 94, 0.08);
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .sidebar-brand {
        border: 1px solid var(--card-border);
        background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(246,249,255,0.98));
        border-radius: 18px;
        padding: 1rem 1rem 0.95rem 1rem;
        margin-bottom: 0.9rem;
        box-shadow: var(--shadow-soft);
    }
    .sidebar-brand-label {
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 0.09em;
        font-size: 0.75rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .sidebar-brand-title {
        color: var(--text-strong);
        font-size: 1.2rem;
        font-weight: 700;
        line-height: 1.2;
        margin: 0;
    }
    .sidebar-brand-copy {
        color: var(--text-body);
        font-size: 0.87rem;
        margin-top: 0.35rem;
    }
    .sidebar-panel {
        border: 1px solid var(--card-border);
        background: var(--card-bg);
        border-radius: 16px;
        padding: 0.95rem 1rem;
        margin-bottom: 0.85rem;
    }
    .sidebar-panel-title {
        color: var(--text-strong);
        font-size: 0.9rem;
        font-weight: 700;
        margin: 0 0 0.75rem 0;
    }
    .sidebar-status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.32rem 0.62rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    .sidebar-status-badge.neutral {
        background: rgba(127, 223, 255, 0.12);
        color: #7fdfff;
    }
    .sidebar-status-badge.success {
        background: rgba(46, 204, 113, 0.14);
        color: #7cf1af;
    }
    .sidebar-status-badge.warning {
        background: rgba(255, 183, 77, 0.15);
        color: #ffd083;
    }
    .sidebar-status-badge.danger {
        background: rgba(255, 107, 129, 0.16);
        color: #ff9aad;
    }
    .sidebar-keyline {
        color: var(--text-body);
        font-size: 0.82rem;
        margin-bottom: 0.8rem;
        line-height: 1.45;
    }
    .sidebar-stat-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.65rem;
    }
    .sidebar-stat {
        border: 1px solid rgba(27, 46, 94, 0.08);
        background: rgba(246, 249, 255, 0.95);
        border-radius: 14px;
        padding: 0.72rem 0.78rem;
        min-height: 88px;
    }
    .sidebar-stat-label {
        color: var(--text-muted);
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .sidebar-stat-value {
        color: var(--text-strong);
        font-size: 0.97rem;
        font-weight: 700;
        margin-top: 0.32rem;
        line-height: 1.25;
        word-break: break-word;
    }
    .sidebar-stat-detail {
        color: var(--text-body);
        font-size: 0.77rem;
        margin-top: 0.28rem;
        line-height: 1.3;
    }
    .sidebar-activity-list {
        display: grid;
        gap: 0.65rem;
    }
    .sidebar-activity-item {
        border-left: 2px solid rgba(47, 107, 255, 0.45);
        padding-left: 0.7rem;
    }
    .sidebar-activity-item strong {
        display: block;
        color: var(--text-strong);
        font-size: 0.84rem;
        margin-bottom: 0.15rem;
    }
    .sidebar-activity-item span {
        display: block;
        color: var(--text-body);
        font-size: 0.77rem;
        line-height: 1.35;
    }
    .dashboard-hero {
        border: 1px solid var(--card-border);
        background:
            radial-gradient(circle at top right, rgba(47, 107, 255, 0.10), transparent 24%),
            linear-gradient(180deg, rgba(255,255,255,0.97), rgba(244,248,255,0.98));
        border-radius: 18px;
        padding: 1.2rem 1.25rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-soft);
    }
    .dashboard-hero h3 {
        margin: 0;
        font-size: 1.2rem;
        color: var(--text-strong);
    }
    .dashboard-hero p {
        margin: 0.38rem 0 0 0;
        color: var(--text-body);
        line-height: 1.5;
    }
    .dashboard-note {
        border: 1px solid var(--card-border);
        background: var(--card-bg-strong);
        border-radius: 16px;
        padding: 1rem 1.05rem;
        margin-bottom: 1rem;
    }
    .dashboard-note-label {
        color: var(--text-muted);
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.45rem;
    }
    .dashboard-note-value {
        color: var(--text-strong);
        font-size: 1.05rem;
        font-weight: 700;
        line-height: 1.3;
    }
    .dashboard-note-copy {
        color: var(--text-body);
        font-size: 0.86rem;
        margin-top: 0.4rem;
        line-height: 1.45;
    }
    .author-page-hero {
        border: 1px solid var(--card-border);
        background:
            radial-gradient(circle at top right, rgba(47, 107, 255, 0.08), transparent 26%),
            linear-gradient(180deg, rgba(255,255,255,0.98), rgba(246,249,255,0.98));
        border-radius: 24px;
        padding: 1.55rem 1.7rem;
        margin-bottom: 1.2rem;
        box-shadow: var(--shadow-soft);
    }
    .author-page-kicker {
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 0.74rem;
        font-weight: 800;
        margin-bottom: 0.7rem;
    }
    .author-page-hero h1 {
        margin: 0 0 0.95rem 0;
        color: var(--text-strong);
        font-size: 2.15rem;
        line-height: 1.05;
    }
    .author-page-hero p {
        margin: 0 0 0.9rem 0;
        color: var(--text-body);
        font-size: 1rem;
        line-height: 1.78;
        max-width: 68rem;
    }
    .author-page-hero p:last-child {
        margin-bottom: 0;
    }
    .author-photo-frame {
        border: none;
        border-radius: 0;
        overflow: visible;
        background: transparent;
        box-shadow: none;
    }
    .author-photo-frame img {
        display: block;
        width: 100%;
        height: auto;
    }
    .author-photo-caption {
        margin-top: 0.75rem;
        color: var(--text-body);
        font-size: 0.87rem;
        line-height: 1.55;
        padding-left: 0.1rem;
    }
    .author-profile-details {
        margin-top: 0.9rem;
        color: var(--text-body);
    }
    .author-profile-title {
        color: var(--text-strong);
        font-size: 1rem;
        font-weight: 700;
        line-height: 1.45;
        margin-bottom: 0.45rem;
    }
    .author-profile-line {
        font-size: 0.92rem;
        line-height: 1.7;
        margin: 0;
    }
    .author-profile-label {
        color: var(--text-strong);
        font-weight: 700;
    }
    .author-profile-line a {
        color: var(--accent);
        text-decoration: none;
        font-weight: 600;
    }
    .author-profile-line a:hover {
        text-decoration: underline;
    }
    .author-qr-section {
        margin-top: 2.25rem;
        text-align: center;
    }
    .author-qr-label {
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }
    .author-qr-title {
        color: var(--text-strong);
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .author-qr-copy {
        color: var(--text-body);
        font-size: 0.92rem;
        line-height: 1.6;
        margin-bottom: 0.95rem;
    }
    .author-qr-image {
        width: 180px;
        max-width: 100%;
        height: auto;
        display: block;
        margin: 0 auto;
    }
    .author-copy-card {
        border: 1px solid var(--card-border);
        border-radius: 24px;
        padding: 1.5rem 1.55rem;
        background:
            radial-gradient(circle at top right, rgba(47, 107, 255, 0.06), transparent 25%),
            rgba(255,255,255,0.97);
        box-shadow: var(--shadow-soft);
    }
    .author-kicker {
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 0.09em;
        font-size: 0.75rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .author-name {
        color: var(--text-strong);
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.06;
        margin: 0;
    }
    .author-role {
        color: var(--text-body);
        font-size: 1rem;
        margin-top: 0.35rem;
        margin-bottom: 1rem;
    }
    .author-quote {
        margin: 0 0 1rem 0;
        padding: 0.95rem 1rem;
        border-left: 4px solid var(--accent);
        background: rgba(47, 107, 255, 0.06);
        border-radius: 0 16px 16px 0;
        color: var(--text-strong);
        font-size: 1rem;
        line-height: 1.62;
    }
    .author-body {
        color: var(--text-body);
        font-size: 0.97rem;
        line-height: 1.82;
    }
    .author-body p {
        margin: 0 0 0.95rem 0;
    }
    .author-body p:last-child {
        margin-bottom: 0;
    }
    @media (max-width: 980px) {
        .author-page-hero h1 {
            font-size: 1.8rem;
        }
        .author-copy-card {
            padding: 1.25rem 1.2rem;
        }
    }
</style>
"""

PAGE_OPTIONS = [
    "Dashboard",
    "Cleaning",
    "Filtering",
    "Validation",
    "Transformation",
    "Merging",
    "Visual Analytics",
    "Statistical Tables",
    "Export",
    "History",
    "About Developer",
]

PAGE_DESCRIPTIONS = {
    "Dashboard": "Workspace controls, current dataset health, and session status.",
    "Cleaning": "Deduplicate records and apply missing-value treatment.",
    "Filtering": "Refine the active dataset with structured conditions.",
    "Validation": "Run schema, range, and outlier checks before downstream work.",
    "Transformation": "Rename, derive, convert, and reshape the dataset.",
    "Merging": "Join datasets in multi-dataset workflows.",
    "Visual Analytics": "Build export-ready charts from the active dataset.",
    "Statistical Tables": "Generate formatted statistical summaries for reporting.",
    "Export": "Package the active dataset for downstream delivery.",
    "History": "Review the audit trail for the current workspace session.",
    "About Developer": "Meet the developer behind the platform in a dedicated author profile view.",
}

QUICK_ACTIONS = [
    "Cleaning",
    "Validation",
    "Transformation",
    "Visual Analytics",
]


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
    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"
    if "generator_large_export_artifact" not in st.session_state:
        st.session_state.generator_large_export_artifact = None


def dataset_names() -> list[str]:
    return list(st.session_state.datasets.keys())


@st.cache_data(show_spinner=False)
def build_qr_code_png_bytes(content: str) -> bytes | None:
    if qrcode is None:
        return None
    qr_image = qrcode.make(content)
    buffer = BytesIO()
    qr_image.save(buffer, format="PNG")
    return buffer.getvalue()


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


def _format_timestamp(value) -> str:
    return value.astimezone().strftime("%b %d, %Y %I:%M %p")


def _workspace_summary() -> dict[str, object]:
    dataset = get_selected_dataset()
    records = st.session_state.engine.logger.list_records()
    last_record = records[-1] if records else None
    last_validation = st.session_state.last_validation
    input_mode = st.session_state.input_mode

    if input_mode == "Generate Dataset":
        mode_label = "Generated schema workspace"
        mode_detail = "Synthetic dataset builder is active."
    elif st.session_state.mode:
        mode_label = str(st.session_state.mode)
        mode_detail = "Upload flow is configured for the current workspace."
    else:
        mode_label = "Scope not selected"
        mode_detail = "Choose single or multiple dataset handling."

    status_label = "Awaiting data"
    status_detail = "Load or generate a dataset to activate the workflow."
    status_tone = "warning"
    if dataset:
        status_label = "Workspace live"
        status_detail = f"`{dataset.name}` is ready for the next module."
        status_tone = "success"
    if last_record:
        status_label = "Recent activity"
        status_detail = last_record.summary
        status_tone = "neutral"
    if last_validation and dataset and last_validation.dataset_name == dataset.name:
        validation_summary = last_validation.summary()
        error_count = validation_summary.get("error", 0)
        warning_count = validation_summary.get("warning", 0)
        if error_count:
            status_label = "Validation issues"
            status_detail = f"{error_count} error(s) and {warning_count} warning(s) need review."
            status_tone = "danger"
        elif warning_count:
            status_label = "Validation warnings"
            status_detail = f"{warning_count} warning(s) were flagged in the latest validation run."
            status_tone = "warning"
        elif dataset:
            status_label = "Validated"
            status_detail = "The active dataset passed the latest validation checks."
            status_tone = "success"

    return {
        "dataset": dataset,
        "dataset_count": len(dataset_names()),
        "records": records,
        "record_count": len(records),
        "last_record": last_record,
        "input_mode": input_mode,
        "mode_label": mode_label,
        "mode_detail": mode_detail,
        "status_label": status_label,
        "status_detail": status_detail,
        "status_tone": status_tone,
    }


def _render_sidebar_stat(label: str, value: str, detail: str) -> str:
    return f"""
    <div class="sidebar-stat">
        <div class="sidebar-stat-label">{escape(label)}</div>
        <div class="sidebar-stat-value">{escape(value)}</div>
        <div class="sidebar-stat-detail">{escape(detail)}</div>
    </div>
    """


def _recent_activity_frame(records: list, *, limit: int = 6) -> pd.DataFrame:
    recent_records = records[-limit:][::-1]
    return pd.DataFrame(
        [
            {
                "time": _format_timestamp(record.timestamp),
                "operation": record.operation_name,
                "summary": record.summary,
                "result_dataset": record.dataset_after or record.dataset_before or "-",
            }
            for record in recent_records
        ]
    )


def _set_page(page: str) -> None:
    st.session_state.page = page
    st.rerun()


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


def render_workspace_card(label: str, value: str, detail: str) -> None:
    st.markdown(
        f"""
        <div class="workspace-card">
            <div class="workspace-label">{label}</div>
            <div class="workspace-value">{value}</div>
            <div class="workspace-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_module_shell(title: str, description: str) -> None:
    dataset = get_selected_dataset()
    input_mode = st.session_state.input_mode
    workspace_scope = st.session_state.mode or "Unset"
    active_dataset = dataset.name if dataset else "No dataset"
    dataset_shape = f"{dataset.row_count:,} rows x {dataset.column_count:,} columns" if dataset else "No active data"
    source_type = dataset.source_type.upper() if dataset else "Workspace waiting for data"

    st.markdown(
        f"""
        <div class="module-header">
            <div class="module-kicker">Workflow Module</div>
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    workspace_columns = st.columns(4)
    with workspace_columns[0]:
        render_workspace_card("Input Mode", input_mode, "How data enters the workspace")
    with workspace_columns[1]:
        render_workspace_card("Workspace Scope", workspace_scope, "Single or multiple dataset handling")
    with workspace_columns[2]:
        render_workspace_card("Active Dataset", active_dataset, source_type)
    with workspace_columns[3]:
        render_workspace_card("Current Shape", dataset_shape, "Current artifact under analysis")


def render_stage_header(title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="stage-block">
            <h3>{title}</h3>
            <p>{description}</p>
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


def _column_priority_score(column_name: str) -> int:
    normalized = column_name.lower()
    score = 0
    priority_keywords = {
        "id": 10,
        "name": 9,
        "status": 9,
        "state": 8,
        "type": 8,
        "category": 8,
        "date": 9,
        "time": 8,
        "created": 8,
        "updated": 7,
        "amount": 8,
        "total": 8,
        "count": 8,
        "value": 7,
        "score": 7,
        "price": 7,
        "email": 7,
        "phone": 6,
        "error": 8,
        "warning": 7,
    }
    for keyword, weight in priority_keywords.items():
        if keyword in normalized:
            score += weight
    return score


def _select_priority_columns(dataframe: pd.DataFrame, *, limit: int = 8) -> list[str]:
    ordered = sorted(
        enumerate(list(dataframe.columns)),
        key=lambda item: (-_column_priority_score(item[1]), item[0]),
    )
    return [column for _, column in ordered[: min(limit, len(ordered))]]


def _format_table_value(value: object) -> str:
    if isinstance(value, dict):
        return "; ".join(f"{key}: {_format_table_value(item)}" for key, item in list(value.items())[:4]) or "—"
    if isinstance(value, (list, tuple, set)):
        preview_values = [_format_table_value(item) for item in list(value)[:4]]
        return ", ".join(preview_values) if preview_values else "—"
    if value is None or pd.isna(value):
        return "—"
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%Y-%m-%d %H:%M")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return f"{int(value):,}"
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        numeric_value = float(value)
        if abs(numeric_value) >= 1000:
            return f"{numeric_value:,.2f}"
        if numeric_value.is_integer():
            return f"{numeric_value:,.0f}"
        return f"{numeric_value:,.3f}".rstrip("0").rstrip(".")
    return str(value)


def _prepare_table_frame(
    dataframe: pd.DataFrame,
    *,
    rows: int | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    visible_frame = dataframe.copy()
    if columns is not None:
        visible_columns = [column for column in columns if column in visible_frame.columns]
        visible_frame = visible_frame.loc[:, visible_columns]
    if rows is not None:
        visible_frame = visible_frame.head(rows)
    if visible_frame.empty:
        return visible_frame
    return visible_frame.apply(lambda column: column.map(_format_table_value))


def _style_severity_value(value: object) -> str:
    normalized = str(value).strip().lower()
    palette = {
        "error": ("#fff1f2", "#c62828"),
        "warning": ("#fff7e8", "#b26a00"),
        "info": ("#eef6ff", "#1e5eff"),
    }
    background, foreground = palette.get(normalized, ("#f4f7fb", "#53627f"))
    return f"background-color: {background}; color: {foreground}; font-weight: 700;"


def render_table(
    title: str,
    dataframe: pd.DataFrame,
    *,
    caption: str,
    rows: int | None = None,
    columns: list[str] | None = None,
    severity_column: str | None = None,
) -> None:
    st.markdown(f"### {title}")
    st.caption(caption)
    display_frame = _prepare_table_frame(dataframe, rows=rows, columns=columns)
    if display_frame.empty:
        st.info("No rows are available for this view yet.")
        return
    if severity_column and severity_column in display_frame.columns:
        styled_frame = display_frame.style.map(_style_severity_value, subset=[severity_column]).hide(axis="index")
        st.dataframe(styled_frame, width="stretch")
        return
    st.dataframe(display_frame, width="stretch", hide_index=True)


def render_preview(dataset: DatasetArtifact, *, title: str = "Data Preview", rows: int = 15) -> None:
    preview_columns = _select_priority_columns(dataset.dataframe, limit=8)
    render_table(
        title,
        dataset.dataframe,
        caption=(
            f"Current artifact: `{dataset.name}` from `{dataset.source_name}`. "
            f"Showing the first {min(rows, dataset.row_count):,} rows across {len(preview_columns)} priority columns."
        ),
        rows=rows,
        columns=preview_columns,
    )


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
        schema_frame = pd.DataFrame(
            {
                "column": list(profile.dtypes.keys()),
                "dtype": list(profile.dtypes.values()),
                "missing_values": [profile.missing_values[column] for column in profile.dtypes],
            }
        )
        render_table(
            "Profiling Summary",
            schema_frame,
            caption="Core schema fields ordered for quick review: column name, detected type, and missing-value count.",
        )
    with summary_columns[1]:
        summary_frame = pd.DataFrame(profile.summary_statistics).T.reset_index().rename(columns={"index": "statistic"})
        render_table(
            "Summary Statistics",
            summary_frame,
            caption="Reference metrics from the active profile to help you spot outliers and shape changes quickly.",
        )


def reset_workspace() -> None:
    st.session_state.datasets = {}
    st.session_state.selected_dataset = None
    st.session_state.last_validation = None
    st.session_state.merge_preview = None
    st.session_state.mode = None
    st.session_state.input_mode = "Upload Dataset"
    st.session_state.generator_column_count = 4
    st.session_state.generator_large_export_artifact = None
    st.session_state.engine.logger.clear()
    st.session_state.export_artifact = None
    st.session_state.export_context = None
    st.session_state.page = "Dashboard"
    reset_generator_builder()


def upload_phase() -> None:
    st.markdown("## Dataset Intake")
    st.caption("Use CSV or XLSX files. Single-dataset mode reads the first worksheet; multi-dataset mode loads every worksheet as its own dataset.")
    mode = st.session_state.mode
    accept_multiple = mode == "Multiple Dataset"
    uploaded_files = st.file_uploader(
        "Upload dataset files",
        type=["csv", "xlsx"],
        accept_multiple_files=accept_multiple,
        help="Single mode accepts one file. Multiple mode accepts several files and multi-sheet workbooks.",
    )

    load_clicked = st.button("Load Datasets", type="primary", width="stretch")
    if not load_clicked:
        return

    if not uploaded_files:
        st.error("Add at least one dataset file before loading the workspace.")
        return

    files = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
    if mode == "Single Dataset" and len(files) != 1:
        st.error("Single-dataset mode accepts exactly one uploaded file.")
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
        st.error("Single-dataset mode must resolve to exactly one dataset after loading.")
        return

    for dataset in loaded_datasets:
        add_dataset(dataset)

    st.success(f"Dataset intake complete. Loaded {len(loaded_datasets)} dataset(s) into the workspace.")


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


def _generator_type_label(data_type: str) -> str:
    labels = {
        "integer": "Integer",
        "float": "Float",
        "string": "String",
        "category": "Category",
        "date": "Date",
        "boolean": "Boolean",
    }
    return labels.get(data_type, data_type.title())


def _default_generator_dataset_name() -> str:
    return f"generated_dataset_{len(dataset_names()) + 1}"


def _looks_like_identifier_column(column_name: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", column_name.strip().lower()).strip("_")
    return bool(normalized) and any(
        token in {"id", "identifier", "key", "uuid"}
        for token in normalized.split("_")
    )


def _ensure_generator_state(column_count: int) -> None:
    default_end = date.today()
    default_start = default_end - timedelta(days=90)
    pending_dataset_name = st.session_state.pop("generator_dataset_name_pending", None)
    if pending_dataset_name is not None:
        st.session_state.generator_dataset_name = str(pending_dataset_name)
    elif "generator_dataset_name" not in st.session_state:
        st.session_state.generator_dataset_name = _default_generator_dataset_name()
    if "generator_row_count" not in st.session_state:
        st.session_state.generator_row_count = 50
    if "generator_use_seed" not in st.session_state:
        st.session_state.generator_use_seed = False
    if "generator_random_seed" not in st.session_state:
        st.session_state.generator_random_seed = 42
    if "generator_large_row_count" not in st.session_state:
        st.session_state.generator_large_row_count = 100_000
    if "generator_large_chunk_size" not in st.session_state:
        st.session_state.generator_large_chunk_size = 50_000

    for index in range(column_count):
        defaults = {
            f"generator_name_{index}": f"column_{index + 1}",
            f"generator_type_{index}": "integer",
            f"generator_primary_{index}": False,
            f"generator_duplicates_{index}": True,
            f"generator_min_{index}": "",
            f"generator_max_{index}": "",
            f"generator_sample_numeric_{index}": "",
            f"generator_pattern_{index}": "none",
            f"generator_sample_string_{index}": "",
            f"generator_categories_{index}": ("A, B, C" if index == 0 else ""),
            f"generator_sample_category_{index}": "",
            f"generator_start_date_{index}": default_start,
            f"generator_end_date_{index}": default_end,
            f"generator_sample_date_{index}": "",
            f"generator_probability_{index}": 0.5,
            f"generator_sample_boolean_{index}": "",
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
        if st.session_state[f"generator_primary_{index}"]:
            st.session_state[f"generator_duplicates_{index}"] = False

    if not st.session_state.get("generator_primary_defaults_migrated"):
        for index in range(column_count):
            column_name = str(st.session_state.get(f"generator_name_{index}", f"column_{index + 1}")).strip()
            is_primary = bool(st.session_state.get(f"generator_primary_{index}", False))
            allows_duplicates = bool(st.session_state.get(f"generator_duplicates_{index}", True))
            if is_primary and not allows_duplicates and not _looks_like_identifier_column(column_name):
                st.session_state[f"generator_primary_{index}"] = False
                st.session_state[f"generator_duplicates_{index}"] = True
        st.session_state.generator_primary_defaults_migrated = True


def reset_generator_builder() -> None:
    generator_keys = [key for key in list(st.session_state.keys()) if key.startswith("generator_")]
    for key in generator_keys:
        del st.session_state[key]
    st.session_state.generator_column_count = 4
    st.session_state.generator_large_export_artifact = None


def _format_file_size(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(size_bytes)
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    return f"{size:.1f} {units[unit_index]}"


def _build_large_export_path(dataset_name: str) -> Path:
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", dataset_name.strip().lower()).strip("_") or "generated_dataset"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return GENERATED_EXPORT_DIR / f"{safe_name}_{timestamp}.csv"


def _generator_sample_preview(index: int, data_type: str) -> str:
    if data_type in {"integer", "float"}:
        sample_value = str(st.session_state.get(f"generator_sample_numeric_{index}", "")).strip()
    elif data_type == "string":
        sample_value = str(st.session_state.get(f"generator_sample_string_{index}", "")).strip()
    elif data_type == "category":
        sample_value = str(st.session_state.get(f"generator_sample_category_{index}", "")).strip()
    elif data_type == "date":
        sample_value = str(st.session_state.get(f"generator_sample_date_{index}", "")).strip()
    else:
        sample_value = str(st.session_state.get(f"generator_sample_boolean_{index}", "")).strip()
    return sample_value or "None"


def _generator_rule_preview(index: int, data_type: str) -> str:
    if data_type in {"integer", "float"}:
        min_value = str(st.session_state.get(f"generator_min_{index}", "")).strip()
        max_value = str(st.session_state.get(f"generator_max_{index}", "")).strip()
        return f"{min_value or 'auto'} to {max_value or 'auto'}" if (min_value or max_value) else "Auto range"
    if data_type == "string":
        pattern = str(st.session_state.get(f"generator_pattern_{index}", "none"))
        return "Free text" if pattern == "none" else f"{pattern.title()} pattern"
    if data_type == "category":
        categories = [value.strip() for value in str(st.session_state.get(f"generator_categories_{index}", "")).split(",") if value.strip()]
        return f"{len(categories)} values" if categories else "Add category values"
    if data_type == "date":
        start_date = st.session_state.get(f"generator_start_date_{index}")
        end_date = st.session_state.get(f"generator_end_date_{index}")
        if start_date and end_date:
            return f"{start_date.isoformat()} to {end_date.isoformat()}"
        return "Rolling date window"
    true_probability = float(st.session_state.get(f"generator_probability_{index}", 0.5))
    return f"True {true_probability:.0%}"


def _generator_uniqueness_preview(index: int) -> str:
    if bool(st.session_state.get(f"generator_primary_{index}", False)):
        return "Primary key"
    if bool(st.session_state.get(f"generator_duplicates_{index}", True)):
        return "Duplicates allowed"
    return "Unique values"


def _generator_status_preview(index: int, data_type: str) -> str:
    column_name = str(st.session_state.get(f"generator_name_{index}", "")).strip()
    if not column_name:
        return "Needs name"
    if data_type == "category":
        categories = [value.strip() for value in str(st.session_state.get(f"generator_categories_{index}", "")).split(",") if value.strip()]
        if not categories:
            return "Needs categories"
    if data_type == "date":
        start_date = st.session_state.get(f"generator_start_date_{index}")
        end_date = st.session_state.get(f"generator_end_date_{index}")
        if start_date and end_date and start_date > end_date:
            return "Check range"
    if bool(st.session_state.get(f"generator_primary_{index}", False)):
        return "Primary key"
    if not bool(st.session_state.get(f"generator_duplicates_{index}", True)):
        return "Unique values"
    return "Ready"


def _build_generator_schema_summary(column_count: int) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for index in range(column_count):
        data_type = str(st.session_state.get(f"generator_type_{index}", "integer"))
        column_name = str(st.session_state.get(f"generator_name_{index}", f"column_{index + 1}")).strip() or "Untitled"
        rows.append(
            {
                "Column": f"{index + 1}",
                "Name": column_name,
                "Type": _generator_type_label(data_type),
                "Uniqueness": _generator_uniqueness_preview(index),
                "Rule": _generator_rule_preview(index, data_type),
                "Reference": _generator_sample_preview(index, data_type),
                "Status": _generator_status_preview(index, data_type),
            }
        )
    return pd.DataFrame(rows)


def _collect_generator_column_configs(column_count: int) -> list[dict[str, object]]:
    column_configs: list[dict[str, object]] = []
    for index in range(column_count):
        primary_key = bool(st.session_state.get(f"generator_primary_{index}", False))
        allow_duplicates = False if primary_key else bool(st.session_state.get(f"generator_duplicates_{index}", True))
        data_type = str(st.session_state.get(f"generator_type_{index}", "integer"))
        column_configs.append(
            {
                "name": str(st.session_state.get(f"generator_name_{index}", f"column_{index + 1}")).strip(),
                "data_type": data_type,
                "allow_duplicates": allow_duplicates,
                "primary_key": primary_key,
                "min_input": str(st.session_state.get(f"generator_min_{index}", "")),
                "max_input": str(st.session_state.get(f"generator_max_{index}", "")),
                "sample_input": (
                    str(st.session_state.get(f"generator_sample_numeric_{index}", ""))
                    if data_type in {"integer", "float"}
                    else str(st.session_state.get(f"generator_sample_string_{index}", ""))
                    if data_type == "string"
                    else str(st.session_state.get(f"generator_sample_category_{index}", ""))
                    if data_type == "category"
                    else str(st.session_state.get(f"generator_sample_date_{index}", ""))
                ),
                "boolean_sample": str(st.session_state.get(f"generator_sample_boolean_{index}", "")),
                "category_input": str(st.session_state.get(f"generator_categories_{index}", "")),
                "pattern": str(st.session_state.get(f"generator_pattern_{index}", "none")),
                "start_date": st.session_state.get(f"generator_start_date_{index}"),
                "end_date": st.session_state.get(f"generator_end_date_{index}"),
                "true_probability": float(st.session_state.get(f"generator_probability_{index}", 0.5)),
            }
        )
    return column_configs


def _build_generation_request_from_state(*, row_count: int, column_count: int) -> DatasetGenerationRequest:
    column_configs = _collect_generator_column_configs(column_count)
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
    return DatasetGenerationRequest(
        dataset_name=str(st.session_state.generator_dataset_name).strip(),
        row_count=int(row_count),
        columns=column_schemas,
        random_seed=(int(st.session_state.generator_random_seed) if st.session_state.generator_use_seed else None),
    )


def generator_phase() -> None:
    st.markdown("## Data Generator")
    st.caption(
        "Build a schema in a cleaner workspace, keep required settings visible, and open optional rules only when you need them. "
        "Generated datasets move through the same cleaning, validation, merge, analytics, and export pipeline as uploaded files."
    )

    column_count = int(st.session_state.generator_column_count)
    _ensure_generator_state(column_count)
    generator_success_message = st.session_state.pop("generator_success_message", None)
    schema_summary = _build_generator_schema_summary(column_count)
    unique_columns = int((schema_summary["Uniqueness"] != "Duplicates allowed").sum()) if not schema_summary.empty else 0
    reference_columns = int((schema_summary["Reference"] != "None").sum()) if not schema_summary.empty else 0

    render_stage_header("Schema Builder", "Adjust the schema size, scan the builder summary, and open only the columns that need detailed configuration.")
    if generator_success_message:
        st.success(generator_success_message)
    generator_controls = st.columns([0.95, 0.7, 0.7, 0.8, 1.1])
    with generator_controls[0]:
        render_metric_card("Schema Columns", f"{column_count:,}", "Current columns in the generator schema")
    with generator_controls[1]:
        if st.button("Add Column", width="stretch"):
            st.session_state.generator_column_count = min(column_count + 1, 30)
            st.rerun()
    with generator_controls[2]:
        if st.button("Remove Column", width="stretch", disabled=(column_count <= 1)):
            st.session_state.generator_column_count = max(column_count - 1, 1)
            st.rerun()
    with generator_controls[3]:
        if st.button("Reset Builder", width="stretch"):
            reset_generator_builder()
            st.rerun()
    with generator_controls[4]:
        st.number_input(
            "Total Schema Columns",
            min_value=1,
            max_value=30,
            step=1,
            key="generator_column_count",
            help="Increase this when the schema needs more than the default four columns.",
        )

    schema_metrics = st.columns(3)
    with schema_metrics[0]:
        render_metric_card("Unique Columns", f"{unique_columns:,}", "Primary keys and duplicate-free fields")
    with schema_metrics[1]:
        render_metric_card("Reference Columns", f"{reference_columns:,}", "Columns using sample values to shape output")
    with schema_metrics[2]:
        render_metric_card("Row Limit", "500 max", "Generator safeguard for quick iteration")

    render_table(
        "Schema Summary",
        schema_summary,
        caption="Scan the generator plan before opening column editors. Status highlights anything that still needs attention.",
    )

    render_stage_header("Required Setup", "Start with the dataset identity and row volume. Optional controls stay out of the way until needed.")
    top_controls = st.columns([1.25, 0.75, 0.8])
    with top_controls[0]:
        st.text_input("Dataset Name", key="generator_dataset_name")
    with top_controls[1]:
        st.number_input("Rows", min_value=1, max_value=500, step=1, key="generator_row_count")
    with top_controls[2]:
        render_workspace_card("Schema Width", f"{int(st.session_state.generator_column_count):,} columns", "Adjust the column count above")

    with st.expander("Optional Generation Controls", expanded=False):
        optional_controls = st.columns([0.7, 0.8, 1.5])
        with optional_controls[0]:
            st.checkbox("Use Random Seed", key="generator_use_seed")
        with optional_controls[1]:
            st.number_input("Seed", min_value=0, step=1, key="generator_random_seed", disabled=not st.session_state.generator_use_seed)
        with optional_controls[2]:
            st.caption(
                "Use a random seed when you want repeatable output across runs. Leave it off when you want fresh variation each time."
            )

    render_stage_header("Column Builder", "Required fields stay on the first tab. Optional rules handle ranges, categories, patterns, dates, and sample guidance.")
    st.caption(
        "Supported types: integer, float, string, category, date, boolean. "
        "Use `Primary Identifier` for a stable key column, or disable duplicates when a field needs unique values."
    )

    for index in range(column_count):
        data_type = str(st.session_state.get(f"generator_type_{index}", "integer"))
        column_name = str(st.session_state.get(f"generator_name_{index}", f"column_{index + 1}")).strip() or f"column_{index + 1}"
        status = _generator_status_preview(index, data_type)
        expander_label = f"Column {index + 1} | {column_name} | {_generator_type_label(data_type)} | {status}"
        with st.expander(expander_label, expanded=(index == 0)):
            required_tab, optional_tab = st.tabs(["Required", "Optional Rules"])
            with required_tab:
                identity_columns = st.columns([1.25, 1, 0.85, 0.9])
                with identity_columns[0]:
                    st.text_input("Column Name", key=f"generator_name_{index}")
                with identity_columns[1]:
                    st.selectbox(
                        "Type",
                        options=["integer", "float", "string", "category", "date", "boolean"],
                        key=f"generator_type_{index}",
                        format_func=_generator_type_label,
                    )
                with identity_columns[2]:
                    st.checkbox("Primary Identifier", key=f"generator_primary_{index}")
                if st.session_state.get(f"generator_primary_{index}", False):
                    st.session_state[f"generator_duplicates_{index}"] = False
                with identity_columns[3]:
                    st.checkbox(
                        "Allow Duplicates",
                        key=f"generator_duplicates_{index}",
                        disabled=bool(st.session_state.get(f"generator_primary_{index}", False)),
                    )
                st.caption(
                    f"Uniqueness: {_generator_uniqueness_preview(index)} | Current rule: {_generator_rule_preview(index, str(st.session_state.get(f'generator_type_{index}', data_type)))}"
                )

            current_type = str(st.session_state.get(f"generator_type_{index}", data_type))
            with optional_tab:
                if current_type in {"integer", "float"}:
                    numeric_columns = st.columns(3)
                    with numeric_columns[0]:
                        st.text_input("Minimum Value", key=f"generator_min_{index}")
                    with numeric_columns[1]:
                        st.text_input("Maximum Value", key=f"generator_max_{index}")
                    with numeric_columns[2]:
                        st.text_input(
                            "Sample Value",
                            key=f"generator_sample_numeric_{index}",
                            help="Optional example value used to shape generated output around a realistic reference.",
                        )
                    st.caption("Optional range rules are useful when the column should stay inside a known numeric band.")
                elif current_type == "string":
                    string_columns = st.columns([0.9, 1.1])
                    with string_columns[0]:
                        st.selectbox(
                            "Pattern",
                            options=["none", "email", "phone", "name", "company"],
                            key=f"generator_pattern_{index}",
                            help="Choose a pattern when generated strings should follow a business-friendly format.",
                        )
                    with string_columns[1]:
                        st.text_input(
                            "Sample Value",
                            key=f"generator_sample_string_{index}",
                            help="Optional example text to steer the generated values toward a recognizable style.",
                        )
                elif current_type == "category":
                    category_columns = st.columns([1.2, 0.8])
                    with category_columns[0]:
                        st.text_input(
                            "Category Values",
                            key=f"generator_categories_{index}",
                            help=(
                                "Comma-separated category values used as the allowed set for this field. "
                                "You can enter labels directly like `male, female` or coded mappings like "
                                "`1=male, 2=female`; generated output uses the category labels."
                            ),
                        )
                    with category_columns[1]:
                        st.text_input(
                            "Sample Category",
                            key=f"generator_sample_category_{index}",
                            help="Optional example category to anchor the distribution. For mapped values, you can enter either the code or the label.",
                        )
                elif current_type == "date":
                    date_columns = st.columns(3)
                    with date_columns[0]:
                        st.date_input("Start Date", key=f"generator_start_date_{index}")
                    with date_columns[1]:
                        st.date_input("End Date", key=f"generator_end_date_{index}")
                    with date_columns[2]:
                        st.text_input(
                            "Sample Date",
                            key=f"generator_sample_date_{index}",
                            placeholder="2024-01-15",
                            help="Optional example date used to anchor the generated window.",
                        )
                elif current_type == "boolean":
                    boolean_columns = st.columns([0.9, 0.7])
                    with boolean_columns[0]:
                        st.slider(
                            "True Probability",
                            min_value=0.0,
                            max_value=1.0,
                            step=0.05,
                            key=f"generator_probability_{index}",
                        )
                    with boolean_columns[1]:
                        st.selectbox(
                            "Sample Value",
                            options=["", "True", "False"],
                            key=f"generator_sample_boolean_{index}",
                            help="Optional example value that nudges the generated boolean series.",
                        )
                st.caption(
                    f"Reference value: {_generator_sample_preview(index, current_type)} | Status: {_generator_status_preview(index, current_type)}"
                )

    generate_clicked = st.button("Generate Dataset", type="primary", width="stretch")

    if generate_clicked:
        try:
            generation_request = _build_generation_request_from_state(
                row_count=int(st.session_state.generator_row_count),
                column_count=column_count,
            )
            generated_dataset = st.session_state.engine.generate_dataset(generation_request)
            add_dataset(generated_dataset)
            st.session_state.generator_dataset_name_pending = _default_generator_dataset_name()
            st.session_state.generator_success_message = (
                f"Generated dataset `{generated_dataset.name}` with {generated_dataset.row_count} rows and "
                f"{generated_dataset.column_count} columns."
            )
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
        except DataAutomationError as exc:
            st.error(str(exc))

    render_stage_header(
        "Production-Scale Export",
        "Use the same schema to build a file-first CSV export up to 1,000,000 rows without loading the full dataset into the live workspace.",
    )
    export_controls = st.columns([1.0, 0.9, 0.95, 1.1])
    with export_controls[0]:
        st.number_input(
            "Export Rows",
            min_value=501,
            max_value=1_000_000,
            step=10_000,
            key="generator_large_row_count",
            help="Large export mode is optimized for file delivery rather than in-app editing.",
        )
    with export_controls[1]:
        st.selectbox(
            "Chunk Size",
            options=[10_000, 25_000, 50_000, 100_000, 200_000],
            key="generator_large_chunk_size",
            format_func=lambda value: f"{value:,} rows",
            help="Larger chunks write faster, while smaller chunks reduce peak memory usage.",
        )
    with export_controls[2]:
        render_workspace_card("Delivery Mode", "CSV export", "Generated in chunks and kept out of the interactive workspace")
    with export_controls[3]:
        build_large_export = st.button("Build Large CSV Export", width="stretch")

    st.caption(
        "This mode is intended for production-scale synthetic data delivery. "
        "The file is generated in chunks, written to disk, and made available for download without loading all rows into Streamlit session memory."
    )

    if build_large_export:
        try:
            large_request = _build_generation_request_from_state(
                row_count=int(st.session_state.generator_large_row_count),
                column_count=column_count,
            )
            export_artifact = st.session_state.engine.generate_large_dataset_export(
                large_request,
                output_path=_build_large_export_path(large_request.dataset_name),
                chunk_size=int(st.session_state.generator_large_chunk_size),
            )
            st.session_state.generator_large_export_artifact = export_artifact
            st.success(
                f"Large CSV export ready. `{export_artifact.file_name}` contains {export_artifact.row_count:,} rows "
                f"and was written in {export_artifact.chunk_count:,} chunk(s)."
            )
        except ValueError as exc:
            st.error(str(exc))
        except DataAutomationError as exc:
            st.error(str(exc))

    large_export_artifact: GeneratedFileArtifact | None = st.session_state.generator_large_export_artifact
    if large_export_artifact is not None and large_export_artifact.file_path.exists():
        export_summary_columns = st.columns(4)
        with export_summary_columns[0]:
            render_metric_card("Export Rows", f"{large_export_artifact.row_count:,}", "Generated rows in the current file-first export")
        with export_summary_columns[1]:
            render_metric_card("Columns", f"{large_export_artifact.column_count:,}", "Schema width written to the export")
        with export_summary_columns[2]:
            render_metric_card("File Size", _format_file_size(large_export_artifact.file_size_bytes), "Current CSV size on disk")
        with export_summary_columns[3]:
            render_metric_card("Chunks", f"{large_export_artifact.chunk_count:,}", "Write batches used during generation")

        with large_export_artifact.file_path.open("rb") as export_handle:
            st.download_button(
                "Download Large CSV Export",
                data=export_handle,
                file_name=large_export_artifact.file_name,
                mime=large_export_artifact.mime_type,
                width="stretch",
            )
        st.caption(
            f"Saved locally at `{large_export_artifact.file_path}`. "
            "This export is file-first and is not added to the active workspace dataset list."
        )

    selected = get_selected_dataset()
    if selected and selected.source_type == "generated":
        render_stage_header("Generated Output", "Download the generated artifact immediately, then continue through the rest of the workflow.")
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
    render_module_shell(
        "Dashboard",
        "Run the workspace from one command center: set the input path, confirm the active dataset, and hand off into the next module with clear session context.",
    )
    summary = _workspace_summary()
    records = summary["records"]
    status_label = escape(str(summary["status_label"]))
    status_detail = escape(str(summary["status_detail"]))
    mode_label = str(summary["mode_label"])
    mode_detail = str(summary["mode_detail"])

    render_stage_header("Workspace Command Center", "Set the workspace path, review current status, and move the active dataset into the next step.")

    hero_columns = st.columns([1.3, 0.9])
    with hero_columns[0]:
        st.markdown(
            """
            <div class="dashboard-hero">
                <h3>Command Center</h3>
                <p>Use this page to define how data enters the workspace, confirm the active artifact, and move directly into cleaning, validation, transformation, analytics, or export without scanning raw utility screens.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hero_columns[1]:
        st.markdown(
            f"""
            <div class="dashboard-note">
                <div class="dashboard-note-label">Workspace Status</div>
                <div class="dashboard-note-value">{status_label}</div>
                <div class="dashboard-note-copy">{status_detail}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    input_mode = st.radio(
        "Input Path",
        options=["Upload Dataset", "Generate Dataset"],
        horizontal=True,
        key="input_mode",
    )

    status_columns = st.columns(4)
    with status_columns[0]:
        render_metric_card("Datasets in Workspace", f"{int(summary['dataset_count']):,}", "Loaded artifacts available for downstream modules")
    with status_columns[1]:
        render_metric_card("Session Events", f"{int(summary['record_count']):,}", "Operations recorded in the current audit trail")
    with status_columns[2]:
        render_metric_card("Current Mode", mode_label, mode_detail)
    with status_columns[3]:
        last_record = summary["last_record"]
        render_metric_card(
            "Latest Update",
            (_format_timestamp(last_record.timestamp) if last_record else "No activity yet"),
            (last_record.operation_name if last_record else "The workspace has not executed any operations."),
        )

    render_stage_header("Workspace Actions", "Choose how the workspace receives data, then activate the dataset you want to operate on.")
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
        st.info("No datasets are active yet. Load a file above, generate a dataset from schema, or use the sample files in `sample_data/`.")
        return

    profile = st.session_state.engine.profile(selected)
    missing_total = sum(profile.missing_values.values())
    preview_columns = _select_priority_columns(selected.dataframe, limit=8)
    schema_frame = pd.DataFrame(
        {
            "column": list(profile.dtypes.keys()),
            "dtype": list(profile.dtypes.values()),
            "missing": [profile.missing_values[column] for column in profile.dtypes],
        }
    ).head(10)

    render_stage_header("Active Workspace", "Keep the selected dataset in view, inspect the health summary, and move into the next operational module.")

    dataset_control_columns = st.columns([1.1, 0.9])
    with dataset_control_columns[0]:
        dataset_options = dataset_names()
        if dataset_options:
            selected_name = st.selectbox(
                "Active Dataset",
                options=dataset_options,
                index=dataset_options.index(st.session_state.selected_dataset),
            )
            st.session_state.selected_dataset = selected_name
            selected = st.session_state.datasets[selected_name]
            profile = st.session_state.engine.profile(selected)
            missing_total = sum(profile.missing_values.values())
            preview_columns = _select_priority_columns(selected.dataframe, limit=8)
            schema_frame = pd.DataFrame(
                {
                    "column": list(profile.dtypes.keys()),
                    "dtype": list(profile.dtypes.values()),
                    "missing": [profile.missing_values[column] for column in profile.dtypes],
                }
            ).head(10)
    with dataset_control_columns[1]:
        st.markdown(
            f"""
            <div class="dashboard-note">
                <div class="dashboard-note-label">Active Dataset</div>
                <div class="dashboard-note-value">{escape(selected.name)}</div>
                <div class="dashboard-note-copy">Source: {escape(selected.source_name)} | Type: {escape(selected.source_type.upper())}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    metric_columns = st.columns(4)
    with metric_columns[0]:
        render_metric_card("Rows", f"{profile.row_count:,}", "Records currently in the active dataset")
    with metric_columns[1]:
        render_metric_card("Columns", f"{profile.column_count:,}", "Fields currently available for analysis")
    with metric_columns[2]:
        render_metric_card("Missing Cells", f"{missing_total:,}", "Tracked null or empty values")
    with metric_columns[3]:
        render_metric_card("Duplicate Rows", f"{profile.duplicate_rows:,}", "Exact duplicates found in the active dataset")

    focus_columns = st.columns([1.05, 0.95])
    with focus_columns[0]:
        render_table(
            "Schema Snapshot",
            schema_frame,
            caption="Key fields in the active dataset, prioritized for structure review before deeper workflow steps.",
        )
    with focus_columns[1]:
        st.markdown("### Next Modules")
        st.caption("Move directly into the next workflow step from the command center.")
        quick_action_columns = st.columns(2)
        for index, page in enumerate(QUICK_ACTIONS):
            with quick_action_columns[index % 2]:
                if st.button(f"Open {page}", key=f"dashboard_quick_{page}", width="stretch"):
                    _set_page(page)
                st.caption(PAGE_DESCRIPTIONS[page])

    render_stage_header("Dataset Preview", "Review the first rows of the highest-priority visible columns before running downstream actions.")
    render_table(
        "Workspace Preview",
        selected.dataframe,
        caption=f"Previewing {len(preview_columns)} priority columns: {', '.join(preview_columns)}",
        rows=12,
        columns=preview_columns,
    )

    render_stage_header("Recent Activity", "Track the latest operations and their resulting datasets without leaving the command center.")
    if records:
        render_table(
            "Recent Activity",
            _recent_activity_frame(records, limit=8),
            caption="Latest workspace events with the resulting dataset kept in view for quick traceability.",
        )
    else:
        st.info("No workspace events are recorded yet. Uploads, generation, cleaning, validation, and export actions will appear here.")


def render_cleaning() -> None:
    dataset = get_selected_dataset()
    render_module_shell(
        "Cleaning",
        "Apply controlled data quality operations such as duplicate removal and missing-value treatment while preserving an auditable artifact trail.",
    )
    if not dataset:
        st.info("Load a dataset before using cleaning operations.")
        return

    render_stage_header("Configure Cleaning", "Select the quality operations to apply to the active dataset.")
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
                st.warning("No cleaning changes were selected, so the active dataset was left unchanged.")
            else:
                replace_dataset(dataset.name, working_dataset)
                st.success("Cleaning complete. The active dataset now reflects the selected quality rules.")
        except DataAutomationError as exc:
            st.error(str(exc))

    render_stage_header("Cleaning Results", "Inspect the updated profile and sample records after the cleaning step.")
    render_profile(get_selected_dataset())
    render_preview(get_selected_dataset(), title="Cleaned Dataset Preview")


def render_filtering() -> None:
    dataset = get_selected_dataset()
    render_module_shell(
        "Filtering",
        "Narrow the working dataset with structured conditions so the downstream workflow operates only on the records you intend to keep.",
    )
    if not dataset:
        st.info("Load a dataset before using filtering.")
        return

    render_stage_header("Configure Filters", "Define one or more filter conditions to shape the active dataset.")
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
            st.success("Filtering complete. The active dataset now contains only the records that matched the active conditions.")
        except DataAutomationError as exc:
            st.error(str(exc))

    render_stage_header("Filtering Results", "Review the remaining records after the active filters are applied.")
    render_preview(get_selected_dataset(), title="Filtered Dataset Preview")


def render_validation() -> None:
    dataset = get_selected_dataset()
    render_module_shell(
        "Validation",
        "Run targeted data checks across types, ranges, schema alignment, and outlier behavior to surface risk before transformation or export.",
    )
    if not dataset:
        st.info("Load a dataset before using validation.")
        return

    render_stage_header("Configure Validation", "Select the checks you want the engine to run against the active dataset.")
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
                st.error("Validation finished with blocking issues. Review the flagged rows and fields before continuing.")
            else:
                st.success("Validation finished. No blocking issues were detected for the active dataset.")
        except DataAutomationError as exc:
            st.error(str(exc))

    report = st.session_state.last_validation
    if report and report.dataset_name == dataset.name:
        render_stage_header("Validation Results", "Review the issue summary and inspect any blocking or warning-level findings.")
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
        render_table(
            "Validation Findings",
            issues_frame,
            caption="Severity is styled so blocking findings stand out before you move to transformation, analytics, or export.",
            severity_column="severity",
        )
    render_preview(dataset, title="Validated Dataset Preview")


def render_transformation() -> None:
    dataset = get_selected_dataset()
    render_module_shell(
        "Transformation",
        "Reshape the active dataset through column selection, renaming, derivation, and controlled type conversion.",
    )
    if not dataset:
        st.info("Load a dataset before using transformations.")
        return

    render_stage_header("Configure Transformations", "Choose the structural changes to apply to the active dataset.")
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
                st.warning("No transformation changes were selected, so the active dataset was left unchanged.")
            else:
                replace_dataset(dataset.name, working_dataset)
                st.success("Transformation complete. The active dataset now reflects the selected structural changes.")
        except DataAutomationError as exc:
            st.error(str(exc))

    render_stage_header("Transformation Results", "Inspect the transformed dataset before continuing to merge, chart, or export.")
    render_preview(get_selected_dataset(), title="Transformed Dataset Preview")


def render_merging() -> None:
    render_module_shell(
        "Merging",
        "Combine two workspace datasets with explicit key selection, risk preview, and controlled join behavior.",
    )
    names = dataset_names()
    if len(names) < 2:
        st.error("Merge is blocked until at least two datasets are loaded.")
        return

    render_stage_header("Configure Merge", "Choose the left and right datasets, select join keys, and inspect merge risk before execution.")
    left_column, config_column, right_column = st.columns([1, 1.2, 1])
    with left_column:
        left_name = st.selectbox("Left dataset", options=names, key="merge_left")
        left_dataset = st.session_state.datasets[left_name]
        render_table(
            "Left Dataset Preview",
            left_dataset.dataframe,
            caption="Priority columns from the left-side dataset to confirm keys before merging.",
            rows=8,
            columns=_select_priority_columns(left_dataset.dataframe, limit=6),
        )
    with right_column:
        right_options = [name for name in names if name != left_name]
        right_name = st.selectbox("Right dataset", options=right_options, key="merge_right")
        right_dataset = st.session_state.datasets[right_name]
        render_table(
            "Right Dataset Preview",
            right_dataset.dataframe,
            caption="Priority columns from the right-side dataset to confirm key alignment before merging.",
            rows=8,
            columns=_select_priority_columns(right_dataset.dataframe, limit=6),
        )
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
                    st.success(f"Merge complete. Result dataset `{merged.name}` is now available in the workspace.")
                except DataAutomationError as exc:
                    st.error(str(exc))

    if st.session_state.merge_preview:
        render_stage_header("Merge Warnings", "Review detected merge risks and cardinality issues before trusting the output.")
        st.markdown("### Merge Risk Assessment")
        for warning in st.session_state.merge_preview:
            st.warning(warning)

    selected = get_selected_dataset()
    if selected:
        render_stage_header("Merge Results", "Inspect the merged artifact that was added back into the workspace.")
        render_preview(selected, title="Merged Dataset Preview")


def render_export() -> None:
    dataset = get_selected_dataset()
    render_module_shell(
        "Export",
        "Prepare a controlled output package from the active dataset, choose the exact columns to include, and download the final file.",
    )
    if not dataset:
        st.info("Load a dataset before exporting.")
        return

    render_stage_header("Configure Export", "Select the target columns and output format for the exported artifact.")
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
            st.success("Export package is ready. Review the selection and download when you are ready.")
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
        render_stage_header("Export Package", "Download the prepared artifact once the export configuration has been validated.")
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


def _chart_type_label(chart_type: str) -> str:
    labels = {
        "line": "Line",
        "scatter": "Scatter",
        "bar": "Bar",
        "pie": "Pie",
        "heatmap": "Heatmap",
    }
    return labels.get(chart_type.lower(), chart_type.title())


def _chart_value_label(y_column: str | None) -> str:
    return y_column if y_column is not None else "Count Rows"


def _chart_scope_label(chart_type: str, aggregation: str, top_n: int) -> str:
    if chart_type in {"scatter", "heatmap"}:
        return "Direct plot"
    if chart_type == "line":
        return aggregation
    return f"{aggregation} | top {top_n}"


def _render_chart_selection_summary(title: str, spec: ChartSpec) -> None:
    summary_columns = st.columns(4)
    with summary_columns[0]:
        render_workspace_card(title, _chart_type_label(spec.chart_type), "Selected chart family")
    with summary_columns[1]:
        render_workspace_card("Primary Column", spec.x_column, "Current x-axis or category field")
    with summary_columns[2]:
        render_workspace_card("Value Field", _chart_value_label(spec.y_column), "Metric or count used in the plot")
    with summary_columns[3]:
        render_workspace_card("Plot Scope", _chart_scope_label(spec.chart_type, spec.aggregation, spec.top_n), "Aggregation and display window")


def _chart_resolution_message(requested_spec: ChartSpec, resolved_spec: ChartSpec, fallback_error: str | None) -> tuple[str, str]:
    if requested_spec == resolved_spec:
        return (
            "Requested plot confirmed",
            "The current chart matches the exact chart type and fields selected in the controls.",
        )

    reasons: list[str] = []
    if requested_spec.chart_type != resolved_spec.chart_type:
        reasons.append(
            f"The requested `{_chart_type_label(requested_spec.chart_type)}` view did not have a compatible field layout, so the chart switched to `{_chart_type_label(resolved_spec.chart_type)}`."
        )
    if requested_spec.x_column != resolved_spec.x_column:
        reasons.append(f"`{requested_spec.x_column}` could not support the final plot, so the x field moved to `{resolved_spec.x_column}`.")
    if requested_spec.y_column != resolved_spec.y_column:
        reasons.append(
            f"The value field changed from `{_chart_value_label(requested_spec.y_column)}` to `{_chart_value_label(resolved_spec.y_column)}`."
        )
    if fallback_error:
        reasons.append(f"Fallback trigger: {fallback_error}")
    return (
        "Plot adjusted for compatibility",
        " ".join(reasons),
    )


def _apply_chart_theme(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_view(stroke=None)
        .configure_axis(
            labelColor="#53627f",
            titleColor="#172033",
            gridColor="rgba(111,127,155,0.18)",
            domainColor="rgba(83,98,127,0.22)",
            tickColor="rgba(83,98,127,0.22)",
        )
        .configure_title(
            color="#172033",
            fontSize=20,
            subtitleColor="#53627f",
            anchor="start",
        )
        .configure_legend(
            titleColor="#172033",
            labelColor="#53627f",
            orient="bottom",
        )
    )


def render_visual_analytics() -> None:
    dataset = get_selected_dataset()
    render_module_shell(
        "Visualization Studio",
        "Build publication-grade charts from the active dataset, switch between line, scatter, bar, pie, and heatmap views, and download the result as a polished vector image.",
    )
    if not dataset:
        st.info("Load a dataset before using the visualization studio.")
        return

    render_stage_header("Configure Visualization", "Choose the chart type and plotting fields. The app will recover with compatible defaults when needed.")
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
        chart_type = st.selectbox("Chart Type", options=list(chart_type_labels), format_func=chart_type_labels.get, key="chart_type")
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
            "Display Range",
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

    requested_spec = ChartSpec(
        chart_type=chart_type,
        x_column=x_column,
        y_column=y_column,
        aggregation=aggregation,
        top_n=top_n,
    )
    render_stage_header("Selection Review", "Confirm what is currently requested before the app resolves the final chart.")
    _render_chart_selection_summary("Requested Plot", requested_spec)

    spec = None
    chart_artifact = None
    errors: list[str] = []
    requested_error = None
    try:
        chart_artifact = build_chart_artifact(dataframe, requested_spec, dataset_name=dataset.name)
        spec = requested_spec
    except ValueError as exc:
        requested_error = str(exc)
        errors.append(requested_error)

    if chart_artifact is None or spec is None:
        candidate_specs = [
            candidate
            for candidate in recommend_chart_specs(
                dataframe,
                chart_type=chart_type,
                preferred_x=x_column,
                preferred_y=y_column,
                aggregation=aggregation,
                top_n=top_n,
            )
            if candidate != requested_spec
        ]
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

    resolution_title, resolution_copy = _chart_resolution_message(requested_spec, spec, requested_error)
    render_stage_header("Plot Resolution", "See what the app actually plotted and whether a fallback was needed to keep the chart valid.")
    _render_chart_selection_summary("Resolved Plot", spec)
    if requested_spec == spec:
        st.success(f"{resolution_title}. {resolution_copy}")
    else:
        st.warning(f"{resolution_title}. {resolution_copy}")

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

    render_stage_header("Visualization Output", "Review the generated chart, export it, and inspect the plotted dataset behind the visual.")
    st.altair_chart(chart, width="stretch", theme=None)
    download_columns = st.columns(2)
    with download_columns[0]:
        st.download_button(
            "Download Graph as PNG",
            data=chart_artifact.png_bytes,
            file_name=f"{dataset.name}_{spec.chart_type}.png",
            mime="image/png",
            width="stretch",
        )
    with download_columns[1]:
        st.download_button(
            "Download Graph as SVG",
            data=chart_artifact.svg_bytes,
            file_name=f"{dataset.name}_{spec.chart_type}.svg",
            mime="image/svg+xml",
            width="stretch",
        )
    st.caption("PNG exports use a large white canvas focused on the chart for presentations and screenshots. SVG remains available for crisp vector output.")

    render_table(
        "Plot Data",
        chart_artifact.dataframe,
        caption=(
            f"Showing the exact dataset behind the plotted result: `{_chart_type_label(spec.chart_type)}` "
            f"with `{spec.x_column}` and `{_chart_value_label(spec.y_column)}`."
        ),
    )
    


def render_statistical_tables() -> None:
    dataset = get_selected_dataset()
    render_module_shell(
        "Statistical Tables",
        "Select any columns, generate a presentation-ready summary table with key statistics, and copy the formatted output directly into Word while keeping table structure intact.",
    )
    if not dataset:
        st.info("Load a dataset before generating statistical tables.")
        return

    render_stage_header("Configure Summary Table", "Choose the columns and statistical measures that should appear in the output matrix.")
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

    render_stage_header("Table Output", "Review the statistical matrix, copy it to Word, or download the formatted HTML version.")
    render_table(
        "Summary Matrix",
        stats_frame,
        caption="A formatted view of the selected metrics, ready for report review before copying to Word.",
    )

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
    st.caption("If your browser blocks clipboard access, download the HTML file and open it in Word to preserve the table styling.")

    render_table(
        "Selected Data Preview",
        dataframe,
        caption="A source preview of the selected columns so you can cross-check the summary table against the underlying records.",
        rows=50,
        columns=selected_columns,
    )


def render_history() -> None:
    render_module_shell(
        "Operation History",
        "Review the audit trail of actions executed in the current workspace session, including transformations, validation, generation, and export events.",
    )
    records = st.session_state.engine.logger.list_records()
    if not records:
        st.info("No workspace history is available yet. Activity will appear here after the first workflow action.")
        return
    render_stage_header("Session Audit Trail", "Inspect the recorded sequence of operations for traceability and workflow review.")
    history_frame = pd.DataFrame(
        {
            "time": [_format_timestamp(record.timestamp) for record in records],
            "operation": [record.operation_name for record in records],
            "summary": [record.summary for record in records],
            "dataset_before": [record.dataset_before or "—" for record in records],
            "dataset_after": [record.dataset_after or "—" for record in records],
            "parameters": [record.parameters for record in records],
        }
    )
    render_table(
        "Session Audit Trail",
        history_frame,
        caption="Chronological workspace history with parameters, source dataset, and resulting dataset preserved for review.",
    )


def render_about_developer() -> None:
    st.markdown(
        """
       
        """,
        unsafe_allow_html=True,
    )

    author_columns = st.columns([0.82, 1.18], gap="large")
    with author_columns[0]:
        if AUTHOR_PHOTO_PATH is not None:
            photo_bytes = AUTHOR_PHOTO_PATH.read_bytes()
            photo_base64 = base64.b64encode(photo_bytes).decode("ascii")
            photo_mime = "image/png" if AUTHOR_PHOTO_PATH.suffix.lower() == ".png" else "image/jpeg"
            st.markdown(
                f'''
                <div class="author-photo-frame">
                    <img src="data:{photo_mime};base64,{photo_base64}" alt="Portrait of Abu Jr. Vandi" />
                </div>
                ''',
                unsafe_allow_html=True,
            )
        else:
            st.info("Developer photo is currently unavailable at the configured path.")
        st.markdown(
            """
            <div class="author-profile-details">
                <div class="author-profile-title">Computer Science &amp; Data Analytics Professional</div>
                <p class="author-profile-line">+232 73914398 | <a href="mailto:abujuniorv@gmail.com">abujuniorv@gmail.com</a></p>
                <p class="author-profile-line"><span class="author-profile-label">Portfolio:</span> <a href="https://abujuniorvandi.vercel.app/" target="_blank" rel="noopener noreferrer">abujuniorvandi.vercel.app</a></p>
                <p class="author-profile-line">Sierra Leonean | Korlie Limited (WanGov)</p>
                <p class="author-profile-line"><span class="author-profile-label">LinkedIn:</span> <a href="https://www.linkedin.com/in/abu-junior-vandi-67b12425a/" target="_blank" rel="noopener noreferrer">abu-junior-vandi-67b12425a</a></p>
                <p class="author-profile-line"><span class="author-profile-label">Facebook:</span> <a href="https://www.facebook.com/people/Abu-Markovic-Vandi-Jr/100007970957639/?mibextid=wwXIfr&amp;rdid=GRDZ5tNRWUNmgkz6&amp;share_url=https%3A%2F%2Fwww.facebook.com%2Fshare%2F1ADLjFL2aK%2F%3Fmibextid%3DwwXIfr" target="_blank" rel="noopener noreferrer">Abu Markovic Vandi Jr</a></p>
                <p class="author-profile-line"><span class="author-profile-label">Instagram:</span> <a href="https://www.instagram.com/abuzo_marvani?igsh=Znh2cDl6M24xcnk3&amp;utm_source=qr" target="_blank" rel="noopener noreferrer">@abuzo_marvani</a></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with author_columns[1]:
        st.markdown(
            """
            <div class="author-kicker">Developer & Project Author</div>
            <h2 class="author-name">Abu Jr. Vandi</h2>
            <div class="author-role">Creator of the Vandi Data Center Automation Engine</div>
            <div class="author-quote">
                “Strong software should read with clarity, execute with discipline, and leave behind a system that others can trust.”
            </div>
            <div class="author-body">
                <p>
                    Abu Jr. Vandi is presented here as the author behind the platform’s design direction, engineering structure, and operational vision.
                    The profile is written in the tone of a professional publication: concise, deliberate, and focused on authorship rather than decoration.
                </p>
                <p>
                    This page recognizes the developer not merely as a coder, but as the principal steward of a working system built for data operations,
                    workflow discipline, and practical usability. The application itself reflects that authorship through organized modules, controlled
                    state handling, workflow continuity, and a presentation layer designed to remain readable under real use.
                </p>
                
            </div>
            """,
            unsafe_allow_html=True,
        )

    qr_code_png_bytes = build_qr_code_png_bytes(AUTHOR_PORTFOLIO_URL)
    if qr_code_png_bytes:
        st.markdown(
            f"""
            <div class="author-qr-section">
                <div class="author-qr-label">Portfolio Access</div>
                <div class="author-qr-title">Scan To Open </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        qr_columns = st.columns([1, 0.42, 1])
        with qr_columns[1]:
            st.image(qr_code_png_bytes, use_container_width=True)
    else:
        st.caption("Install the project requirements to enable the portfolio QR code on this page.")


def sidebar() -> str:
    with st.sidebar:
        summary = _workspace_summary()
        dataset = summary["dataset"]
        current_page = st.session_state.page if st.session_state.page in PAGE_OPTIONS else "Dashboard"
        st.session_state.page = current_page
        recent_records = list(summary["records"])[-3:][::-1]
        status_tone = escape(str(summary["status_tone"]))
        status_label = escape(str(summary["status_label"]))
        status_detail = escape(str(summary["status_detail"]))
        mode_label = str(summary["mode_label"])
        input_mode = str(summary["input_mode"])
        active_dataset_name = dataset.name if dataset else "No dataset"
        active_dataset_type = dataset.source_type.upper() if dataset else "Load or generate data"
        row_count = f"{dataset.row_count:,}" if dataset else "0"
        column_count = f"{dataset.column_count:,}" if dataset else "0"

        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-brand-label">Vandi Data Center</div>
                <div class="sidebar-brand-title">Workspace Control Panel</div>
                <div class="sidebar-brand-copy">One place to monitor mode, active data, and session movement across the app.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            
            """,
            unsafe_allow_html=True,
        )
        current_names = dataset_names()
        if current_names:
            selected_name = st.selectbox(
                "Active Workspace Dataset",
                options=current_names,
                index=current_names.index(st.session_state.selected_dataset) if st.session_state.selected_dataset in current_names else 0,
            )
            st.session_state.selected_dataset = selected_name
        else:
            st.caption("No datasets are loaded into the workspace yet.")
        st.markdown(
            """
            <div class="sidebar-panel">
                <div class="sidebar-panel-title">Navigation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Workflow Modules",
            options=PAGE_OPTIONS,
            index=PAGE_OPTIONS.index(current_page),
            label_visibility="collapsed",
        )
        if page != current_page:
            st.session_state.page = page
            current_page = page
        st.markdown(
            f"""
            <div class="sidebar-panel">
                <div class="sidebar-panel-title">Current Focus</div>
                <div class="sidebar-keyline">{escape(current_page)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(PAGE_DESCRIPTIONS[current_page])
        if recent_records:
            activity_markup = "".join(
                f"""
                <div class="sidebar-activity-item">
                    <strong>{escape(record.operation_name)}</strong>
                    <span>{escape(record.summary)}</span>
                    <span>{escape(_format_timestamp(record.timestamp))}</span>
                </div>
                """
                for record in recent_records
            )
            st.markdown(
                f"""
                <div class="sidebar-panel">
                    <div class="sidebar-panel-title">Recent Status</div>
                    <div class="sidebar-activity-list">{activity_markup}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="sidebar-panel">
                    <div class="sidebar-panel-title">Recent Status</div>
                    <div class="sidebar-keyline">No session events yet. Activity will appear here as soon as the workspace executes an action.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if st.button("Reset Workspace", width="stretch"):
            reset_workspace()
            st.rerun()
        return current_page


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
    elif page == "About Developer":
        render_about_developer()

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
