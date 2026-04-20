# Vandi Data Center

AstroData Automation Engine is a local Streamlit application for structured dataset profiling, cleaning, filtering, validation, transformation, merging, and export. The codebase is organized as reusable modules so the same services power both the UI and the test suite.

## Features

- Upload single or multiple datasets in `CSV` and `XLSX` format
- Generate synthetic datasets from a schema-driven form with up to `500` rows
- Automatic profiling on load:
  shape, dtypes, missing values, duplicate counts, summary statistics, memory usage
- Cleaning operations:
  duplicate removal, missing value drop/imputation, constant fill
- Safe row filtering with structured conditions
- Validation:
  dtype checks, numeric range checks, schema comparison, z-score outlier detection
- Transformation:
  column selection, renaming, derived columns, type conversion
- Merge workflow:
  key validation, duplicate key warnings, join risk analysis, `inner/left/right/outer`
- Export processed data to `CSV` or `XLSX`
- Export only selected columns without changing the working dataset
- Visual Analytics module for line, scatter, bar, pie, and heatmap charts with SVG download
- Statistical Tables module with copy-to-Word HTML tables and downloadable summaries
- Operation logging to `logs/operations.jsonl`
- Unit and integration tests

## Project Structure

```text
.
├── app.py
├── sample_data/
├── src/astrodata_tool/
│   ├── data_cleaner.py
│   ├── data_exporter.py
│   ├── data_filter.py
│   ├── data_loader.py
│   ├── data_merger.py
│   ├── data_profiler.py
│   ├── data_transformer.py
│   ├── data_validator.py
│   ├── engine.py
│   ├── exceptions.py
│   ├── logger.py
│   └── models.py
└── tests/
```

## Run Locally

1. Create and activate a virtual environment.
2. Install dependencies.
3. Start the Streamlit app.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Example Workflows

### Cleaning

1. Load `sample_data/messy_events.csv` in `Single Dataset` mode.
2. Open `Cleaning`.
3. Enable duplicate removal.
4. Use `median` or `forward_fill` for missing value handling.
5. Review the updated preview and history log.

### Filtering

1. After loading `messy_events.csv`, open `Filtering`.
2. Add conditions such as:
   `event_type == purchase`
   `value_usd >= 100`
3. Apply filters and review the result preview.

### Validation

1. Open `Validation`.
2. Validate `value_usd` with a range like `0` to `5000`.
3. Compare against a reference dataset schema when multiple datasets are loaded.
4. Review warnings for outliers and type mismatches.

### Merging

1. Switch to `Multiple Dataset` mode.
2. Load `sample_data/sales_orders.csv` and `sample_data/customer_profiles.csv`.
3. Open `Merging`.
4. Use `customer_id` on the left and `id` on the right.
5. Preview merge risk and execute a `left` or `inner` join.

### Export

1. Open `Export`.
2. Select the exact columns you want to include in the output file.
3. Choose `csv` or `xlsx`.
4. Click `Prepare Export`, then download the generated file.

### Generate Dataset

1. Open `Dashboard`.
2. Set `Select Mode` to `Generate Dataset`.
3. Define a dataset name, row count, and schema columns.
4. Generate the dataset and continue through the same workflow modules used for uploaded files.

## Testing

Run the automated test suite with:

```bash
pytest -q
```

## Engineering Notes

- Original datasets are never mutated. Each operation returns a new `DatasetArtifact`.
- Operation history is recorded both on each dataset artifact and in `logs/operations.jsonl`.
- Filtering and derived columns are implemented with controlled parsing rather than unrestricted code execution.
- Merge analysis warns about duplicate join keys and unmatched records before execution.
