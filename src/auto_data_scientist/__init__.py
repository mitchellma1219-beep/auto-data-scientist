"""Auto Data Scientist: profiling, guarded SQL, modelling, forecasting and reporting."""

from auto_data_scientist.config import Limits
from auto_data_scientist.errors import (
    AdsError,
    DataError,
    QueryTimeoutError,
    TranslationError,
    UnsafeQueryError,
)
from auto_data_scientist.forecasting import forecast_series
from auto_data_scientist.ingest import Dataset, load_dataset
from auto_data_scientist.modeling import train_supervised
from auto_data_scientist.pipeline import analyze, ask, write_artifacts
from auto_data_scientist.profiling import profile_frame
from auto_data_scientist.query import run_query
from auto_data_scientist.report import render_report
from auto_data_scientist.sql_guard import validate_sql

__version__ = "0.1.0"

__all__ = [
    "AdsError",
    "DataError",
    "Dataset",
    "Limits",
    "QueryTimeoutError",
    "TranslationError",
    "UnsafeQueryError",
    "analyze",
    "ask",
    "forecast_series",
    "load_dataset",
    "profile_frame",
    "render_report",
    "run_query",
    "train_supervised",
    "validate_sql",
    "write_artifacts",
]
