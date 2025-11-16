"""Read CSV/Excel uploaded files safely and return a pandas DataFrame."""

from typing import Union, IO
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".csv", ".xls", ".xlsx"}

def _ext(filename: str) -> str:
    return os.path.splitext(filename)[1].lower()

def is_allowed_filename(filename: str) -> bool:
    return _ext(filename) in ALLOWED_EXTENSIONS

def read_uploaded_file(file: Union[str, IO, "st.runtime.uploaded_file_manager.UploadedFile"]) -> pd.DataFrame:
    """
    Accepts:
      - local path string
      - file-like object (Streamlit upload)
    Returns a pandas DataFrame or raises ValueError / pandas errors.
    """
    # Path string
    if isinstance(file, str):
        if not is_allowed_filename(file):
            raise ValueError("Only CSV / Excel files are supported.")
        if file.lower().endswith(".csv"):
            return pd.read_csv(file)
        return pd.read_excel(file)

    # File-like uploaded object
    name = getattr(file, "name", None)
    if not name or not is_allowed_filename(name):
        raise ValueError("Uploaded file must be CSV or Excel.")
    try:
        # try to reset pointer
        if hasattr(file, "seek"):
            file.seek(0)
        if name.lower().endswith(".csv"):
            return pd.read_csv(file)
        return pd.read_excel(file)
    except Exception as exc:
        logger.exception("Failed to read uploaded file %s", name)
        # try a permissive CSV fallback for weird encodings
        if name.lower().endswith(".csv"):
            try:
                if hasattr(file, "seek"):
                    file.seek(0)
                return pd.read_csv(file, encoding="utf-8", engine="python", sep=None)
            except Exception:
                pass
        raise
