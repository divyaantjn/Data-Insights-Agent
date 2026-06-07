from fastapi import APIRouter, HTTPException, Form, Request
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, List, Annotated, Union
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
import re
from datetime import datetime, timezone
import traceback
import io
import json
import logging
import plotly.io as pio
import psycopg2
from psycopg2 import pool
from psycopg2.extras import Json, RealDictCursor
import pickle
from contextlib import contextmanager
import gc
from opentelemetry import trace
try:
    from aws_xray_sdk.core import xray_recorder
except ImportError:
    xray_recorder = None
    
import asyncio
from concurrent.futures import ThreadPoolExecutor
from src.llm.litellm_client import LiteLLMClient
from src.analytics.data_processor import DataProcessor
from src.analytics.plot_generator import PlotGenerator
from src.utils.config_loader import ConfigLoader
from src.utils.s3_utility import S3Utility
from src.utils.config import get_model_config
from src.utils.obs import LLMUsageTracker
from src.utils.kafka import create_event_logger
from src.analytics.html_report_generator import ReportGenerator
from src.utils.reasoning_extractor import REASONING_SECTION_PROMPT
import os
import time
import logging
from src.utils.opik_setup import track_llm_calls, update_current_trace
from src.utils.follow_up_generator import generate_follow_up_questions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

router = APIRouter()

# Load configuration
config = ConfigLoader()

# Initialize clients
llm_client = LiteLLMClient(config.get_llm_config())
data_processor = DataProcessor()
plot_generator = PlotGenerator(llm_client)
s3_utility = S3Utility()
token_tracker = LLMUsageTracker()
report_generator = ReportGenerator(llm_client, plot_generator)

# Retry configuration
MAX_DB_RETRIES = 3
RETRY_DELAY = 0.5

# Code execution retry config
MAX_CODE_RETRIES = 2  # Number of times to retry failed pandas code generation

# Connection timeout configuration
DB_OPERATION_TIMEOUT = 30
CONNECTION_ACQUIRE_TIMEOUT = 5

COUNT_OF = "count of"  # used for simple count detection in questions
SUM_OF = "sum of"        # used for simple sum detection in questions
MIN_OF = "min of"        # used for simple min detection in questions
MAX_OF = "max of"        # used for simple max detection in questions
AVERAGE_OF = "average of"  # used for simple average detection in questions

DB_CONFIG = {   
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "database": os.getenv("DB_NAME", "your_database"),
    "user": os.getenv("DB_USER", "your_username"),
    "password": os.getenv("DB_PASSWORD"),
}

# Global connection pool
connection_pool = None

# Thread pool for async database operations
db_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="db_worker")

MSG_CONFIGURING_ENGINE = "Configuring analysis engine..."
MSG_LOADING_DATA = "Loading your data..."
MSG_INVALID_FILE_LINK = "Invalid file link. Please check and try again..."

def initialize_connection_pool():
    """
    Initialize the PostgreSQL connection pool with optimized settings.
    This should be called once at application startup.
    """
    global connection_pool
    
    if connection_pool:
        try:
            connection_pool.closeall()
        except Exception:
            pass
    
    try:
        min_conn = 5
        max_conn = 50
        
        connection_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=min_conn,
            maxconn=max_conn,
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            database=DB_CONFIG["database"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            connect_timeout=5,
            keepalives=1,
            keepalives_idle=30,
            keepalives_interval=5,
            keepalives_count=3,
        )
        
        if connection_pool:
            logger.info("✓ Database connection pool created successfully")
            logger.info(f"  Pool size: {min_conn}-{max_conn} connections")
            
            try:
                test_conn = None
                try:
                    test_conn = connection_pool.getconn()
                    with test_conn.cursor() as cur:
                        cur.execute("SELECT 1")
                    logger.info("✓ Database connection pool test successful")
                finally:
                    if test_conn:
                        connection_pool.putconn(test_conn)
            except Exception as test_error:
                logger.warning(f"⚠ Warning: Pool test failed: {test_error}")
        else:
            raise RuntimeError("Connection pool creation failed")
            
    except Exception as e:
        logger.error(f"✗ Error creating connection pool: {e}", exc_info=True)
        raise e


def close_connection_pool():
    """Close all connections in the pool."""
    global connection_pool
    
    if connection_pool:
        try:
            connection_pool.closeall()
            logger.info("✓ All database connections closed")
        except Exception as e:
            logger.error(f"⚠ Error closing connection pool: {e}", exc_info=True)
    
    db_executor.shutdown(wait=True)


def _validate_and_refresh_connection(conn):
    """Check connection is alive, refresh if needed."""
    try:
        conn.isolation_level
        return conn
    except Exception:
        try:
            connection_pool.putconn(conn, close=True)
        except Exception:
            pass
        refreshed = connection_pool.getconn()
        if refreshed is None:
            raise psycopg2.OperationalError("Unable to get valid connection from pool")
        return refreshed


def _return_connection_to_pool(conn):
    """Return connection to pool, closing it if broken."""
    try:
        try:
            conn.isolation_level
            connection_pool.putconn(conn)
        except Exception:
            connection_pool.putconn(conn, close=True)
    except Exception as putconn_error:
        logger.error(f"⚠ Error returning connection to pool: {putconn_error}", exc_info=True)


@contextmanager
def get_db_connection():
    """
    Optimized context manager for getting database connections from the pool.
    """
    global connection_pool

    if connection_pool is None:
        raise RuntimeError("Connection pool not initialized. Call initialize_connection_pool() first.")

    conn = None
    start_time = time.time()

    try:
        conn = connection_pool.getconn()

        if conn is None:
            raise psycopg2.OperationalError("Unable to get connection from pool (pool exhausted)")

        acquire_time = time.time() - start_time
        if acquire_time > CONNECTION_ACQUIRE_TIMEOUT:
            logger.warning(f"⚠ Warning: Connection acquisition took {acquire_time:.2f}s (limit: {CONNECTION_ACQUIRE_TIMEOUT}s)")

        conn = _validate_and_refresh_connection(conn)
        conn.autocommit = False

        yield conn

        conn.commit()

    except Exception:
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        raise

    finally:
        if conn:
            _return_connection_to_pool(conn)

async def execute_db_operation(operation_func, *args, **kwargs):
    """Execute database operations asynchronously in thread pool."""
    loop = asyncio.get_event_loop()
    
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(db_executor, operation_func, *args, **kwargs),
            timeout=DB_OPERATION_TIMEOUT
        )
        return result
    except asyncio.TimeoutError:
        raise TimeoutError(f"Database operation timed out after {DB_OPERATION_TIMEOUT} seconds") from None


def _db_create_or_update_session(session_id: str, user_id: str, s3_urls: List[str], 
                                 all_sheets_metadata: List[Dict], 
                                 file_info: Dict) -> str:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO user_sessions 
                (session_id, user_id, s3_urls, all_sheets_metadata, file_info, last_accessed)
                VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (session_id) 
                DO UPDATE SET 
                    s3_urls = EXCLUDED.s3_urls,
                    all_sheets_metadata = EXCLUDED.all_sheets_metadata,
                    file_info = EXCLUDED.file_info,
                    user_id = EXCLUDED.user_id,
                    last_accessed = CURRENT_TIMESTAMP
            """, (session_id, user_id, Json(s3_urls), 
                  Json(all_sheets_metadata), Json(file_info)))
            
            return session_id


async def create_or_update_session(session_id: str, user_id: str, s3_urls: List[str], 
                                   all_sheets_metadata: List[Dict], 
                                   file_info: Dict) -> str:
    return await execute_db_operation(
        _db_create_or_update_session,
        session_id, user_id, s3_urls, all_sheets_metadata, file_info
    )


def _db_get_session_data(session_id: str) -> Optional[Dict[str, Any]]:
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                UPDATE user_sessions 
                SET last_accessed = CURRENT_TIMESTAMP 
                WHERE session_id = %s
                RETURNING session_id, user_id, s3_urls, all_sheets_metadata, file_info, last_accessed
            """, (session_id,))
            
            result = cur.fetchone()
            
            if result:
                return dict(result)
            
            return None


async def get_session_data(session_id: str) -> Optional[Dict[str, Any]]:
    return await execute_db_operation(_db_get_session_data, session_id)


def _db_cleanup_old_sessions(days_old: int = 7) -> int:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM user_sessions 
                WHERE last_accessed < CURRENT_TIMESTAMP - INTERVAL '%s days'
            """, (days_old,))
            
            deleted_count = cur.rowcount
            return deleted_count


async def cleanup_old_sessions(days_old: int = 7):
    deleted_count = await execute_db_operation(_db_cleanup_old_sessions, days_old)
    logger.info(f"✓ Cleaned up {deleted_count} old sessions")
    return deleted_count


def _db_delete_session(session_id: str) -> bool:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM user_sessions 
                WHERE session_id = %s
            """, (session_id,))
            
            return cur.rowcount > 0


async def delete_session(session_id: str) -> bool:
    return await execute_db_operation(_db_delete_session, session_id)


def _db_initialize_database():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id VARCHAR(255) PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    s3_urls JSONB NOT NULL,
                    all_sheets_metadata JSONB NOT NULL,
                    file_info JSONB NOT NULL
                );
            """)
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id 
                ON user_sessions(user_id);
            """)
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_sessions_last_accessed 
                ON user_sessions(last_accessed);
            """)


def initialize_database():
    _db_initialize_database()
    logger.info("✓ Database tables initialized successfully")


# ==================== CLEANING FUNCTIONS ====================

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    FIX A: Normalize column names themselves — strip whitespace.
    We preserve original casing in column names to avoid breaking downstream code,
    but strip leading/trailing whitespace which is the most common issue.
    Column name → stripped version only (no lowercasing to avoid breaking schema expectations).
    """
    df.columns = [str(col).strip() if col is not None else col for col in df.columns]
    return df


def remove_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns with 'Unnamed' in their names (usually from Excel index columns)"""
    unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
    
    if unnamed_cols:
        logger.info(f"  ✓ Removing {len(unnamed_cols)} unnamed columns: {unnamed_cols}")
        df_cleaned = df.drop(columns=unnamed_cols)
        return df_cleaned
    
    return df


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from dataframe"""
    initial_rows = len(df)
    df_cleaned = df.drop_duplicates()
    removed_count = initial_rows - len(df_cleaned)
    
    if removed_count > 0:
        logger.info(f"  ✓ Removed {removed_count} duplicate rows")
    
    return df_cleaned


def handle_special_characters(df: pd.DataFrame) -> pd.DataFrame:
    """Clean special/control characters in object columns. True in-place — no column copies."""
    for i in range(len(df.columns)):
        if df.dtypes.iloc[i] != 'object':
            continue
        col = df.iloc[:, i]
        mask_notna = col.notna()
        if not mask_notna.any():
            continue

        cleaned = (
            col[mask_notna]
            .astype(str)
            .str.replace(r'[^\x20-\x7E\u00A0-\uFFFF]', '', regex=True)
            .str.replace(r'[\n\r\t\xa0]', ' ', regex=True)
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip()
            .replace('nan', np.nan)
        )
        df.iloc[mask_notna.values, i] = cleaned.values
    return df

BOOL_MAP = {
    'true': True, 'false': False,
    'yes': True, 'no': False,
    '1': True, '0': False,
    't': True, 'f': False,
    'y': True, 'n': False
}
BOOL_LIKE_VALUES = set(BOOL_MAP.keys())


def _try_convert_to_numeric(series: pd.Series) -> Optional[pd.Series]:
    temp = series.astype(str).str.replace(',', '').str.strip()
    converted = pd.to_numeric(temp, errors='coerce')
    non_null_original = series.notna().sum()
    
    # SAFE: only convert if near-100% success, not just 70%
    if non_null_original > 0 and converted.notna().sum() / non_null_original > 0.95:
        return converted  # almost all values converted cleanly
    return None  # leave as-is rather than silently nulling 30%


def _try_convert_to_bool(series: pd.Series) -> Optional[pd.Series]:
    """Try to convert a string series to boolean. Returns converted series or None."""
    unique_vals = series.dropna().unique()
    if len(unique_vals) <= 2:
        bool_like = {str(v).lower() for v in unique_vals}
        if bool_like.issubset(BOOL_LIKE_VALUES):
            return series.astype(str).str.lower().map(BOOL_MAP)
    return None


def validate_and_convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    df_cleaned = df

    for col in df_cleaned.columns:
        if pd.api.types.is_numeric_dtype(df_cleaned[col]):
            continue

        if df_cleaned[col].dtype != 'object':
            continue

        numeric = _try_convert_to_numeric(df_cleaned[col])
        if numeric is not None:
            df_cleaned[col] = numeric
            logger.info(f"  ✓ Converted '{col}' to numeric type")
            continue

        boolean = _try_convert_to_bool(df_cleaned[col])
        if boolean is not None:
            df_cleaned[col] = boolean
            logger.info(f"  ✓ Converted '{col}' to boolean type")

    return df_cleaned


def _try_convert_to_datetime(series: pd.Series) -> Optional[pd.Series]:
    sample = series.dropna().head(10)
    try:
        parsed = pd.to_datetime(sample, errors='coerce', format='mixed')
        # Only convert if 95%+ of sample parsed successfully
        if parsed.notna().sum() / len(sample) > 0.95:
            return pd.to_datetime(series, errors='coerce', dayfirst=True)
    except Exception:
        pass
    return None


def standardize_date_formats(df: pd.DataFrame) -> pd.DataFrame:
    # operate in-place instead of copying
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        if df[col].dtype != 'object':
            continue
        converted = _try_convert_to_datetime(df[col])
        if converted is not None:
            df[col] = converted
            logger.info(f"  ✓ Converted '{col}' to datetime format")
    return df

def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    # operate in-place instead of copying
    for col in df.columns:
        if df[col].dtype == 'object':
            mask = df[col].notna()
            str_mask = mask & df[col].apply(lambda x: isinstance(x, str))
            df.loc[str_mask, col] = (
                df.loc[str_mask, col]
                .str.lower()
                .str.strip()
            )
            df[col] = df[col].replace('', np.nan)
    return df

def _downcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric types and convert low-cardinality strings to categorical."""
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'object':
            n_unique = df[col].nunique()
            n_total = len(df[col])
            # Convert to categorical if cardinality < 50% of rows and < 500 unique values
            if n_total > 0 and n_unique / n_total < 0.5 and n_unique < 500:
                df[col] = df[col].astype('category')
    return df
    
def comprehensive_dataframe_cleaning(df: pd.DataFrame, file_name: str = "", sheet_name: str = "") -> pd.DataFrame:
    identifier = f"{file_name}" + (f" > {sheet_name}" if sheet_name else "")
    logger.info("Cleaning dataframe: %s | shape: %s", identifier, df.shape)

    df = normalize_column_names(df)
    df = remove_unnamed_columns(df)
    df = remove_duplicate_rows(df)

    # handle_special_characters returns a mutated df — no extra copy needed
    df = handle_special_characters(df)

    df = validate_and_convert_data_types(df)
    df = standardize_date_formats(df)
    df = normalize_dataframe_columns(df)

    df.dropna(how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = _downcast_dataframe(df)

    # Force categorical for object columns below 500 unique values
    # _downcast_dataframe already does this, but an explicit gc here
    # releases any string interning the cleaning pipeline accumulated.
    gc.collect()

    logger.info("Cleaning complete: %s | final shape: %s", identifier, df.shape)
    return df


def build_dataframe_schema_for_prompt(
    df: pd.DataFrame,
    var_name: str,
    identifier: str,
    pivot_context: Optional[dict] = None,
) -> str:
    """Build a detailed, live schema string from the actual cleaned dataframe."""
    lines = []
    _append_pivot_preamble(lines, var_name, identifier, pivot_context)

    lines.append(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    lines.append("  Columns (name | dtype | sample_values):")

    for col in df.columns:
        lines.append(_build_column_schema_line(df, col))

    return "\n".join(lines)


def _append_pivot_preamble(lines: List[str], var_name: str, identifier: str, pivot_context: Optional[dict]) -> None:
    """Prepend pivot table context to the schema description."""
    if not pivot_context:
        lines.append(f"DataFrame '{var_name}' (from {identifier}):")
        return

    lines.append(f"[PIVOT TABLE] DataFrame '{var_name}' (from {identifier}):")
    metric = pivot_context.get("pivot_metric")
    row_field = pivot_context.get("pivot_row_field")
    col_field = pivot_context.get("pivot_col_field")
    filters = pivot_context.get("pivot_filters", {})

    if metric:
        lines.append(f"  Aggregation: {metric}")
    if row_field:
        lines.append(f"  Row dimension: {row_field}")
    if col_field:
        lines.append(f"  Column values (these are COUNT columns, NOT categories): {col_field}")
    if filters:
        filter_str = ", ".join(f"{k} = {v}" for k, v in filters.items())
        lines.append(f"  Pre-applied filters: {filter_str}")
    lines.append(
        "  NOTE: Numeric columns here represent aggregated counts/sums — "
        "do NOT treat them as raw event records. "
        "Do NOT attempt to filter by individual employee records; "
        "use groupby/sum on the existing pivot columns instead."
    )
    lines.append("")


def _build_column_schema_line(df: pd.DataFrame, col: str) -> str:
    """Generate a single line describing a column's metadata and sample values."""
    dtype = str(df[col].dtype)
    series = df[col]
    
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        mn, mx, mean = series.min(), series.max(), series.mean()
        sample_info = f"min={mn:.4g}, max={mx:.4g}, mean={mean:.4g}"
    elif pd.api.types.is_datetime64_any_dtype(series):
        mn, mx = series.min(), series.max()
        sample_info = f"range: {mn} to {mx}"
    else:
        samples = series.dropna().unique()[:8].tolist()
        sample_info = f"values: {[str(s) for s in samples]}"

    null_count = series.isna().sum()
    null_note = f" ({null_count} nulls)" if null_count > 0 else ""
    return f"    - \"{col}\" ({dtype}){null_note} | {sample_info}"


def load_dataframes_from_s3(s3_urls: List[str]) -> List[Dict[str, Any]]:
    """Load dataframes from S3 URLs."""
    all_sheets_data = []
    
    for s3_url in s3_urls:
        try:
            logger.info(f"Loading from S3: {s3_url}")
            content = s3_utility.get_data_from_s3_by_url(s3_url)
            file_name = s3_utility.extract_filename_from_s3_url(s3_url)
            
            if not content:
                logger.warning(f"Warning: Empty file or not found: {s3_url}")
                continue
            
            file_data = process_file_with_sheets(content, file_name)
            
            for sheet_name, df in file_data['sheets'].items():
                if df.empty:
                    continue
                
                cleaned_df = comprehensive_dataframe_cleaning(df, file_name, sheet_name)
                # FIX F: Extract metadata AFTER cleaning so column names/types are accurate
                metadata = extract_file_metadata(cleaned_df, file_name, sheet_name)
                
                all_sheets_data.append({
                    'file_name': file_name,
                    'sheet_name': sheet_name,
                    'df': cleaned_df,
                    's3_url': s3_url,
                    'metadata': metadata
                })
        
        except Exception as e:
            logger.error(f"Error loading {s3_url}: {e}", exc_info=True)
            continue
    
    return all_sheets_data


def extract_metadata_fields(user_metadata: str, request: Request = None) -> Dict[str, str]:
    """Extract session_id, user_id, and message_id from user_metadata JSON string."""
    try:
        metadata_dict = json.loads(user_metadata) if user_metadata else {}
        
        logger.info(f"Extracted metadata: {metadata_dict}")
        
        update_current_trace(
                user=metadata_dict.get('user_email', None),
                message_id=metadata_dict.get('message_id', None),
                team_id=metadata_dict.get('team_id', None),
                organization_id=metadata_dict.get('organization_id', None),
            )
        
        session_id = metadata_dict.get("session_id")
        if not session_id:
            raise ValueError("session_id is required in user_metadata")
        
        user_id = metadata_dict.get("user_id")
        if not user_id:
            raise ValueError("user_id is required in user_metadata")
        
        message_id = metadata_dict.get("message_id")
    
        logger.info(f"[TRACING] message_id={message_id}")
        
        span = trace.get_current_span()
        try:
            segment = xray_recorder.current_segment() if xray_recorder else None
        except Exception:
            segment = None
        
        if message_id:
            if segment:
                segment.put_annotation('message_id', str(message_id))
                logger.info(f"✅ Set X-Ray message_id annotation: {message_id}")
            if span and span.is_recording():
                span.set_attribute('message_id', str(message_id))
                logger.info(f"✅ Set OTEL message_id attribute: {message_id}")
            
            if request:
                request.state.message_id = str(message_id)
        
        return {
            "session_id": str(session_id),
            "user_id": str(user_id),
            "message_id": str(message_id) if message_id else None
        }
    except json.JSONDecodeError:
        raise ValueError("Invalid user_metadata JSON format")


# ==================== EXISTING HELPER FUNCTIONS ====================

def extract_file_metadata(
    df: pd.DataFrame,
    file_name: str,
    sheet_name: Optional[str] = None,
    pivot_context: Optional[dict] = None,       # NEW
) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from a dataframe for file relevance analysis.
    Should always be called on the CLEANED dataframe.
    """
    metadata = {
        "file_name": file_name,
        "sheet_name": sheet_name,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": [],
        "is_pivot": pivot_context is not None,          # NEW
        "pivot_context": pivot_context or {},           # NEW
    }

    for col in df.columns:
        col_metadata = {
            "name": col,
            "dtype": str(df[col].dtype),
            "non_null_count": int(df[col].count()),
            "null_count": int(df[col].isnull().sum()),
            "unique_count": int(df[col].nunique())
        }

        if df[col].dtype == 'object' or df[col].nunique() < 20:
            try:
                unique_vals = df[col].dropna().unique()[:10].tolist()
                col_metadata["sample_values"] = [str(v) for v in unique_vals]
            except Exception:
                col_metadata["sample_values"] = []

        elif pd.api.types.is_numeric_dtype(df[col]):
            try:
                col_metadata["min"] = float(df[col].min())
                col_metadata["max"] = float(df[col].max())
                col_metadata["mean"] = float(df[col].mean())
            except Exception:
                pass

        metadata["columns"].append(col_metadata)

    return metadata

def generate_pandas_code_multisheet(
    question: str,
    relevant_sheets_data: List[Dict[str, Any]],  # FIX C: now takes actual sheet data with df
    llm_params: Dict[str, Any],
    authorization: str,
    previous_code: Optional[str] = None,
    previous_error: Optional[str] = None
) -> str:
    """
    FIX C + D: Generate pandas code using live schema built from actual cleaned dataframes.
    Supports retry mode: if previous_code and previous_error are provided, asks LLM to fix the code.
    """
    # Build rich schema from actual dataframes (not stale metadata)
    schemas_str = ""
    for idx, sheet_data in enumerate(relevant_sheets_data):
        var_name = f"df_{idx+1}"
        identifier = sheet_data['sheet_name'] if sheet_data.get('sheet_name') else "Sheet1"
        pivot_context = sheet_data.get('pivot_context')          # NEW

        schemas_str += build_dataframe_schema_for_prompt(
            sheet_data['df'], var_name, identifier, pivot_context   # NEW
        )
        schemas_str += "\n\n"

    # Build base rules
    base_rules = f"""You are a Python pandas expert. Generate ONLY executable pandas code to answer the user's question.

Available DataFrames (with live schema from cleaned data):
{schemas_str}

User Question: {question}

CRITICAL RULES:
1. DataFrames are available as: {', '.join([f'df_{i+1}' for i in range(len(relevant_sheets_data))])}
2. Generate ONLY executable Python/pandas code. No explanations, no markdown, no comments.
3. Store the FINAL result in a variable called 'result'.
4. The result must directly answer the question — a scalar, DataFrame, Series, or dict.
5. QUERY INTENT & FILTER COMPLETENESS RULE:
   - The user's question is the SOLE source of filter conditions. Every entity, value, category, 
     time period, ID, or name mentioned in the question MUST be applied as an explicit filter in 
     the code — nothing may be omitted or assumed to be already handled.
   - Do NOT invent filter conditions that are not in the question, and do NOT drop conditions that are.

STRING VALUE RULES (IMPORTANT):
6. ALL string/categorical column values have been normalized to LOWERCASE during preprocessing.
   ALWAYS use lowercase when filtering with ==. Example: df[df['status'] == 'open'] NOT 'Open'
7. For fuzzy/partial text matches, use .str.contains('keyword', case=False, na=False)
8. When the user mentions a value like "Active", filter with 'active'. "New York" → 'new york'.
9. For string comparisons, prefer: .str.lower().str.strip() == 'target_value' as a safety measure.

COLUMN NAME RULES:
10. Use EXACT column names as listed in the schema above (they are case-sensitive).
   The column names shown are the actual names after cleaning — use them verbatim.
11. If a user refers to a column ambiguously (e.g., "revenue"), map it to the closest column name
    from the schema. Do NOT invent column names.


MULTI-DATAFRAME RULE: 
-When multiple DataFrames are provided (df_1, df_2, ...), you MUST use ALL of them.
-If they have the same schema (same columns), always combine them first: 
  combined = pd.concat([df_1, df_2, ...], ignore_index=True) 
-Then operate on `combined`. Never query only one DataFrame when multiple are provided. 
-The only exception is if the question explicitly names a specific file/sheet."

NULL HANDLING RULES:
12. Always handle nulls in aggregations: use .dropna() before aggregating, or skipna=True.
13. For groupby aggregations, chain .dropna() on the groupby key column before groupby.
14. If a filter might return an empty DataFrame, add a fallback:
    result = filtered_df if not filtered_df.empty else "No records found matching the criteria"

AGGREGATION & ANALYSIS RULES:
15. For counting: use .value_counts(), .groupby().size(), or len()
16. For aggregations: use .agg(), .groupby()[col].mean(), .sum(), etc.
17. For merging multiple dataframes: use pd.merge() with appropriate keys, or pd.concat()
18. For date operations: ensure the column is datetime dtype before using .dt accessor
19. For ranking/top-N: use .nlargest(n) or .sort_values().head(n)
20. For percentage calculations: divide by .sum() and multiply by 100
21. Always return unique/distinct values unless the user explicitly asks for "all records", "list all", or "show duplicates". For counts use .nunique(), for lists use .unique() or .drop_duplicates() on the relevant identifying column.

COMPLEX QUERY HANDLING:
22. If the question involves multiple steps (filter → aggregate → sort), do them sequentially.
23. If the question asks for "trend" or "over time", group by the date column and aggregate.
24. If the question asks for "comparison", compute values for each group separately.
25. If the question is ambiguous about which column to use, pick the most semantically relevant one.

OUTPUT FORMAT:
26. If result is a DataFrame with many rows (>20), it's acceptable — don't truncate.
27. If the result is a single number, assign it directly: result = df['col'].sum()

DATE FILTERING RULES:
28. For any time-series or trend analysis, ALWAYS filter out future dates before plotting.
    Apply this filter on any datetime column used as x-axis:
    df = df[df['<date_col>'] <= pd.Timestamp.today()]
    Never plot data points beyond today's date.

DATE PARSING RULES:
- When converting date columns, ALWAYS pass dayfirst=True if dates appear to be in DD/MM/YYYY format:
  df['<date_col>'] = pd.to_datetime(df['<date_col>'], errors='coerce', dayfirst=True)
- The preprocessing pipeline normalizes text to lowercase but does NOT fix date parsing order.
  Always inspect sample values in the schema to determine if day comes first.

{REASONING_SECTION_PROMPT}"""

    if previous_code and previous_error:
        # RETRY MODE: Ask LLM to fix the broken code
        prompt = f"""{base_rules}

--- PREVIOUS ATTEMPT (FAILED) ---
The following code was generated but produced an error. Fix it.

Previous code:
```python
{previous_code}
```

Error encountered:
{previous_error}

INSTRUCTIONS FOR FIX:
- Identify why the error occurred based on the schema and error message
- Check column names are exact matches to the schema
- Check string values are lowercase
- Check null handling is correct
- Generate the corrected, complete code below (no markdown, no backticks):"""
    else:
        prompt = f"""{base_rules}

Generate the pandas code now (no markdown, no backticks, just executable Python):"""

    code = llm_client.generate(prompt, llm_params, token_tracker, authorization, temperature=0.0)
    
    # Clean the generated code
    code = code.strip()
    code = re.sub(r'^```python\s*\n', '', code)
    code = re.sub(r'^```\s*\n', '', code)
    code = re.sub(r'\n```$', '', code)
    code = re.sub(r'```$', '', code)
    code = code.strip()
    
    return code


def execute_pandas_code_multisheet(code: str, dataframes_dict: Dict[str, pd.DataFrame]) -> tuple[Any, Optional[str]]:
    try:
        namespace = {
            'pd': pd,
            'np': np,
            'result': None
        }
        namespace.update(dataframes_dict)
        exec(code, namespace)
        result = namespace.get('result')

        if result is None:
            return None, "Code executed but no result was produced. Variable 'result' is None."

        # NEW: detect integer-indexed pivot/DataFrame that would produce blank heatmap
        if isinstance(result, pd.DataFrame):
            index_is_integer = pd.api.types.is_integer_dtype(result.index.dtype)
            cols_are_integer = pd.api.types.is_integer_dtype(result.columns.dtype)
            if index_is_integer and cols_are_integer and result.shape[0] > 0:
                return None, (
                    "Result DataFrame has integer index and integer columns — "
                    "this looks like an unformatted pivot/heatmap table. "
                    "Use pd.crosstab(df['col1'], df['col2']) instead of pivot_table. "
                    "Ensure heat_df.index = heat_df.index.astype(str) and "
                    "heat_df.columns = heat_df.columns.astype(str) before px.imshow()."
                )
        return result, None

    except Exception as e:
        error_msg = f"Error executing pandas code: {str(e)}"
        return None, error_msg


def execute_pandas_with_retry(
    question: str,
    relevant_sheets_data: List[Dict[str, Any]],
    llm_params: Dict[str, Any],
    authorization: str,
    max_retries: int = MAX_CODE_RETRIES
) -> tuple[Any, Optional[str], str]:
    """
    FIX C (retry loop): Generate pandas code, execute it, and retry on failure.
    Returns (result, error, final_code).
    On success: (result, None, code)
    On all retries exhausted: (None, last_error, last_code)
    """
    dataframes_dict = {f'df_{idx+1}': s['df'] for idx, s in enumerate(relevant_sheets_data)}
    
    previous_code = None
    previous_error = None
    
    for attempt in range(max_retries + 1):
        if attempt == 0:
            logger.info(f"  [Code Gen] Attempt {attempt + 1}/{max_retries + 1}: Generating initial code...")
        else:
            logger.info(f"  [Code Gen] Attempt {attempt + 1}/{max_retries + 1}: Retrying with error feedback...")
        
        code = generate_pandas_code_multisheet(
            question=question,
            relevant_sheets_data=relevant_sheets_data,
            llm_params=llm_params,
            authorization=authorization,
            previous_code=previous_code,
            previous_error=previous_error
        )
        
        logger.info(f"  Generated code:\n{code}")
        
        result, error = execute_pandas_code_multisheet(code, dataframes_dict)
        
        if error is None:
            logger.info(f"  ✓ Code execution successful on attempt {attempt + 1}")
            return result, None, code
        
        logger.info(f"  ✗ Code execution failed on attempt {attempt + 1}: {error}")
        previous_code = code
        previous_error = error
    
    # All retries exhausted
    return None, previous_error, previous_code

def _build_metadata_str(all_sheets_metadata: List[Dict[str, Any]]) -> str:
    """Build metadata string for LLM prompt."""
    metadata_str = "Available Files/Sheets:\n\n"
    for idx, sheet_meta in enumerate(all_sheets_metadata, 1):
        identifier = sheet_meta['file_name']
        if sheet_meta.get('sheet_name'):
            identifier += f" > Sheet: {sheet_meta['sheet_name']}"
        metadata_str += f"Source {idx}: {identifier}\n"
        for col in sheet_meta['columns']:
            col_info = f"    - {col['name']} ({col['dtype']})"
            if 'sample_values' in col and col['sample_values']:
                col_info += f" | Samples: {col['sample_values'][:5]}"
            metadata_str += col_info + "\n"
        metadata_str += "\n"
    return metadata_str


def _parse_llm_json_response(response: str) -> dict:
    """Strip markdown fences and parse JSON from LLM response."""
    cleaned = response.strip()
    if cleaned.startswith('```json'):
        cleaned = cleaned.split('```json')[1].split('```')[0].strip()
    elif cleaned.startswith('```'):
        cleaned = cleaned.split('```')[1].split('```')[0].strip()
    return json.loads(cleaned)


def _validate_sources(relevant_sources_raw: list, all_sheets_metadata: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Validate and filter relevant sources against known metadata."""
    validated = []
    for source in relevant_sources_raw:
        file_name = source.get('file_name')
        sheet_name = source.get('sheet_name')
        for meta in all_sheets_metadata:
            if meta['file_name'] == file_name and meta.get('sheet_name') == sheet_name:
                validated.append({"file_name": file_name, "sheet_name": sheet_name})
                break
    return validated


def classify_intent_and_identify_sheets(
    question: str,
    all_sheets_metadata: List[Dict[str, Any]],
    llm_params: Dict[str, Any],
    authorization: str
) -> tuple[str, List[Dict[str, str]]]:
    """
    Unified classifier that detects both intent mode AND relevant sheets in one LLM call.
    
    Modes:
    - "plot"     : user wants a chart or visualization
    - "qna"      : user wants a specific factual answer from the data
    - "insights" : user wants patterns, trends, open-ended analysis
    - "goal"     : user states a business objective they want to achieve
                   e.g. "I want to reduce churn", "help me improve delivery times"
    """
    metadata_str = _build_metadata_str(all_sheets_metadata)

    prompt = f"""You are a data analytics assistant. Classify the user's input and identify relevant data sources.

{metadata_str}

User Input: {question}

TASK 1 — Classify the mode:
- "plot"     → user explicitly asks for a chart, graph, visualization, or says "show me", "visualize", "plot"
- "qna"      → user asks a specific factual question about the data: counts, averages, lists, rankings, filters
- "insights" → user asks for patterns, trends, summary, analysis, or open-ended exploration of the data
- "goal"     → user states a forward-looking business objective or problem they want to solve.
               Signs of goal mode: phrases like "I want to", "help me", "how can I", "we need to",
               "reduce X", "improve Y", "increase Z", "optimize", "achieve", "our goal is".
               Goal inputs are typically short, business-oriented, and NOT asking about a specific data value.
               Example goals: "I want to reduce customer churn", "help me improve sales performance",
               "we need to optimize delivery times", "I want to grow revenue in Q4"
               Example non-goals (these are qna/insights): "what is the churn rate?",
               "show me sales trends", "which region has the best delivery time?"

TASK 2 — Identify relevant sources:
- Include only sources whose columns relate to the user input
- For goal mode, be generous — include all sources that could possibly help address the goal
- If none are clearly relevant, return empty array

Respond ONLY with valid JSON:
{{
  "mode": "plot|qna|insights|goal",
  "relevant_sources": [
    {{"file_name": "file1.xlsx", "sheet_name": "Sheet1"}},
    {{"file_name": "file2.csv", "sheet_name": null}}
  ]
}}

{REASONING_SECTION_PROMPT}"""

    response = llm_client.generate(prompt, llm_params, token_tracker, authorization, temperature=0.0)

    try:
        parsed = _parse_llm_json_response(response)
        mode = parsed.get('mode', 'qna').lower()
        if mode not in ['plot', 'qna', 'insights', 'goal']:
            mode = 'qna'
        validated = _validate_sources(parsed.get('relevant_sources', []), all_sheets_metadata)
        return mode, validated
    except Exception:
        return 'qna', []


def _result_to_string(result: Any) -> str:
    """Convert a pandas result to a display string."""
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return "No data found."
        if len(result) > 20:
            return f"{len(result)} records found. Top 20:\n{result.head(20).fillna('N/A').to_string()}"
        return _safe_fillna(result).to_string()
    if isinstance(result, pd.Series):
        return result.head(20).fillna('N/A').to_string() if len(result) > 20 else _safe_fillna(result).to_string()
    return str(result)


def _parse_sub_questions(raw: str) -> List[str]:
    """Strip markdown fences and parse sub-questions JSON array."""
    raw = re.sub(r'^```json\s*\n', '', raw)
    raw = re.sub(r'^```\s*\n', '', raw)
    raw = re.sub(r'\n```$', '', raw).strip()
    sub_questions = json.loads(raw)
    if not isinstance(sub_questions, list):
        raise ValueError("Not a list")
    return [q for q in sub_questions if isinstance(q, str)][:4]


def _run_sub_questions(
    sub_questions: List[str],
    relevant_sheets_data: List[Dict[str, Any]],
    llm_params: Dict[str, Any],
    authorization: str
) -> List[Dict[str, str]]:
    """Execute each sub-question and collect results."""
    sub_results = []
    for i, sub_q in enumerate(sub_questions):
        logger.info(f"  Running sub-question {i+1}/{len(sub_questions)}: {sub_q}")
        try:
            result, error, _ = execute_pandas_with_retry(
                question=sub_q,
                relevant_sheets_data=relevant_sheets_data,
                llm_params=llm_params,
                authorization=authorization
            )
            if error or result is None:
                logger.warning(f"  Sub-question {i+1} failed: {error}")
                continue
            sub_results.append({"question": sub_q, "result": _result_to_string(result)})
        except Exception as e:
            logger.error(f"  Sub-question {i+1} exception: {e}", exc_info=True)
    return sub_results

def generate_goal_oriented_analysis(
    goal: str,
    relevant_sheets_data: List[Dict[str, Any]],
    llm_params: Dict[str, Any],
    authorization: str
) -> str:
    """
    Takes a business goal and decomposes it into data-driven questions,
    runs each through the pandas retry pipeline, and returns a structured
    goal-focused analysis.
    """
    schema_lines = [
        build_dataframe_schema_for_prompt(
            s['df'],
            f"df_{idx+1}",
            s['sheet_name'] if s.get('sheet_name') else "Sheet1",
            s.get('pivot_context')
        )
        for idx, s in enumerate(relevant_sheets_data)
    ]
    schema_str = "\n\n".join(schema_lines)

    # Step 1: Decompose goal into measurable data questions
    decomposition_prompt = f"""You are a senior data analyst. A business user has stated a goal they want to achieve.
Your job is to translate that goal into 4-5 specific, measurable questions that can be answered from the available data.

Business Goal: {goal}

Available Data Schema:
{schema_str}

RULES:
- Each question must be directly answerable using pandas on the data above
- Questions should together give a complete picture of the goal — covering current state, problem areas, top performers, and trends
- Focus on: identifying root causes, finding segments that drive the problem, quantifying the gap
- Do NOT generate generic questions — every question must reference actual column names from the schema
- If the goal cannot be meaningfully addressed with this data, return an empty array

Respond with ONLY a JSON array of 4-5 strings. Example:
["What is the current average X across all records?", "Which top 5 segments have the highest Y?", "How does Z vary across categories of A?", "What percentage of records fall below threshold B?"]

Your response:"""

    raw = llm_client.generate(decomposition_prompt, llm_params, token_tracker, authorization, temperature=0.0).strip()

    try:
        questions = _parse_sub_questions(raw)  # reuse your existing parser
    except Exception as e:
        logger.warning(f"Goal decomposition failed to parse: {e}")
        return "I couldn't decompose your goal into answerable questions from the available data. Please try rephrasing or ensure your data contains relevant columns."

    if not questions:
        return "The data available doesn't seem to contain columns relevant to your stated goal. Please check if the correct file was uploaded."

    logger.info(f"Goal '{goal}' decomposed into {len(questions)} questions: {questions}")

    # Step 2: Run each question through existing pandas retry pipeline
    sub_results = _run_sub_questions(questions, relevant_sheets_data, llm_params, authorization)

    if not sub_results:
        return "I wasn't able to extract meaningful data to address your goal. Please try uploading more relevant data."

    # Step 3: Stitch into a goal-focused report
    sub_results_str = "".join(
        f"\nFinding {i}: {sr['question']}\nResult:\n{sr['result']}\n"
        for i, sr in enumerate(sub_results, 1)
    )

    stitch_prompt = f"""You are a senior business analyst. A user has stated a business goal, and data analysis has been run to help address it.
Synthesize the findings into a structured, actionable goal achievement report.

User's Business Goal: {goal}

Data Findings:
{sub_results_str}

Write your report in this exact structure:

**Goal Assessment:**
[1-2 sentences on whether the data supports addressing this goal and what the current state looks like]

**Key Findings:**
[4-5 bullet points — each must include a specific number or percentage from the findings]

**Problem Areas:**
[2-3 bullet points identifying where the biggest gaps or risks are, with data to back it up]

**Recommended Actions:**
[3-4 concrete, prioritized actions the user should take, ordered by potential impact]

**What to Measure Next:**
[2-3 specific metrics or follow-up analyses the user should track to monitor progress toward this goal]

RULES:
- Never mention pandas, dataframes, or technical terms
- Use specific numbers from the findings throughout
- Be direct and business-focused
- If findings are insufficient to address part of the goal, say so briefly rather than fabricating"""

    return llm_client.generate(stitch_prompt, llm_params, token_tracker, authorization, temperature=0.1)

def generate_insights(
    question: str,
    relevant_sheets_data: List[Dict[str, Any]],
    llm_params: Dict[str, Any],
    authorization: str
) -> str:
    schema_lines = [
        build_dataframe_schema_for_prompt(
            s['df'],
            f"df_{idx+1}",
            s['sheet_name'] if s.get('sheet_name') else "Sheet1",
            s.get('pivot_context')          # NEW
        )
        for idx, s in enumerate(relevant_sheets_data)
    ]
    schema_str = "\n\n".join(schema_lines)

    sub_question_prompt = f"""You are a data analyst. A user wants insights from their data.

Data Schema:
{schema_str}

User's Request: {question}

Generate exactly 4 specific, targeted analytical sub-questions that together would give a comprehensive answer to the user's request.
Each sub-question must be answerable using pandas on the data above.
Focus on: distributions, top/bottom performers, comparisons, trends or anomalies.

Respond with ONLY a JSON array of 4 strings. Example:
["What is the distribution of X?", "Which top 5 Y have the highest Z?", "How does A compare across B categories?", "What is the trend of C over time?"]

Your response:"""

    sub_questions_raw = llm_client.generate(sub_question_prompt, llm_params, token_tracker, authorization, temperature=0.0).strip()

    try:
        sub_questions = _parse_sub_questions(sub_questions_raw)
    except Exception as e:
        logger.warning(f"Failed to parse sub-questions, falling back to single qna: {e}")
        result, error, _ = execute_pandas_with_retry(
            question=question, relevant_sheets_data=relevant_sheets_data,
            llm_params=llm_params, authorization=authorization
        )
        if error:
            return "I encountered an issue generating insights for your data. Please try a more specific question."
        return format_result_for_llm(result, question, llm_params, authorization)

    logger.info(f"Generated {len(sub_questions)} sub-questions for insights: {sub_questions}")

    sub_results = _run_sub_questions(sub_questions, relevant_sheets_data, llm_params, authorization)

    if not sub_results:
        return "I wasn't able to extract meaningful insights from the data. Please try a more specific question."

    sub_results_str = "".join(
        f"\nAnalysis {i}: {sr['question']}\nResult:\n{sr['result']}\n"
        for i, sr in enumerate(sub_results, 1)
    )

    stitch_prompt = f"""You are a senior data analyst. Below are the results of several targeted analyses run on the data to answer the user's request.

User's Original Request: {question}

Analysis Results:
{sub_results_str}

Write a cohesive, professional insights report based on these results. Structure it as:

**Key Insights:**
[3-5 bullet points of the most important findings, each 1-2 sentences, with specific numbers]

**Detailed Analysis:**
[2-3 short paragraphs expanding on the key findings, connecting patterns across the analyses]

**Recommendations:**
[2-3 actionable recommendations based on the data]

RULES:
- Use specific numbers and percentages from the results
- Do not mention technical details like dataframes, pandas, or code
- Write for a business audience
- Be concise and direct"""

    return llm_client.generate(stitch_prompt, llm_params, token_tracker, authorization, temperature=0.1)

def _safe_fillna(df: Any, value: str = 'N/A') -> Any:
    """Convert categorical columns to object before fillna to avoid TypeError."""
    if isinstance(df, pd.DataFrame):
        cat_cols = df.select_dtypes(['category']).columns.tolist()
        if cat_cols:
            df = df.copy()
            df[cat_cols] = df[cat_cols].astype(object)
    return df.fillna(value)

def format_result_for_llm(result: Any, question: str, llm_params: Dict[str, Any], authorization: str) -> str:
    """
    FIX E: Format the pandas result into a readable answer.
    Temperature lowered to 0.1 for more consistent answers.
    """
    if result is None:
        return "No results found."
    
    if isinstance(result, pd.DataFrame):
        if len(result) == 0:
            result_str = "Empty DataFrame - no records match the criteria."
        elif len(result) > 50:
            result_str = f"Found {len(result)} records. Here are the first 50:\n{_safe_fillna(result.head(50)).to_string()}"
        else:
            result_str = _safe_fillna(result).to_string()
    elif isinstance(result, pd.Series):
        if len(result) == 0:
            result_str = "No data found."
        elif len(result) > 50:
            result_str = f"Found {len(result)} items. Here are the first 50:\n{result.head(50).fillna('N/A').to_string()}"
        else:
            result_str = _safe_fillna(result).to_string()
    elif isinstance(result, (dict, list)):
        result_str = str(result)
    else:
        result_str = str(result)
    
    prompt = f"""You are a helpful data analyst. Based on the query result below, provide a comprehensive response that includes both a direct answer and actionable insights.

User Question: {question}

Query Result:
{result_str}

RESPONSE FORMAT:
Provide your response in TWO sections:

**Answer:**
[Provide a detailed, complete answer to the user's question. Do NOT summarize or truncate — include every relevant record, value, and number from the result. If the result is a list or dataframe, list all items with their full details rather than just the highlights. Format using markdown: use tables for structured data, bullet points for lists, and bold for important numbers.]

**Insights:**
[Provide 2-4 actionable insights based on the data. Focus on:
- Patterns or trends you notice
- Notable observations (highest/lowest values, concentrations)
- Potential implications for business decisions
Keep insights concise (1-2 sentences each) and business-focused.]

# New
CRITICAL RULES:
- Use clear, professional language
- Be specific with numbers and percentages
- Do not provide any technical details about data, like percentile, standard deviation, etc.
- Focus on what matters to the user
- Make insights actionable and relevant
- Format your entire response in proper markdown — use **bold** for key values, bullet points for lists, and tables where the data suits it

{REASONING_SECTION_PROMPT}"""
    
    # FIX E: Use temperature=0.1 instead of 0.5 for consistent answers
    answer = llm_client.generate(prompt, llm_params, token_tracker, authorization, temperature=0.1)
    return answer


def clean_sheet_name(sheet_name: str) -> str:
    """Remove unwanted characters and whitespace from sheet names"""
    cleaned = ''.join(char for char in sheet_name if char.isprintable())
    cleaned = cleaned.strip()
    # Only strip UUID-style suffixes (8-4-4-4-12 hex format), not regular words
    cleaned = re.sub(r'[_\s]+[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$', '', cleaned, flags=re.IGNORECASE)
    return cleaned

def detect_header_row(excel_file: pd.ExcelFile, sheet_name: str, max_scan_rows: int = 15) -> int:
    try:
        df_raw = pd.read_excel(excel_file, sheet_name=sheet_name, header=None, nrows=max_scan_rows)
    except Exception:
        return 0
    best_row, best_score = 0, -1
    for i, row in df_raw.iterrows():
        score = sum(1 for v in row if pd.notna(v) and isinstance(v, str) and v.strip() != "")
        if score > best_score:
            best_score = score
            best_row = i
    return best_row if best_row > 0 else 0


def is_default_header(df: pd.DataFrame) -> bool:
    unnamed_count = sum(1 for col in df.columns if str(col).startswith("Unnamed:"))
    return unnamed_count > len(df.columns) * 0.5


def check_sheet_has_pivot(wb, sheet_name: str) -> bool:
    try:
        return len(wb[sheet_name]._pivots) > 0
    except Exception:
        return False

def extract_pivot_filter_context(excel_file: pd.ExcelFile, sheet_name: str, pivot_range: dict) -> dict:
    """
    Parse rows above the pivot table and the header rows for filter context metadata.
    """
    context = {
        "pivot_filters": {},
        "pivot_metric": None,
        "pivot_row_field": None,
        "pivot_col_field": None,
    }
    rows_above = pivot_range["min_row"] - 1
    if rows_above < 1:
        _populate_pivot_header_context(context, excel_file, sheet_name, pivot_range)
        return context

    _populate_pivot_filters(context, excel_file, sheet_name, rows_above)
    _populate_pivot_header_context(context, excel_file, sheet_name, pivot_range)
    return context


def _populate_pivot_filters(context: dict, excel_file: pd.ExcelFile, sheet_name: str, rows_above: int) -> None:
    """Scan rows above the pivot for filter key-value pairs."""
    try:
        df_above = pd.read_excel(excel_file, sheet_name=sheet_name, header=None, nrows=rows_above)
        for _, row in df_above.iterrows():
            non_null = row.dropna()
            if len(non_null) == 2:
                key, val = str(non_null.iloc[0]).strip(), str(non_null.iloc[1]).strip()
                if key:
                    context["pivot_filters"][key] = val
    except Exception as e:
        logger.warning(f"  ⚠ Could not extract pivot filter context for '{sheet_name}': {e}")


def _populate_pivot_header_context(context: dict, excel_file: pd.ExcelFile, sheet_name: str, pivot_range: dict) -> None:
    """Extract metric and dimension information from the pivot table's descriptor rows."""
    try:
        df_header = pd.read_excel(
            excel_file, sheet_name=sheet_name, skiprows=pivot_range["min_row"] - 1,
            nrows=2, header=None, usecols=range(pivot_range["min_col"] - 1, pivot_range["max_col"])
        )
        PIVOT_DESCRIPTOR_KEYWORDS = ("column labels", "row labels", COUNT_OF, SUM_OF, AVERAGE_OF, MAX_OF, MIN_OF)
        row0_vals = [str(v).strip() for v in df_header.iloc[0].tolist() if str(v).strip() != 'nan']
        row1_vals = [str(v).strip() for v in df_header.iloc[1].tolist() if str(v).strip() != 'nan']

        if any(any(kw in v.lower() for kw in PIVOT_DESCRIPTOR_KEYWORDS) for v in row0_vals):
            for val in row0_vals:
                val_lower = val.lower()
                for kw in (COUNT_OF, SUM_OF, AVERAGE_OF, MAX_OF, MIN_OF):
                    if val_lower.startswith(kw):
                        context["pivot_metric"] = val
                        break
            if row1_vals:
                context["pivot_row_field"] = row1_vals[0]
                context["pivot_col_field"] = ", ".join(row1_vals[1:])
    except Exception as e:
        logger.warning(f"  ⚠ Could not extract pivot header context for '{sheet_name}': {e}")

def extract_pivot_ranges(wb_full, sheet_name: str) -> list:
    """
    Accepts an already-loaded non-readonly openpyxl workbook.
    Caller is responsible for loading and closing it.
    Returns list of pivot range dicts for the given sheet.
    """
    if wb_full is None:
        return []
    try:
        from openpyxl.utils import range_boundaries
        ws = wb_full[sheet_name]
        if not ws._pivots:
            return []
        ranges = []
        for pivot in ws._pivots:
            ref = pivot.location.ref
            min_col, min_row, max_col, max_row = range_boundaries(ref)
            ranges.append({
                "min_row": min_row,
                "max_row": max_row,
                "min_col": min_col,
                "max_col": max_col,
                "ref": ref
            })
        return ranges
    except Exception as e:
        logger.warning(f"  ⚠ Could not extract pivot ranges for '{sheet_name}': {e}")
        return []

def read_pivot_as_dataframe(excel_file: pd.ExcelFile, sheet_name: str, pivot_range: dict) -> pd.DataFrame:
    """
    Read a single pivot table from a sheet using its exact row range.
    Handles double-row pivot headers (descriptor row + actual column names row).
    """
    PIVOT_DESCRIPTOR_KEYWORDS = ("column labels", "row labels", COUNT_OF, SUM_OF, AVERAGE_OF, MAX_OF, MIN_OF)

    try:
        header_row_0based = pivot_range["min_row"] - 1  # openpyxl is 1-based
        data_rows = pivot_range["max_row"] - pivot_range["min_row"]
        usecols = range(pivot_range["min_col"] - 1, pivot_range["max_col"])

        # Read first 2 rows to detect double-row header
        df_peek = pd.read_excel(
            excel_file,
            sheet_name=sheet_name,
            skiprows=header_row_0based,
            nrows=2,
            header=None,
            usecols=usecols
        )

        # Check if row 0 is a descriptor row (e.g. "Count of Time Sheet State | Column Labels")
        row0_vals = [str(v).lower().strip() for v in df_peek.iloc[0].dropna().tolist()]
        is_descriptor_row = any(
            any(kw in cell for kw in PIVOT_DESCRIPTOR_KEYWORDS)
            for cell in row0_vals
        )

        if is_descriptor_row:
            # Row 0 = descriptor, Row 1 = actual column names → skip one extra row
            actual_header_skip = header_row_0based + 1
            actual_data_rows = data_rows - 1
        else:
            actual_header_skip = header_row_0based
            actual_data_rows = data_rows

        df = pd.read_excel(
            excel_file,
            sheet_name=sheet_name,
            skiprows=actual_header_skip,
            nrows=actual_data_rows,
            usecols=usecols
        )
        return flatten_pivot_dataframe(df)

    except Exception as e:
        logger.warning(f"  ⚠ Failed reading pivot range {pivot_range['ref']}: {e}")
        return pd.DataFrame()
    
def flatten_pivot_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how='all').dropna(axis=1, how='all')

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            ' '.join(str(c) for c in col if str(c) != 'nan').strip()
            for col in df.columns
        ]

    # Drop Grand Total summary row (pivot artifact)
    if len(df.columns) > 0:
        first_col = df.columns[0]
        grand_total_mask = df[first_col].astype(str).str.strip().str.lower() == 'grand total'
        if grand_total_mask.any():
            df = df[~grand_total_mask]
            logger.info("  ✓ Dropped 'Grand Total' summary row from pivot")

    # Only ffill first column if it looks like merged cells (sparse unique values)
    if len(df.columns) > 0:
        first_col = df.columns[0]
        non_null_count = df[first_col].notna().sum()
        unique_count = df[first_col].nunique()

        if non_null_count > 0 and unique_count / non_null_count < 0.5:
            df[first_col] = df[first_col].ffill()
            logger.info(f"  ✓ ffill applied to '{first_col}' (merged cell pattern detected)")
        else:
            logger.info(f"  ✓ ffill skipped for '{first_col}' (not a merged cell pattern)")

    return df.reset_index(drop=True)


def _process_multiple_pivots(excel_file, sheet_name: str, pivot_ranges: list, cleaned_name: str) -> Dict[str, pd.DataFrame]:
    """Read multiple pivot tables from a sheet, return dict of key -> (df, pivot_context)."""
    logger.info(f"  ℹ Sheet '{sheet_name}' has {len(pivot_ranges)} pivot tables — splitting")
    result = {}
    for i, prange in enumerate(pivot_ranges, 1):
        pivot_context = extract_pivot_filter_context(excel_file, sheet_name, prange)
        pivot_df = read_pivot_as_dataframe(excel_file, sheet_name, prange)
        if not pivot_df.empty:
            pivot_key = f"{cleaned_name}_pivot_{i}"
            result[pivot_key] = (pivot_df, pivot_context)
            logger.info(f"    ✓ Pivot {i} extracted as '{pivot_key}' ({len(pivot_df)} rows)")
    return result


def _process_single_pivot(excel_file, sheet_name: str, pivot_ranges: list, cleaned_name: str) -> Dict[str, pd.DataFrame]:
    """Read a single pivot table, fallback to full sheet read if needed."""
    logger.info(f"  ℹ Sheet '{sheet_name}' contains a pivot table — flattening")

    pivot_context = extract_pivot_filter_context(excel_file, sheet_name, pivot_ranges[0])
    pivot_df = read_pivot_as_dataframe(excel_file, sheet_name, pivot_ranges[0])

    if not pivot_df.empty:
        return {cleaned_name: (pivot_df, pivot_context)}

    logger.warning(f"  ⚠ Pivot range extraction failed for '{sheet_name}', falling back to full sheet read")
    df = pd.read_excel(excel_file, sheet_name=sheet_name, header=0)
    return {cleaned_name: (flatten_pivot_dataframe(df), {})}


_MAX_ROWS_PER_SHEET = 50000

def _process_plain_sheet(excel_file, sheet_name: str, cleaned_name: str) -> Dict[str, tuple]:
    """Read a plain sheet using header detection. Hard-caps rows at _MAX_ROWS_PER_SHEET."""
    header_row = detect_header_row(excel_file, sheet_name)
    df = pd.read_excel(
        excel_file,
        sheet_name=sheet_name,
        header=header_row,
        nrows=_MAX_ROWS_PER_SHEET,
    )
    if len(df) >= _MAX_ROWS_PER_SHEET:
        logger.warning(
            "Sheet '%s' capped at %d rows — file may be larger.",
            sheet_name, _MAX_ROWS_PER_SHEET
        )
    if is_default_header(df):
        logger.info("Sheet '%s' still has unnamed headers — retrying with row 0.", sheet_name)
        df = pd.read_excel(
            excel_file,
            sheet_name=sheet_name,
            header=0,
            nrows=_MAX_ROWS_PER_SHEET,
        )
    return {cleaned_name: (df, None)}

def _parse_excel(content_buffer: io.BytesIO) -> Dict[str, Any]:
    """
    Parse all sheets from an Excel file.
    Pivot detection removed — all sheets treated as plain data.
    Use header detection + flatten for all sheets.
    """
    excel_file = pd.ExcelFile(content_buffer)

    all_sheets: Dict[str, Any] = {}
    pivot_contexts: Dict[str, Any] = {}

    for sheet_name in excel_file.sheet_names:
        results = _process_plain_sheet(excel_file, sheet_name, clean_sheet_name(sheet_name))
        for key, (df, ctx) in results.items():
            all_sheets[key] = df
            pivot_contexts[key] = ctx  # always None on this path

    return {
        "file_type": "excel",
        "sheets": all_sheets,
        "sheet_names": list(all_sheets.keys()),
        "pivot_contexts": pivot_contexts,
    }


def process_file_with_sheets(content: bytes, file_name: str) -> Dict[str, Any]:
    """Process file content and detect all sheets."""
    try:
        content_buffer = io.BytesIO(content)
        return _parse_excel(content_buffer)
    except Exception:
        try:
            df = pd.read_csv(io.BytesIO(content))
            return {
                "file_type": "csv",
                "sheets": {file_name: df},
                "sheet_names": [file_name]
            }
        except Exception:
            raise ValueError(
                f"Unsupported file format for {file_name}. Please ensure file is Excel or CSV format."
            )

def validate_s3_url(s3_url: str) -> bool:
    """Validate if the URL is a proper S3 URL"""
    return s3_url.startswith(('https://', 's3://'))


async def get_llm_config(user_metadata: str) -> Dict[str, Any]:
    """Get LLM configuration"""
    team_id = None
    try:
        async with get_model_config() as config:
            metadata_dict = json.loads(user_metadata) if user_metadata else {}
            team_id = metadata_dict.get("team_id")
            
            if not team_id:
                raise ValueError("team_id is required in user_metadata")
            
            team_config = await config.get_team_model_config(team_id)
            model = team_config["selected_model"]
            provider = team_config["provider"]
            provider_model = f"{provider}/{model}"
            model_config = team_config["config"]
            
            llm_params = {
                "model": provider_model,
                **model_config
            }
            
            return llm_params
            
    except Exception as e:
        logging.error(f"Failed to create LLM instance for team {team_id}: {str(e)}")
        raise ValueError(f"Failed to get model configuration for team {team_id}: {str(e)}")


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# @router.get("/heartbeat-health")
# async def heartbeat_health_check():
#     """Heartbeat health endpoint - sends heartbeat signal to license backend"""
#     try:
#         from src.utils.heartbeat import heartbeat_client
        
#         # Send heartbeat signal
#         await heartbeat_client.send_heartbeat(status="healthy")
        
#         return {
#             "status": "healthy",
#             "server_name": heartbeat_client.agent_name,
#             "heartbeat_sent": True,
#             "timestamp": datetime.utcnow().isoformat(),
#             "message": "Data Insights backend is running and heartbeat sent to license backend"
#         }
#     except Exception as e:
#         logger.error(f"Heartbeat health check error: {str(e)}")
#         return {
#             "status": "unhealthy",
#             "heartbeat_sent": False,
#             "error": str(e),
#             "timestamp": datetime.utcnow().isoformat()
#         }


def _source_names(sheets: List[Dict]) -> List[str]:
    return [
        f"{s['file_name']}" + (f" > {s['sheet_name']}" if s['sheet_name'] and s['sheet_name'] != s['file_name'] else "")
        for s in sheets
    ]


async def _load_sheets_for_new_upload(
    s3_url_list: List[str],
    session_id: str,
    user_id: str,
    user_metadata: str,
    event_logger,
    authorization: str,
    skip_db_upsert: bool = False,
) -> tuple[List[Dict], List[Dict], Any]:
    """Process new upload: load sheets, store session. Returns (sheets_data, metadata, llm_params)."""
    logger.info(f"NEW UPLOAD: Processing {len(s3_url_list)} S3 files")
    event_logger.log_event(MSG_CONFIGURING_ENGINE, authorization)
    llm_params = await get_llm_config(user_metadata)
    event_logger.log_event(MSG_LOADING_DATA, authorization)

    all_sheets_data, all_sheets_metadata = _load_all_sheets_with_s3_url(s3_url_list)

    if not all_sheets_data:
        event_logger.log_event("Couldn't process the uploaded files...", authorization)
        raise HTTPException(status_code=400, detail="No valid sheets could be processed from the provided S3 URLs")

    if skip_db_upsert:
        logger.info(f"✓ URLs unchanged for session {session_id} — skipping DB upsert")
    else:
        file_info = {
            "total_files": len(s3_url_list),
            "total_sheets": len(all_sheets_data),
            "files": [
                {"file_name": s['file_name'], "sheet_name": s['sheet_name'],
                 "rows": len(s['df']), "columns": len(s['df'].columns)}
                for s in all_sheets_data
            ]
        }
        await create_or_update_session(
            session_id=session_id, user_id=user_id,
            s3_urls=s3_url_list, all_sheets_metadata=all_sheets_metadata, file_info=file_info
        )
        logger.info(f"✓ Session stored in database: {session_id}")
    return all_sheets_data, all_sheets_metadata, llm_params


async def _load_sheets_from_session(
    session_id: str,
    user_metadata: str,
    event_logger,
    authorization: str
) -> tuple[List[str], List[Dict], List[Dict], Any]:
    """Load sheets from existing session. Returns (s3_url_list, sheets_data, metadata, llm_params)."""
    logger.info(f"FOLLOW-UP QUESTION: Loading session {session_id}")
    event_logger.log_event(MSG_CONFIGURING_ENGINE, authorization)
    llm_params = await get_llm_config(user_metadata)

    session_data = await get_session_data(session_id)
    if not session_data:
        event_logger.log_event("No previous session found. Please upload files first...", authorization)
        raise HTTPException(
            status_code=400,
            detail=f"No session found with session_id: {session_id}. Please upload files first before asking questions."
        )

    s3_url_list = session_data['s3_urls']
    all_sheets_metadata = session_data['all_sheets_metadata']
    logger.info(f"✓ Session loaded: {session_id}, Files: {len(s3_url_list)}")

    event_logger.log_event(MSG_LOADING_DATA, authorization)
    all_sheets_data = load_dataframes_from_s3(s3_url_list)

    if not all_sheets_data:
        event_logger.log_event("Couldn't load your previous files...", authorization)
        raise HTTPException(status_code=400, detail="Could not load data from session. Files may have been deleted from S3.")

    logger.info(f"✓ Loaded {len(all_sheets_data)} sheets from S3")
    return s3_url_list, all_sheets_data, all_sheets_metadata, llm_params


def _load_all_sheets_with_s3_url(s3_url_list: List[str]) -> tuple[List[Dict], List[Dict]]:
    """Load, clean, and extract metadata for all sheets. Frees each raw file immediately."""
    all_sheets: List[Dict] = []
    all_metadata: List[Dict] = []

    for s3_url in s3_url_list:
        try:
            content = s3_utility.get_data_from_s3_by_url(s3_url)
            if not content:
                logger.warning("Empty or missing file: %s", s3_url)
                continue

            file_name = s3_utility.extract_filename_from_s3_url(s3_url)
            file_data = process_file_with_sheets(content, file_name)

            # Free raw bytes immediately — pivot detection inside process_file_with_sheets
            # already closed the workbook; only the parsed sheet dicts remain.
            del content
            gc.collect()

            pivot_contexts = file_data.get("pivot_contexts", {})
            sheet_names = file_data["sheet_names"]
            logger.info("File: %s (%s) | Sheets: %s", file_name, file_data["file_type"], sheet_names)

            for sheet_name in sheet_names:
                df = file_data["sheets"].pop(sheet_name, None)  # pop frees reference in file_data
                if df is None or df.empty:
                    logger.warning("Empty sheet '%s' — skipping.", sheet_name)
                    continue

                pivot_context = pivot_contexts.get(sheet_name)
                cleaned_df = comprehensive_dataframe_cleaning(df, file_name, sheet_name)
                del df
                gc.collect()

                metadata = extract_file_metadata(cleaned_df, file_name, sheet_name, pivot_context)
                all_sheets.append({
                    "file_name": file_name,
                    "sheet_name": sheet_name,
                    "df": cleaned_df,
                    "s3_url": s3_url,
                    "pivot_context": pivot_context,
                    "metadata": metadata,
                })
                all_metadata.append(metadata)
                logger.info("  Sheet: %s | %d rows × %d cols%s",
                            sheet_name, len(cleaned_df), len(cleaned_df.columns),
                            " [PIVOT]" if pivot_context else "")

            del file_data
            gc.collect()

        except Exception as e:
            logger.error("Error processing %s: %s", s3_url, e, exc_info=True)

    return all_sheets, all_metadata

def _filter_relevant_sheets(relevant_sources: List[Dict], all_sheets_data: List[Dict]) -> List[Dict]:
    """Match relevant sources to actual sheet data."""
    relevant_sheets_data = []
    for source in relevant_sources:
        for sheet_data in all_sheets_data:
            if sheet_data['file_name'] == source['file_name'] and sheet_data['sheet_name'] == source['sheet_name']:
                relevant_sheets_data.append(sheet_data)
                logger.info(f"  - {source['file_name']} > {source['sheet_name']}")
                break
    return relevant_sheets_data


def _handle_plot_mode(
    question: str,
    relevant_sheets_data: List[Dict],
    llm_params: Dict,
    authorization: str,
    event_logger
) -> tuple[str, Optional[str]]:
    """Generate plot and return (answer, plot_data)."""
    event_logger.log_event("Creating your visualization...", authorization)

    df_for_plot = relevant_sheets_data[0]['df'] if len(relevant_sheets_data) == 1 else _concat_sheets(relevant_sheets_data)

    logger.info(f"Generating plot for: {question}")
    plot_json, _ = plot_generator.create_plot(
        question=question, df=df_for_plot,
        llm_params=llm_params, token_tracker=token_tracker, auth_token=authorization
    )

    if not plot_json:
        event_logger.log_event("Couldn't create the visualization...", authorization)
        return "I apologize, but I couldn't create the visualization. Please try rephrasing your request or ask for a different type of chart.", None

    event_logger.log_event("Uploading your results...", authorization)

    import re as _re
    _sanitized = _re.sub(r'\bNaN\b', 'null', plot_json)
    _sanitized = _re.sub(r'\bInfinity\b', 'null', _sanitized)
    _sanitized = _re.sub(r'\b-Infinity\b', 'null', _sanitized)
    fig = pio.from_json(_sanitized)

    plot_html_content = pio.to_html(fig, include_plotlyjs=True, full_html=True)
    download_filename = f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    plot_s3_url = s3_utility.upload_file(file_content=plot_html_content.encode('utf-8'),
                                          file_name=download_filename, folder="plots")
    plot_presigned_url = s3_utility.generate_presigned_url(plot_s3_url)
    event_logger.log_event("Visualization ready...", authorization)

    names = ', '.join(_source_names(relevant_sheets_data))
    return f"I've created the visualization based on data from: {names}. The plot HTML file is available for download.", plot_presigned_url


def _concat_sheets(relevant_sheets_data: List[Dict]) -> pd.DataFrame:
    try:
        return pd.concat([s['df'] for s in relevant_sheets_data], ignore_index=True)
    except Exception:
        return relevant_sheets_data[0]['df']


def _handle_qna_mode(
    question: str,
    relevant_sheets_data: List[Dict],
    llm_params: Dict,
    authorization: str,
    event_logger
) -> str:
    """Run pandas retry pipeline and return answer."""
    event_logger.log_event("Running your analysis...", authorization)
    logger.info("Generating pandas code for multi-sheet analysis")

    result, error, _ = execute_pandas_with_retry(
        question=question, relevant_sheets_data=relevant_sheets_data,
        llm_params=llm_params, authorization=authorization, max_retries=MAX_CODE_RETRIES
    )

    if error:
        logger.info(f"All code generation attempts failed. Last error: {error}")
        names = ', '.join(_source_names(relevant_sheets_data))
        return f"I analyzed the relevant sheets ({names}) but encountered an issue processing your specific query after multiple attempts. Please try rephrasing your question or provide more details."

    logger.info("✓ Pandas code execution successful")
    event_logger.log_event("Preparing your answer...", authorization)
    return format_result_for_llm(result, question, llm_params, authorization)

ANALYTICS_AGENT_CAPABILITIES = [
    "Answer specific questions about your data",
    "Generate insights, trends, and pattern analysis",
    "Analyze data toward a specific business goal (e.g., reduce churn, improve delivery times)",
    "Decompose business objectives into measurable data findings",
    "Generate prioritized action recommendations based on your data",
    "Find top/bottom performers in your data",
    "Calculate aggregations like sum, average, count",
    "Create histograms and distribution charts",
    "Create bar charts and horizontal bar charts",
    "Create pie and donut charts for low-cardinality data",
    "Create scatter plots for numeric relationships",
    "Create line charts and time series trend plots",
    "Create heatmaps for two categorical columns",
    "Create treemaps for high-cardinality categorical data",
    "Create grouped and stacked bar charts",
    "Generate a comprehensive report with visualizations",
]

@router.post(
    "/analyze-multi",
    responses={
        400: {"description": "Invalid S3 URL, no valid sheets, or missing session"},
        500: {"description": "Internal server error during analysis"}
    }
)
@track_llm_calls(
    name="backend-qna",
    tags=["analysis", "multi-sheet", "session-management"],
    metadata={"version": "1.0"},
    avoided_input_params=["request"]
)
async def analyze_data_from_multiple_s3_files(
    request: Request,
    s3_urls: Annotated[Optional[str], Form(description="S3 URLs - can be JSON array string or comma-separated")] = None,
    question: Annotated[str, Form(description="Question about the data")] = ...,
    user_metadata: Annotated[str, Form(description="User metadata in JSON format")] = ...,
):
    """Analyze data from multiple S3 files with session management and multi-sheet Excel support."""
    event_logger = create_event_logger()
    authorization = request.headers.get('Authorization', "")

    try:
        from src.utils.heartbeat import heartbeat_client
        req_id = request.headers.get("x-request-id") or request.headers.get("x-correlation-id") or str(id(request))
        await heartbeat_client.send_execution_heartbeat(api_path=request.url.path, metadata={"request_id": req_id})

        event_logger.log_event("Request received. Starting your analysis...", authorization)

        metadata_fields = extract_metadata_fields(user_metadata, request)
        session_id = metadata_fields["session_id"]
        logger.info(f"Processing request for session: {session_id}, user: {metadata_fields['user_id']}")

        # 1. Data Loading Phase
        s3_url_list, all_sheets_data, all_sheets_metadata, llm_params, is_new_upload = await _perform_analysis_loading(
            s3_urls, session_id, metadata_fields['user_id'], user_metadata, event_logger, authorization
        )

        event_logger.log_event("Analyzing your data...", authorization)
        mode, relevant_sources = classify_intent_and_identify_sheets(question, all_sheets_metadata, llm_params, authorization)

        # 2. Filtering Phase
        relevant_sheets_data = _get_relevant_sheets_for_analysis(mode, relevant_sources, all_sheets_data)
        if not relevant_sheets_data:
            return _build_no_relevant_sheets_json(session_id, is_new_upload, all_sheets_data, question, event_logger, authorization)

        # 3. Execution Phase
        event_logger.log_event("Understanding your question...", authorization)
        answer, plot_data = _execute_intent_mode(mode, question, relevant_sheets_data, llm_params, authorization, event_logger)

        # 4. Finalization Phase
        event_logger.log_event("Analysis complete...", authorization)
        follow_up_questions = await _generate_follow_up_questions_async(
            question, answer, mode, relevant_sheets_data, llm_params
        )

        return _build_analysis_final_response(session_id, is_new_upload, s3_url_list, all_sheets_data, question, answer, follow_up_questions, plot_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multi-sheet analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing multi-sheet data: {str(e)}")


async def _perform_analysis_loading(s3_urls, session_id, user_id, user_metadata, event_logger, auth):
    """Encapsulates the branch logic for loading sheets either from a new upload or an existing session."""
    s3_url_list = _parse_s3_urls(s3_urls) if s3_urls and s3_urls.strip() else []
    is_new_upload = bool(s3_url_list)

    if is_new_upload:
        _validate_s3_urls(s3_url_list, event_logger, auth)
        existing_session = await get_session_data(session_id)
        urls_unchanged = (
            existing_session is not None
            and set(existing_session.get('s3_urls', [])) == set(s3_url_list)
        )
        all_sheets_data, all_sheets_metadata, llm_params = await _load_sheets_for_new_upload(
            s3_url_list, session_id, user_id, user_metadata, event_logger, auth,
            skip_db_upsert=urls_unchanged,
        )
    else:
        s3_url_list, all_sheets_data, all_sheets_metadata, llm_params = await _load_sheets_from_session(
            session_id, user_metadata, event_logger, auth
        )
    return s3_url_list, all_sheets_data, all_sheets_metadata, llm_params, is_new_upload


def _get_relevant_sheets_for_analysis(mode, relevant_sources, all_sheets_data):
    """Determines the list of relevant sheets, with fallback for 'goal' mode."""
    if not relevant_sources:
        return all_sheets_data if mode == 'goal' else []
    return _filter_relevant_sheets(relevant_sources, all_sheets_data)


def _build_no_relevant_sheets_json(session_id, is_new_upload, all_sheets_data, question, event_logger, auth):
    """Builds a JSON response for when no relevant sheets are found."""
    event_logger.log_event("Analysis complete...", auth)
    return JSONResponse({
        "status": "success",
        "session_id": session_id,
        "is_new_upload": is_new_upload,
        "message": "No relevant sheets found for the given question",
        "total_sheets_analyzed": len(all_sheets_data),
        "available_sources": [
            f"{s['file_name']}" + (f" > {s['sheet_name']}" if s['sheet_name'] else "")
            for s in all_sheets_data
        ],
        "relevant_sources": [],
        "analysis": {
            "question": question,
            "answer": "I couldn't find any sheets in the provided files that contain information relevant to your question. Please check if the correct files were uploaded or rephrase your question."
        }
    })


def _execute_intent_mode(mode, question, relevant_sheets_data, llm_params, auth, event_logger):
    """Routes the analysis to the correct handler based on the classified intent mode."""
    plot_data = None
    if mode == "plot":
        answer, plot_data = _handle_plot_mode(question, relevant_sheets_data, llm_params, auth, event_logger)
    elif mode == "insights":
        event_logger.log_event("Generating insights for your data...", auth)
        answer = generate_insights(question=question, relevant_sheets_data=relevant_sheets_data,
                                   llm_params=llm_params, authorization=auth)
    elif mode == "goal":
        event_logger.log_event("Building your goal analysis...", auth)
        answer = generate_goal_oriented_analysis(goal=question, relevant_sheets_data=relevant_sheets_data,
                                                 llm_params=llm_params, authorization=auth)
    else:
        answer = _handle_qna_mode(question, relevant_sheets_data, llm_params, auth, event_logger)
    return answer, plot_data


async def _generate_follow_up_questions_async(question, answer, mode, relevant_sheets_data, llm_params):
    """Wraps follow-up question generation with context building."""
    return await generate_follow_up_questions(
        user_query=question,
        response=answer,
        agent_capabilities=ANALYTICS_AGENT_CAPABILITIES,
        context=_build_analytics_follow_up_context(mode, relevant_sheets_data),
        llm_config={"model": llm_params["model"], **{k: v for k, v in llm_params.items() if k != "model"}},
        num_questions=3,
    )


def _build_analytics_follow_up_context(mode: str, relevant_sheets_data: List[Dict]) -> Dict:
    """Builds the context object for follow-up question generation."""
    return {
        "mode": mode,
        "sources": [
            {
                "file": s["file_name"],
                "sheet": s["sheet_name"],
                "columns": [
                    {
                        "name": col["name"],
                        "dtype": col["dtype"],
                        "cardinality_tier": (
                            "low" if col.get("unique_count", 0) <= 8
                            else "medium" if col.get("unique_count", 0) <= 15
                            else "high" if col.get("unique_count", 0) <= 30
                            else "extreme"
                        ) if col.get("dtype") == "object" else "numeric"
                    }
                    for col in s["metadata"]["columns"]
                ]
            }
            for s in relevant_sheets_data
        ]
    }


def _build_analysis_final_response(session_id, is_new_upload, s3_url_list, all_sheets_data, question, answer, follow_ups, plot_data):
    """Constructs the standard final successful analysis response."""
    resp = {
        "status": "success",
        "session_id": session_id,
        "is_new_upload": is_new_upload,
        "files_info": {
            "total_files_provided": len(s3_url_list),
            "total_sheets_processed": len(all_sheets_data)
        },
        "analysis": {"question": question, "answer": answer},
        "follow_up_questions": follow_ups,
    }
    if plot_data:
        resp["analysis"]["plot_data"] = plot_data
    return JSONResponse(resp)


@router.post(
    "/analyze",
    responses={
        400: {"description": "Invalid S3 URL, no valid sheets, or missing session"},
        500: {"description": "Internal server error during analysis"}
    }
)
async def analyze_data_from_s3(
    request: Request,
    s3_url: Annotated[Optional[str], Form()] = None,
    question: Annotated[str, Form()] = ...,
    user_metadata: Annotated[str, Form()] = ...,
    sheet_name: Annotated[Optional[str], Form()] = None,
):
    """Single-file endpoint - redirects to multi-sheet handler."""
    return await analyze_data_from_multiple_s3_files(
        request=request,
        s3_urls=s3_url if s3_url else "",
        question=question,
        user_metadata=user_metadata
    )


def _parse_s3_urls(s3_urls: str) -> List[str]:
    """Parse S3 URLs from JSON array string or comma-separated string."""
    url_list = []
    try:
        if s3_urls.strip().startswith('['):
            parsed = json.loads(s3_urls)
            if isinstance(parsed, list):
                url_list = [u.strip() for u in parsed if u and u.strip()]
    except (json.JSONDecodeError, AttributeError):
        pass
    if not url_list:
        url_list = [u.strip() for u in s3_urls.split(',') if u.strip()]
    return url_list


async def _resolve_s3_url_list(
    s3_urls: Optional[str],
    session_id: str,
    event_logger,
    authorization: str
) -> List[str]:
    """Return parsed URL list from input or fall back to session."""
    if s3_urls and s3_urls.strip():
        url_list = _parse_s3_urls(s3_urls)
        logger.info(f"Using provided S3 URLs: {len(url_list)} files")
        return url_list

    logger.info(f"No S3 URLs provided, checking session {session_id}...")
    session_data = await get_session_data(session_id)
    if session_data and session_data['s3_urls']:
        logger.info(f"Using S3 URLs from session: {len(session_data['s3_urls'])} files")
        return session_data['s3_urls']

    event_logger.log_event(MSG_INVALID_FILE_LINK, authorization)
    raise HTTPException(
        status_code=400,
        detail=f"No S3 URLs provided and no session found with session_id: {session_id}. Please provide s3_urls or upload files first."
    )


def _validate_s3_urls(s3_url_list: List[str], event_logger, authorization: str) -> None:
    """Raise 400 if any URL is invalid."""
    for s3_url in s3_url_list:
        if not validate_s3_url(s3_url):
            event_logger.log_event(MSG_INVALID_FILE_LINK, authorization)
            raise HTTPException(
                status_code=400,
                detail=f"Invalid S3 URL format: {s3_url}. Must start with https:// or s3://"
            )


def _load_sheets_from_url(s3_url: str) -> tuple[List[Dict], List[Dict]]:
    """Load a single S3 file into cleaned sheet dicts. Frees raw bytes immediately."""
    sheets_data: List[Dict] = []
    metadata_list: List[Dict] = []

    content = s3_utility.get_data_from_s3_by_url(s3_url)
    file_name = s3_utility.extract_filename_from_s3_url(s3_url)

    if not content:
        logger.warning("Empty or missing file: %s", s3_url)
        return sheets_data, metadata_list

    file_data = process_file_with_sheets(content, file_name)
    del content
    gc.collect()

    pivot_contexts = file_data.get("pivot_contexts", {})
    sheet_names = file_data["sheet_names"]
    logger.info("File: %s (%s) | Sheets: %s", file_name, file_data["file_type"], sheet_names)

    for sheet_name in sheet_names:
        df = file_data["sheets"].pop(sheet_name, None)
        if df is None or df.empty:
            logger.warning("Empty sheet '%s' — skipping.", sheet_name)
            continue

        pivot_context = pivot_contexts.get(sheet_name)
        cleaned_df = comprehensive_dataframe_cleaning(df, file_name, sheet_name)
        del df
        gc.collect()

        metadata = extract_file_metadata(cleaned_df, file_name, sheet_name, pivot_context)
        sheets_data.append({
            "file_name": file_name,
            "sheet_name": sheet_name,
            "df": cleaned_df,
            "s3_url": s3_url,
            "pivot_context": pivot_context,
        })
        metadata_list.append(metadata)
        logger.info("  Sheet: %s | %d rows × %d cols%s",
                    sheet_name, len(cleaned_df), len(cleaned_df.columns),
                    " [PIVOT]" if pivot_context else "")

    del file_data
    gc.collect()
    return sheets_data, metadata_list


def _load_all_sheets(s3_url_list: List[str]) -> tuple[List[Dict], List[Dict]]:
    """Load and clean sheets from all S3 URLs."""
    all_sheets = []
    all_metadata = []
    for s3_url in s3_url_list:
        try:
            sheets, meta = _load_sheets_from_url(s3_url)
            all_sheets.extend(sheets)
            all_metadata.extend(meta)
        except Exception as e:
            logger.error(f"Error processing {s3_url}: {e}", exc_info=True)
    return all_sheets, all_metadata


def _get_base_filename(s3_url_list: List[str]) -> str:
    """Extract and clean base filename from first S3 URL."""
    base = s3_url_list[0].split('/')[-1].rsplit('.', 1)[0] if s3_url_list else "multi_file"
    logger.info(f"Base filename for report: {base}")
    base = re.sub(r'_[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$', '', base)
    logger.info(f"Cleaned base filename for report: {base}")
    return base


@router.post(
    "/generate-report",
    responses={
        400: {"description": "Invalid S3 URL, no valid sheets, or missing session"},
        500: {"description": "Internal server error during report generation"}
    }
)
@track_llm_calls(
    name="backend-generate-report",
    tags=["report-generation", "session-management", "multi-file-support"],
    metadata={"version": "1.0"},
    avoided_input_params=["request"]
)
async def generate_eda_report(
    request: Request,
    s3_urls: Annotated[Optional[str], Form()] = None,
    user_metadata: Annotated[str, Form()] = ...,
    sheet_name: Annotated[Optional[str], Form()] = None,
):
    """Generate comprehensive EDA report with session management and multi-file support."""
    event_logger = create_event_logger()
    authorization = ""

    try:
        authorization = request.headers.get('Authorization', "")

        from src.utils.heartbeat import heartbeat_client
        request_id = request.headers.get("x-request-id") or request.headers.get("x-correlation-id") or str(id(request))
        await heartbeat_client.send_execution_heartbeat(
            api_path=request.url.path,
            metadata={"request_id": request_id}
        )

        event_logger.log_event("Request received. Starting report generation...", authorization)

        metadata_fields = extract_metadata_fields(user_metadata, request)
        session_id = metadata_fields["session_id"]
        user_id = metadata_fields["user_id"]
        logger.info(f"Generating report for session: {session_id}, user: {user_id}")

        s3_url_list = await _resolve_s3_url_list(s3_urls, session_id, event_logger, authorization)
        _validate_s3_urls(s3_url_list, event_logger, authorization)

        logger.info(f"Generating EDA report for {len(s3_url_list)} file(s)")
        event_logger.log_event(MSG_CONFIGURING_ENGINE, authorization)
        llm_params = await get_llm_config(user_metadata)

        event_logger.log_event(MSG_LOADING_DATA, authorization)
        all_sheets_to_analyze, all_sheets_metadata = _load_all_sheets(s3_url_list)

        if not all_sheets_to_analyze:
            event_logger.log_event("Couldn't process the uploaded files...", authorization)
            raise HTTPException(status_code=400, detail="No valid sheets could be processed from the provided S3 URLs")

        file_info = {
            "total_files": len(s3_url_list),
            "total_sheets": len(all_sheets_to_analyze),
            "files": [
                {"file_name": s['file_name'], "sheet_name": s['sheet_name'],
                 "rows": len(s['df']), "columns": len(s['df'].columns)}
                for s in all_sheets_to_analyze
            ]
        }

        await create_or_update_session(
            session_id=session_id, user_id=user_id,
            s3_urls=s3_url_list, all_sheets_metadata=all_sheets_metadata, file_info=file_info
        )
        logger.info(f"✓ Session stored/updated in database: {session_id}")

        event_logger.log_event("Analyzing your data...", authorization)
        logger.info(f"Starting EDA report generation for {len(all_sheets_to_analyze)} sheet(s) across {len(s3_url_list)} file(s)...")

        base_filename = _get_base_filename(s3_url_list)

        pdf_bytes, report_filename = await report_generator.generate_multi_sheet_report(
            sheets_data=all_sheets_to_analyze,
            file_name=base_filename if len(s3_url_list) == 1 else "Multi-File Analysis",
            llm_params=llm_params,
            token_tracker=token_tracker,
            auth_token=authorization
        )
        logger.info(f"Report generated: {report_filename}")

        event_logger.log_event("Uploading your results...", authorization)
        report_s3_url = s3_utility.upload_file(file_content=pdf_bytes, file_name=report_filename, folder="reports")
        report_presigned_url = s3_utility.generate_presigned_url(report_s3_url)

        event_logger.log_event("Analysis complete.", authorization)

        return JSONResponse({
            "status": "success",
            "message": "EDA report generated successfully",
            "session_id": session_id,
            "filename": report_filename,
            "report_url": report_presigned_url,
            "files_analyzed": len(s3_url_list),
            "sheets_analyzed": len(all_sheets_to_analyze)
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report generation error: {e}", exc_info=True)
        event_logger.log_event("We hit a system issue. Please try again.", authorization)
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")