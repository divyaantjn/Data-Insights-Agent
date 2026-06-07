# src/utils/event_messages.py

"""
Centralized event message constants for consistent messaging across the Excel Analyzer Backend.
"""


class EventMessages:
    """Standardized event message constants."""

    # Task events
    TASK_RECEIVED = "Data analysis task received..."

    # Progress events
    PROGRESS_CONFIGURING_LLM = "Initializing analysis engine..."
    PROGRESS_DOWNLOADING_FILE = "Downloading file from source..."
    PROGRESS_PROCESSING_DATA = "Processing data file..."
    PROGRESS_ANALYZING_SCHEMA = "Analyzing data schema..."
    PROGRESS_CLASSIFYING_QUESTION = "Understanding your question..."
    PROGRESS_GENERATING_PLOT = "Generating visualization..."
    PROGRESS_EXECUTING_ANALYSIS = "Executing data analysis..."
    PROGRESS_FORMATTING_ANSWER = "Formatting answer..."
    PROGRESS_UPLOADING = "Uploading results to storage..."

    # Success events
    SUCCESS_ANALYSIS_COMPLETE = "Analysis completed successfully"
    SUCCESS_PLOT_GENERATED = "Visualization generated successfully"
    SUCCESS_ANSWER_GENERATED = "Answer generated successfully"

    # Error events
    ERROR_INVALID_REQUEST = "Invalid request parameters"
    ERROR_S3_DOWNLOAD_FAILED = "Failed to download file from source"
    ERROR_FILE_PROCESSING_FAILED = "Failed to process data file"
    ERROR_ANALYSIS_FAILED = "Data analysis failed"
    ERROR_PLOT_GENERATION_FAILED = "Visualization generation failed"
    ERROR_SYSTEM_ERROR = "System error occurred"
