from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class QueryRequest(BaseModel):
    session_id: str
    question: str
    sheet_name: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    mode: str
    plot_data: Optional[Dict[str, Any]] = None

class UploadResponse(BaseModel):
    status: str
    session_id: str
    message: str
    data_size: int
    columns: List[str]
    available_sheets: Optional[List[str]] = None

class PlotResponse(BaseModel):
    status: str
    plot_data: Optional[Dict[str, Any]] = None
    message: str

class SheetListResponse(BaseModel):
    status: str
    sheets: List[str]
    message: str