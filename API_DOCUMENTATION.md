API_DOCS = """
# API Documentation

## Overview

This API provides intelligent data analytics capabilities with support for:
- File upload and processing
- Multiple search modes (Textual/Semantic)
- AI-powered query answering
- Automatic visualization generation
- Statistical analysis

## Base URL

```
http://localhost:8000
```

## Authentication

Currently no authentication required. For production, implement API key authentication.

## Endpoints

### 1. Upload File

**Endpoint:** `POST /upload`

**Description:** Upload an Excel file for processing and analysis.

**Request:**
- Content-Type: `multipart/form-data`

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| file | File | Yes | Excel file (.xlsx, .xls) |
| client_name | String | Yes | Client identifier |
| search_type | String | Yes | "Textual" or "Semantic" |
| batch_size | Integer | No | Embedding batch size (default: 100) |

**Response:**
```json
{
  "status": "success",
  "session_id": "abc123def456",
  "message": "File processed successfully with Semantic search",
  "data_size": 1000,
  "columns": ["column1", "column2", "column3"]
}
```

**Status Codes:**
- 200: Success
- 400: Invalid file format or parameters
- 500: Server error

**Example:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@sales_data.xlsx" \
  -F "client_name=AcmeCorp" \
  -F "search_type=Semantic" \
  -F "batch_size=128"
```

---

### 2. Query Data

**Endpoint:** `POST /query`

**Description:** Query the uploaded data using natural language.

**Request:**
- Content-Type: `application/json`

**Body:**
```json
{
  "session_id": "abc123def456",
  "question": "Show me a bar chart of sales by region"
}
```

**Response:**
```json
{
  "answer": "I've created the visualization based on your request.",
  "mode": "plot",
  "plot_data": {
    "figure": "{...plotly json...}",
    "type": "plotly"
  }
}
```

**Query Modes:**
- `plot`: Generates visualizations
- `qna`: Answers questions using context
- `stats`: Provides statistical analysis

**Status Codes:**
- 200: Success
- 404: Session not found
- 500: Server error

**Example:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "abc123def456",
    "question": "What is the average sales value?"
  }'
```

---

### 3. Get Raw Data

**Endpoint:** `GET /data/{session_id}`

**Description:** Retrieve the raw DataFrame for a session.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| session_id | String | Yes | Session identifier from upload |

**Response:**
```json
{
  "columns": ["column1", "column2"],
  "data": [
    {"column1": "value1", "column2": "value2"},
    {"column1": "value3", "column2": "value4"}
  ],
  "shape": [1000, 10]
}
```

**Status Codes:**
- 200: Success
- 404: Session not found

**Example:**
```bash
curl -X GET "http://localhost:8000/data/abc123def456"
```

---

### 4. Delete Session

**Endpoint:** `DELETE /session/{session_id}`

**Description:** Delete a session and clean up resources.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| session_id | String | Yes | Session identifier to delete |

**Response:**
```json
{
  "status": "success",
  "message": "Session deleted"
}
```

**Status Codes:**
- 200: Success
- 404: Session not found

**Example:**
```bash
curl -X DELETE "http://localhost:8000/session/abc123def456"
```

---

### 5. Health Check

**Endpoint:** `GET /health`

**Description:** Check API health status.

**Response:**
```json
{
  "status": "healthy"
}
```

**Status Codes:**
- 200: Service is healthy

**Example:**
```bash
curl -X GET "http://localhost:8000/health"
```

---

## Usage Flow

1. **Upload File:**
   - Upload Excel file with `/upload`
   - Receive `session_id`

2. **Query Data:**
   - Use `session_id` to query with `/query`
   - Ask natural language questions
   - Get answers, plots, or statistics

3. **Retrieve Data (Optional):**
   - Get raw data with `/data/{session_id}`

4. **Cleanup:**
   - Delete session with `/session/{session_id}`

## Error Handling

All errors follow this format:
```json
{
  "detail": "Error message description"
}
```

Common error codes:
- 400: Bad Request (invalid parameters)
- 404: Not Found (invalid session_id)
- 500: Internal Server Error

## Rate Limiting

Currently no rate limiting. Implement for production use.

## Best Practices

1. **Session Management:**
   - Store `session_id` securely
   - Delete sessions when done
   - Sessions persist in memory (cleared on restart)

2. **File Upload:**
   - Use clean Excel files
   - Remove unnecessary sheets
   - Keep file size reasonable (<50MB)

3. **Queries:**
   - Be specific in questions
   - Use clear visualization requests
   - Combine related questions

4. **Search Type Selection:**
   - Use "Textual" for keyword matching
   - Use "Semantic" for meaning-based search
   - Semantic search requires more processing time

## Example Integration

### Python Client
```python
import requests

# Upload file
with open('data.xlsx', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/upload',
        files={'file': f},
        data={
            'client_name': 'MyApp',
            'search_type': 'Semantic'
        }
    )
session_id = response.json()['session_id']

# Query data
response = requests.post(
    'http://localhost:8000/query',
    json={
        'session_id': session_id,
        'question': 'What are the top 5 products by sales?'
    }
)
print(response.json()['answer'])

# Cleanup
requests.delete(f'http://localhost:8000/session/{session_id}')
```

### JavaScript Client
```javascript
// Upload file
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('client_name', 'MyApp');
formData.append('search_type', 'Semantic');

const uploadRes = await fetch('http://localhost:8000/upload', {
  method: 'POST',
  body: formData
});
const {session_id} = await uploadRes.json();

// Query data
const queryRes = await fetch('http://localhost:8000/query', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    session_id,
    question: 'Show sales trends over time'
  })
});
const result = await queryRes.json();
console.log(result.answer);
```
"""