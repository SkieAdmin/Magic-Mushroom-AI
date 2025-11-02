# Vegetable Freshness Scanner Backend

FastAPI + YOLOv8 backend for real-time vegetable freshness detection.

---

## Features

- **Real-time Detection** via WebSocket streaming
- **Multi-object Detection** with YOLOv8
- **Freshness Classification** (Healthy / Damaged / Rotten)
- **Bounding Box Coordinates** for camera overlay
- **GPU Detection** and hardware acceleration
- **Database History** with SQLite
- **REST API** for image upload
- **Frontend Compatible** with React Native app

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)

Create `.env` file:

```env
MODEL_PATH=yolov8n.pt
HOST=0.0.0.0
PORT=9055
DATABASE_URL=sqlite:///./vegetable_detections.db
```

### 3. Run the Server

```bash
python main.py
```

Server will start on: `http://0.0.0.0:9055`

### 4. Test the Backend

```bash
python test_backend.py
```

---

## API Endpoints

### 🔌 WebSocket: `/ws/detect`

**Real-time detection streaming**

**Client Sends:**
```json
{
  "type": "frame",
  "image": "<base64-encoded-jpeg>",
  "timestamp": 1699999999999
}
```

**Server Responds:**
```json
{
  "detections": [
    {
      "class": "rotten_carrot",
      "confidence": 0.87,
      "bbox": [100.5, 150.2, 300.8, 400.3],
      "timestamp": 1699999999999
    }
  ],
  "inferenceTime": 45.2,
  "fps": 12.5,
  "gpuName": "NVIDIA RTX 3080"
}
```

---

### 📤 POST `/upload`

**Upload image for detection**

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response:**
```json
{
  "detections": [
    {
      "class": "healthy_tomato",
      "confidence": 0.92,
      "bbox": [50, 100, 200, 300],
      "timestamp": 1699999999999
    }
  ],
  "inferenceTime": 38.5,
  "fps": 0,
  "gpuName": "CPU"
}
```

---

### 📊 GET `/history`

**Get detection history**

**Query Parameters:**
- `limit` (optional, default: 20)

**Response:**
```json
{
  "detections": [
    {
      "id": 1,
      "timestamp": "2025-11-02T12:00:00",
      "vegetable": "carrot",
      "confidence": 87.5,
      "freshness": 75.2,
      "status": "Good",
      "recommendation": "Recommended for consumption."
    }
  ],
  "total": 1
}
```

---

### 🏥 GET `/ping`

**Health check**

**Response:**
```json
{
  "status": "online",
  "latency_ms": 0.15
}
```

---

## Vegetable Classification

The backend automatically classifies vegetables into three categories:

### 🟢 Healthy (Good)

- **Prefix:** `healthy_*` or `fresh_*`
- **Status:** `good`
- **Freshness:** 70-100%
- **Recommendation:** "Recommended for consumption"

**Examples:**
- `healthy_carrot`
- `fresh_tomato`
- `healthy_potato`

### 🟡 Damaged (Caution)

- **Prefix:** `damaged_*`, `old_*`, or `aging_*`
- **Status:** `caution`
- **Freshness:** 40-75%
- **Recommendation:** "Caution advised. Cook thoroughly or use soon."

**Examples:**
- `damaged_carrot`
- `old_tomato`
- `aging_potato`

### 🔴 Rotten (Bad)

- **Prefix:** `rotten_*` or `bad_*`
- **Status:** `bad`
- **Freshness:** 0-40%
- **Recommendation:** "Not recommended for consumption"

**Examples:**
- `rotten_carrot`
- `bad_tomato`
- `rotten_potato`

---

## Project Structure

```
backend/
├── main.py                 # FastAPI application
├── database.py             # SQLAlchemy database setup
├── models.py               # Database models
├── schemas.py              # Pydantic schemas (API contracts)
├── requirements.txt        # Python dependencies
├── test_backend.py         # Test suite
├── utils/
│   ├── __init__.py
│   └── freshness.py        # Freshness classification logic
└── README.md               # This file
```

---

## Frontend Compatibility

This backend is **100% compatible** with the React Native frontend.

### Key Features:

✅ Returns `detections` array (not single object)
✅ Includes `bbox` coordinates for camera overlay
✅ Uses `0-1` confidence scale (not 0-100)
✅ Key names match: `class`, `inferenceTime`, `fps`, `gpuName`
✅ WebSocket accepts `{"type": "frame", "image": "..."}` format
✅ Multi-object detection support
✅ GPU information included

---

## YOLOv8 Model

### Default Model

The backend uses `yolov8n.pt` (nano model) by default for fast inference.

### Using Custom Model

To use a custom trained model:

1. Place your `.pt` file in the backend directory
2. Update `.env`:
   ```env
   MODEL_PATH=vegetable_freshness_model.pt
   ```

### Model Requirements

Your YOLO model should detect classes like:
- `carrot`, `tomato`, `potato`, etc.
- Or: `healthy_carrot`, `damaged_carrot`, `rotten_carrot`, etc.

The backend automatically classifies based on class name prefixes.

---

## Database

### Schema

```sql
CREATE TABLE detections (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME DEFAULT NOW(),
    vegetable VARCHAR,
    confidence FLOAT,
    freshness FLOAT,
    status VARCHAR,
    recommendation VARCHAR
);
```

### Location

Default: `vegetable_detections.db` in the backend directory

---

## GPU Support

### Automatic Detection

The backend automatically detects and uses available GPU:

```python
# GPU detected
"gpuName": "NVIDIA RTX 3080"

# No GPU
"gpuName": "CPU"
```

### Enable GPU for YOLO

Install PyTorch with CUDA:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Testing

### Run Test Suite

```bash
python test_backend.py
```

### Tests Included

1. ✅ **Ping Test** - Health check endpoint
2. ✅ **Root Test** - API info endpoint
3. ✅ **History Test** - Database query
4. ✅ **Upload Test** - Image upload and detection
5. ✅ **WebSocket Test** - Real-time streaming

### Expected Output

```
🧪 BACKEND COMPATIBILITY TEST SUITE
============================================================
TEST 1: Ping Endpoint
✅ PASS: /ping endpoint works correctly

TEST 2: Root Endpoint
✅ PASS: Root endpoint works

TEST 3: History Endpoint
✅ PASS: /history endpoint works (found 0 records)

TEST 4: Upload Endpoint
✅ PASS: /upload endpoint returns correct format

TEST 5: WebSocket Endpoint
✅ PASS: WebSocket endpoint returns correct format

============================================================
🎉 ALL TESTS PASSED!
============================================================
```

---

## Performance

| Metric | Value |
|--------|-------|
| Inference Time | 20-50ms (GPU) / 100-300ms (CPU) |
| FPS (Streaming) | 10-30 FPS (depends on hardware) |
| WebSocket Latency | <10ms (local) / 50-200ms (network) |
| Max Concurrent Connections | 100+ |

---

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 9055
netstat -ano | findstr :9055

# Kill the process
taskkill /PID <PID> /F
```

### Model Not Loading

```
FileNotFoundError: yolov8n.pt not found
```

**Solution:** Download YOLOv8 model:
```bash
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### WebSocket Connection Failed

**Check firewall settings:**
- Allow inbound connections on port 9055
- If using remote device, use correct IP address (not localhost)

### Import Errors

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

---

## Development

### Enable Hot Reload

The server automatically reloads on code changes when running via:

```bash
python main.py
```

### Add CORS Origins

Edit `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Production Deployment

### Using Uvicorn

```bash
uvicorn main:app --host 0.0.0.0 --port 9055 --workers 4
```

### Using Docker

```dockerfile
FROM python:3.10

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 9055
CMD ["python", "main.py"]
```

Build and run:
```bash
docker build -t veggie-scanner .
docker run -p 9055:9055 veggie-scanner
```

---

## License

MIT License - Free to use for any purpose.

---

## Credits

- **Framework:** FastAPI
- **AI Model:** YOLOv8 (Ultralytics)
- **Database:** SQLAlchemy + SQLite
- **Computer Vision:** OpenCV

---

## Support

For issues or questions:
1. Check the test suite: `python test_backend.py`
2. Review compatibility report: `BACKEND_FRONTEND_SYNC_REPORT.md`
3. Check frontend types: `../src/types/index.ts`

---

**Backend is now fully compatible with React Native frontend! 🎉**
