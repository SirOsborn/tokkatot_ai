# 🎨 Frontend - User Interface

**Optional UI for web-based streaming and analysis**

## ⚠️ Status

- **Status**: Optional prototype (not required for deployment)
- **Deployed**: ✅ Optional (can skip for Pi)
- **Tech**: Streamlit or Flask-based
- **Features**: Live streaming, photo upload, results display

---

## 📂 Structure

```
frontend/
├── streaming/          # Real-time video UI
│   ├── app.py          # Main streaming app
│   └── components.py   # UI components
│
├── upload/             # Photo analysis UI
│   ├── uploader.py     # Upload handler
│   ├── preview.py      # Image preview
│   └── results.py      # Results display
│
└── components/         # Shared UI components
    ├── navbar.py       # Navigation
    ├── dashboard.py    # Main dashboard
    └── metrics_panel.py # Metrics display
```

---

## 🎯 Streaming UI

**Real-time disease detection display**

### Features
- ✅ Live video stream from camera
- ✅ Detection boxes overlay
- ✅ FPS counter
- ✅ Detection statistics
- ✅ Anomaly rate visualization
- ✅ Alert notifications
- ✅ Recording capability

### Implementation (Streamlit Example)

```python
# frontend/streaming/app.py

import streamlit as st
from backend.core import EdgeApp

st.set_page_config(page_title="Tokkatot Streaming", layout="wide")

app = EdgeApp()

col1, col2 = st.columns([3, 1])

with col1:
    st.header("Live Detection Stream")
    video_placeholder = st.empty()
    
    # Stream frames
    for frame, stats in app.process_stream():
        video_placeholder.image(frame, use_column_width=True)

with col2:
    st.header("Statistics")
    st.metric("FPS", stats['fps'])
    st.metric("Detections", stats['total_detections'])
    st.metric("Disease %", f"{stats['anomaly_rate']:.1f}%")
    
    if stats['should_alert']:
        st.error("⚠️ ALERT: High anomaly rate!")
```

### UI Layout
```
┌────────────────────────────────┬──────────────┐
│                                │              │
│  Video Feed                    │ Metrics      │
│  (Detection Boxes Overlay)     │ • FPS: 22.3  │
│                                │ • Count: 457 │
│                                │ • Disease: 5%│
│                                │              │
│                                │ [ALERT] warn │
│                                │              │
└────────────────────────────────┴──────────────┘
```

---

## 📤 Upload UI

**Analyze individual fecal samples**

### Features
- ✅ Image upload (drag & drop)
- ✅ Image preview
- ✅ Instant detection
- ✅ Results display
- ✅ Report download

### Implementation (Streamlit Example)

```python
# frontend/upload/app.py

import streamlit as st
from backend.services.inference import ChickenDiseaseDetector

st.header("Upload Fecal Sample")

uploaded_file = st.file_uploader("Choose image...", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original")
        st.image(image)
    
    with col2:
        st.subheader("Results")
        
        detector = ChickenDiseaseDetector()
        disease, confidence = detector.predict(image)
        
        st.write(f"**Disease**: {disease}")
        st.write(f"**Confidence**: {confidence:.1%}")
        
        # Color-code results
        if disease == "Healthy":
            st.success("✅ Sample is healthy")
        else:
            st.warning(f"⚠️ {disease} detected")
```

### UI Layout
```
┌────────────────────────────────────────┐
│ Upload Fecal Sample                    │
│                                        │
│ [Upload Image Box]                     │
│                                        │
├──────────────┬───────────────────────┤
│  Original    │  Results              │
│              │  Disease: Coccidiosis │
│ [Image]      │  Confidence: 94%      │
│              │  [✓ Detected]         │
└──────────────┴───────────────────────┘
```

---

## 📊 Components - Shared

### Dashboard (`components/dashboard.py`)
Main UI layout:
- Navigation bar
- Active alerts banner
- Metrics summary
- Time-series graphs

### Metrics Panel (`components/metrics_panel.py`)
Real-time statistics:
- FPS counter
- Detection count (all-time)
- Disease rate (%)
- System resources (CPU, memory)

### Navbar (`components/navbar.py`)
Navigation:
- Home
- Streaming
- Upload
- History
- Settings

---

## 🔌 Connecting to Backend

### API Calls

```python
# frontend/upload/uploader.py

import requests

def analyze_image(image_file):
    """Send image to backend API for analysis"""
    
    response = requests.post(
        "http://localhost:5000/api/v1/detect",
        files={"image": image_file}
    )
    
    result = response.json()
    return {
        "disease": result["disease"],
        "confidence": result["confidence"],
        "boxes": result["boxes"]
    }
```

### WebSocket for Live Streaming

```python
# frontend/streaming/app.py

import websocket
import json

def stream_live_metrics():
    """Connect to backend websocket for live updates"""
    
    ws = websocket.WebSocketApp(
        "ws://localhost:5000/metrics/stream",
        on_message=on_message,
        on_error=on_error
    )
    
    def on_message(ws, message):
        metrics = json.loads(message)
        yield metrics
    
    ws.run()
```

---

## 🚀 Running Frontend

### Option 1: Streamlit (Simplest)

```bash
# Install
pip install streamlit

# Run
streamlit run frontend/streaming/app.py

# Opens: http://localhost:8501
```

### Option 2: Flask (More Control)

```bash
# Install
pip install flask flask-cors

# Run
python frontend/app.py

# Opens: http://localhost:5000
```

### Option 3: Docker

```bash
docker run -p 8501:8501 tokkatot-frontend
```

---

## 📱 Multi-Device Support

### Desktop
- Full dashboard with graphs
- Real-time video stream
- Detailed analytics

### Mobile (Responsive)
- Simplified layout
- Touch-friendly interface
- Essential metrics only

### Tablet
- Balanced view
- Vertical and horizontal orientation
- Dashboard widgets

---

## 🎨 Design System

### Color Scheme
```
✅ Success (Green): #10B981      - Healthy samples
⚠️  Warning (Yellow): #F59E0B    - Disease detected
❌ Error (Red): #EF4444          - High anomaly rate
ℹ️  Info (Blue): #3B82F6         - System status
```

### Typography
- **Heading**: Arial, 24px, bold
- **Body**: Arial, 14px, regular
- **Mono**: Courier, 12px, for logs

### Layout
- Max width: 1200px
- Padding: 20px
- Grid: 12 columns

---

## 📊 Example Dashboards

### Real-time Dashboard
```
╔════════════════════════════════════════════╗
║ Tokkatot - Real-time Monitoring            ║ 
╠════════════════════════════════════════════╣
║                                             ║
║ 📊 Streaming                                ║
║ [Live Video Feed with Boxes]    [Metrics]  ║
║                                             ║
║ Status: ✅ OK  FPS: 22.3  Latency: 45ms   ║
║ Detections: 457  Disease Rate: 5.04%      ║
║                                             ║
╚════════════════════════════════════════════╝
```

### Upload Analysis Dashboard
```
╔════════════════════════════════════════════╗
║ Tokkatot - Single Sample Analysis           ║
╠════════════════════════════════════════════╣
║                                             ║
║ Upload Sample: [Drag & Drop Zone]          ║
║                                             ║
║ ┌──────────────────┬──────────────────┐   ║
║ │ Original         │ Analysis Results │   ║
║ │                  │                  │   ║
║ │ [Image Preview]  │ Disease: Healthy │   ║
║ │                  │ Confidence: 98% │   ║
║ │                  │ ✅ Approved      │   ║
║ └──────────────────┴──────────────────┘   ║
║                                             ║
║ [Download Report] [Save]                   ║
║                                             ║
╚════════════════════════════════════════════╝
```

---

## 🔐 Security

- ✅ HTTPS only (use SSL certificates)
- ✅ API authentication (API keys or OAuth)
- ✅ Rate limiting (prevent abuse)
- ✅ Input validation (sanitize uploads)
- ✅ CORS properly configured

---

## 📈 Performance

- Target load time: < 2 seconds
- Framework: Streamlit (faster prototype) or Flask (more control)
- Image optimization: Compress before displaying
- Caching: Cache model inference results

---

## 🛠️ Development

### Add New Page

```python
# frontend/pages/history.py

import streamlit as st

def show_history():
    st.header("Detection History")
    
    # Fetch history from API
    history = requests.get("http://localhost:5000/metrics/history").json()
    
    # Display table
    st.dataframe(history)
    
    # Display graph
    st.line_chart(history['anomaly_rate'])

show_history()
```

### Add New Component

```python
# frontend/components/alert_banner.py

import streamlit as st

def alert_banner(alert_type, message):
    if alert_type == "error":
        st.error(f"❌ {message}")
    elif alert_type == "warning":
        st.warning(f"⚠️ {message}")
    else:
        st.info(f"ℹ️ {message}")
```

---

## 📦 Dependencies

```
streamlit>=1.0
pillow>=9.0
numpy>=1.20
opencv-python>=4.5
requests>=2.28
```

Or for Flask:
```
flask>=2.0
flask-cors>=3.0
pillow>=9.0
numpy>=1.20
opencv-python>=4.5
requests>=2.28
```

---

## ⚠️ Optional?

Frontend is **NOT REQUIRED** for production:

- ✅ Raspberry Pi runs without frontend (just CLI)
- ✅ Metrics sent to cloud dashboard instead
- ✅ Use Tokkatot web app for central dashboard

Frontend is useful for:
- 🔍 Local testing and debugging
- 📊 Real-time visualization on monitoring station
- 📤 Local photo upload for quick analysis

---

## 🔗 Next Steps

1. **Choose framework**: Streamlit (easy) or Flask (flexible)
2. **Implement streaming**: Connect to backend/core
3. **Implement upload**: Connect to backend/api
4. **Add analytics**: Graphs and statistics
5. **Style**: Apply design system
6. **Deploy**: Docker or direct hosting

---

**Frontend is your local dashboard.** ✨
