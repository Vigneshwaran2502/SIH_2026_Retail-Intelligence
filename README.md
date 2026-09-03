# Retail Intelligence Platform - SIH26179

An AI-Powered Retail Analytics Dashboard designed for Smart India Hackathon (SIH). This platform provides real-time, centralized analytics for retail stores using advanced computer vision and data streaming.

## 🌟 Key Features

* **Advanced Human Behavior Detection**: Uses YOLOv8 Pose estimation to track shopper behavior, detecting actions like:
  * Picking (via hand gesture recognition)
  * Sitting
  * Standing
  * Walking
  * Falling
* **Real-time Analytics Dashboard**: A sleek, responsive frontend built with HTML/CSS/JS and WebSockets for live data streaming.
* **Shopper Analytics & Footfall Tracking**: Monitors visitor counts, conversion rates, and queue status across multiple stores.
* **Inventory Management**: Real-time tracking of stock levels, including alerts for out-of-stock items and simulated restocking.
* **Live Camera Feeds**: Integrated AI models provide live annotations for behavior detection, directly pushed to the dashboard.
* **Exportable Reports**: Generate timeline data and export it via CSV.

## 🚀 Tech Stack

* **Backend**: FastAPI (Python)
* **Real-time Communication**: WebSockets
* **AI / Computer Vision**: YOLOv8 (Ultralytics), OpenCV
* **Data Processing**: Numpy, Pandas
* **Frontend**: HTML5, Vanilla JS, CSS3

## 📁 Project Structure

```
.
├── backend/                  # FastAPI backend and sample data generators
│   ├── main.py               # Core API, WebSocket handlers, and endpoints
│   └── sample_data.py        # Logic for simulating store data
├── frontend/                 # Static web assets for the dashboard
│   ├── css/                  # Styling files
│   ├── js/                   # Dashboard logic and WebSocket connection
│   └── index.html            # Main dashboard view
├── behavior_detector.py      # Core YOLO behavior detection scripts
├── behavior_detector_v2.py   
├── behavior_detector_v3.py   # Advanced multi-scale inference + gesture tracking
├── detection_bridge.py       # Bridges local computer vision data to the API
├── run.py                    # Entry point to launch the server
├── yolov8n-pose.pt           # YOLOv8 nano pose model weights
└── requirements.txt          # Python dependencies
```

## 🛠️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Vigneshwaran2502/SIH_2026_Retail-Intelligence.git
   cd SIH_2026_Retail-Intelligence
   ```

2. **Install the dependencies:**
   Ensure you have Python 3.8+ installed, then run:
   ```bash
   pip install -r requirements.txt
   pip install ultralytics opencv-python  # Required for behavior detection
   ```

3. **Start the application:**
   Launch the FastAPI server and frontend using:
   ```bash
   python run.py
   ```

4. **Access the application:**
   * **Dashboard**: [http://localhost:8000](http://localhost:8000)
   * **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

## 📡 API Endpoints

The platform provides a comprehensive REST API to interface with the data. Some of the key endpoints include:
* `GET /api/stores` - List all connected retail stores and their metrics.
* `GET /api/shopper-analytics` - Retrieve active shopper demographics.
* `GET /api/inventory` - Fetch current inventory status.
* `POST /api/detection/ingest` - Push local YOLO detection data into the centralized dashboard.
* `GET /api/reports/{rtype}?fmt=csv` - Download data in CSV format.

## 🧠 AI Behavior Detection

The `behavior_detector_v3.py` script leverages a multi-scale inference system. It analyzes skeletal keypoints (using `yolov8n-pose.pt`) to determine physical posture and motion trajectories. Advanced logic is used to track wrist distances relative to the camera to detect "picking" behavior (when a shopper reaches out to grab an item from a shelf).

---
*Built for SIH 2026*
