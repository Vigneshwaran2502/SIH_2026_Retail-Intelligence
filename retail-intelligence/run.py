import uvicorn
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.main import app
if __name__ == "__main__":
    print("=" * 60)
    print("  Retail Intelligence Platform - SIH26179")
    print("  AI-Powered Retail Analytics Dashboard")
    print("=" * 60)
    print("\nStarting server...")
    print("Dashboard: http://localhost:8000")
    print("API Docs:  http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
