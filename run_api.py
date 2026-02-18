"""
Run the FastAPI server.

Usage:
    python run_api.py
"""

import uvicorn
from src.config import API_HOST, API_PORT


def main():
    print("=" * 60)
    print("  🔒 FRAUD DETECTION API — Starting …")
    print(f"  📡 http://{API_HOST}:{API_PORT}")
    print(f"  📚 Docs:  http://localhost:{API_PORT}/docs")
    print(f"  📖 ReDoc: http://localhost:{API_PORT}/redoc")
    print("=" * 60)

    uvicorn.run(
        "src.api:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
