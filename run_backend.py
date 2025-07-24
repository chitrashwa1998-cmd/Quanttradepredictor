#!/usr/bin/env python3
"""
Backend startup script for TribexAlpha
Starts the FastAPI server with proper configuration
"""

import os
import sys
import uvicorn
from pathlib import Path

def start_backend():
    """Start the FastAPI backend server"""
    # Add project root and backend to Python path
    project_root = Path(__file__).parent
    backend_dir = project_root / "backend"
    
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(backend_dir))
    
    # Set working directory to project root (not backend)
    os.chdir(project_root)
    
    print("Starting TribexAlpha FastAPI Backend...")
    print(f"Project root: {project_root}")
    print(f"Backend directory: {backend_dir}")
    
    # Start FastAPI server
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload to prevent startup issues
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    start_backend()